"""authexample: An authenticated Flower / PyTorch app with encrypted weights."""

from typing import Dict, Iterable, Optional, Tuple

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.common import MessageType
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes
from logging import INFO

from authexample.encryption import ModelEncryption, load_encryption_key
from authexample.task import Net

app = ServerApp()


class EncryptedFedAvg(FedAvg):
    """FedAvg strategy with per-node encrypted weight transmission."""

    def __init__(
        self,
        node_encryption_keys: Dict[str, bytes],
        **kwargs,
    ):
        """Initialize encrypted FedAvg strategy.

        Parameters
        ----------
        node_encryption_keys : Dict[str, bytes]
            Mapping of string node IDs to their encryption keys
        **kwargs
            Additional arguments for FedAvg
        """
        super().__init__(**kwargs)
        self.node_encryption_keys = node_encryption_keys
        self.encryptors = {
            node_id: ModelEncryption(key)
            for node_id, key in node_encryption_keys.items()
        }
        # Mapping from Grid integer node IDs to string node IDs
        self.grid_id_to_node_id: Dict[int, str] = {}
        log(
            INFO,
            f"[Server] Initialized encryption for {len(self.encryptors)} nodes",
        )

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure training with encrypted weights sent to each node."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []

        # Sample nodes from Grid
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            f"[Server] Round {server_round}: Sampled {len(node_ids)} nodes (out of {len(num_total)})",
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Get the state dict from arrays
        state_dict = arrays.to_torch_state_dict()

        # Create encrypted messages for each sampled node
        messages = []
        for grid_node_id in node_ids:
            # Check if we have a mapping for this Grid ID
            if grid_node_id in self.grid_id_to_node_id:
                # Use per-node encryption
                node_id_str = self.grid_id_to_node_id[grid_node_id]
                encryptor = self.encryptors[node_id_str]
                log(INFO, f"[Server] Encrypting weights for node {node_id_str} (Grid ID: {grid_node_id})")
                encrypted_weights = encryptor.encrypt_state_dict(state_dict)
                encrypted_arrays = ArrayRecord([encrypted_weights])
            else:
                # First round - send unencrypted (we'll learn the mapping in aggregate_train)
                log(INFO, f"[Server] Sending unencrypted weights to Grid ID {grid_node_id} (first contact)")
                encrypted_arrays = ArrayRecord(state_dict)

            # Create message with RecordDict
            content = RecordDict({"arrays": encrypted_arrays, "config": config})
            msg = Message(
                content=content,
                message_type=MessageType.TRAIN,
                dst_node_id=grid_node_id,
            )
            messages.append(msg)

        log(INFO, f"[Server] Prepared {len(messages)} training messages")
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate training results after decrypting weights from each node."""
        log(INFO, f"[Server] Round {server_round}: Aggregating training results")

        decrypted_weights = []
        num_examples = []
        metrics_list = []

        for reply in replies:
            # Get string node ID from metrics and Grid node ID from metadata
            node_id_str = reply.content.get("metrics", MetricRecord({})).data.get(
                "node-id", "unknown"
            )
            grid_node_id = reply.metadata.src_node_id

            # Build mapping from Grid ID to string node ID
            if node_id_str != "unknown" and grid_node_id not in self.grid_id_to_node_id:
                self.grid_id_to_node_id[grid_node_id] = node_id_str
                log(INFO, f"[Server] Learned mapping: Grid ID {grid_node_id} -> {node_id_str}")

            # Decrypt weights from this specific node
            if node_id_str in self.encryptors:
                try:
                    arrays_list = reply.content["arrays"].to_numpy()
                    if len(arrays_list) > 0 and arrays_list[0].dtype.name == "uint8":
                        # Encrypted weights
                        log(INFO, f"[Server] Decrypting weights from node {node_id_str}")
                        encryptor = self.encryptors[node_id_str]
                        encrypted_array = arrays_list[0]
                        state_dict = encryptor.decrypt_state_dict(encrypted_array)
                        decrypted_weights.append(ArrayRecord(state_dict))
                    else:
                        # Unencrypted weights (first round)
                        log(INFO, f"[Server] Receiving unencrypted weights from node {node_id_str}")
                        state_dict = reply.content["arrays"].to_torch_state_dict()
                        decrypted_weights.append(ArrayRecord(state_dict))
                except Exception:
                    # Fallback: try to parse as state dict directly
                    log(INFO, f"[Server] Receiving unencrypted weights from node {node_id_str}")
                    state_dict = reply.content["arrays"].to_torch_state_dict()
                    decrypted_weights.append(ArrayRecord(state_dict))

                num_examples.append(
                    reply.content["metrics"].data.get("num-examples", 1)
                )
                metrics_list.append(reply.content["metrics"])
            else:
                log(
                    INFO,
                    f"[Server] Warning: No encryption key for node {node_id_str}, skipping",
                )

        if not decrypted_weights:
            log(INFO, "[Server] No valid weights received for aggregation")
            return None, None

        # Perform weighted aggregation
        log(INFO, f"[Server] Aggregating {len(decrypted_weights)} weight sets")
        total_examples = sum(num_examples)

        # Aggregate state dicts
        aggregated_state_dict = {}
        first_state_dict = decrypted_weights[0].to_torch_state_dict()

        for key in first_state_dict.keys():
            weighted_sum = torch.zeros_like(first_state_dict[key])
            for i, weight_record in enumerate(decrypted_weights):
                state_dict = weight_record.to_torch_state_dict()
                weight = num_examples[i] / total_examples
                weighted_sum += weight * state_dict[key]
            aggregated_state_dict[key] = weighted_sum

        # Aggregate metrics
        aggregated_metrics = {}
        if metrics_list:
            # Average train_loss weighted by num_examples
            total_loss = sum(
                m.data.get("train_loss", 0) * num_examples[i]
                for i, m in enumerate(metrics_list)
            )
            aggregated_metrics["train_loss"] = total_loss / total_examples

        log(
            INFO,
            f"[Server] Aggregation complete. Metrics: {aggregated_metrics}",
        )

        return ArrayRecord(aggregated_state_dict), MetricRecord(aggregated_metrics)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure evaluation with encrypted weights sent to each node."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes from Grid
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            f"[Server] Round {server_round}: Sampled {len(node_ids)} nodes for evaluation (out of {len(num_total)})",
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Get the state dict from arrays
        state_dict = arrays.to_torch_state_dict()

        # Create encrypted messages for each sampled node
        messages = []
        for grid_node_id in node_ids:
            # Check if we have a mapping for this Grid ID
            if grid_node_id in self.grid_id_to_node_id:
                # Use per-node encryption
                node_id_str = self.grid_id_to_node_id[grid_node_id]
                encryptor = self.encryptors[node_id_str]
                log(INFO, f"[Server] Encrypting weights for node {node_id_str} evaluation (Grid ID: {grid_node_id})")
                encrypted_weights = encryptor.encrypt_state_dict(state_dict)
                encrypted_arrays = ArrayRecord([encrypted_weights])
            else:
                # First round - send unencrypted
                log(INFO, f"[Server] Sending unencrypted weights to Grid ID {grid_node_id} for evaluation (first contact)")
                encrypted_arrays = ArrayRecord(state_dict)

            # Create message with RecordDict
            content = RecordDict({"arrays": encrypted_arrays, "config": config})
            msg = Message(
                content=content,
                message_type=MessageType.EVALUATE,
                dst_node_id=grid_node_id,
            )
            messages.append(msg)

        log(INFO, f"[Server] Prepared {len(messages)} evaluation messages")
        return messages

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation results."""
        log(INFO, f"[Server] Round {server_round}: Aggregating evaluation results")

        num_examples = []
        metrics_list = []

        for reply in replies:
            # Get string node ID from metrics and Grid node ID from metadata
            node_id_str = reply.content["metrics"].data.get("node-id", "unknown")
            grid_node_id = reply.metadata.src_node_id

            # Build mapping from Grid ID to string node ID (if not already learned)
            if node_id_str != "unknown" and grid_node_id not in self.grid_id_to_node_id:
                self.grid_id_to_node_id[grid_node_id] = node_id_str
                log(INFO, f"[Server] Learned mapping: Grid ID {grid_node_id} -> {node_id_str}")

            log(INFO, f"[Server] Received evaluation from node {node_id_str}")

            num_examples.append(reply.content["metrics"].data.get("num-examples", 1))
            metrics_list.append(reply.content["metrics"])

        if not metrics_list:
            return None

        # Aggregate metrics
        total_examples = sum(num_examples)
        aggregated_metrics = {}

        # Average eval_loss and eval_acc weighted by num_examples
        total_loss = sum(
            m.data.get("eval_loss", 0) * num_examples[i]
            for i, m in enumerate(metrics_list)
        )
        total_acc = sum(
            m.data.get("eval_acc", 0) * num_examples[i]
            for i, m in enumerate(metrics_list)
        )

        aggregated_metrics["eval_loss"] = total_loss / total_examples
        aggregated_metrics["eval_acc"] = total_acc / total_examples

        log(
            INFO,
            f"[Server] Evaluation aggregation complete. Metrics: {aggregated_metrics}",
        )

        return MetricRecord(aggregated_metrics)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp with encrypted weight transmission."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]

    log(INFO, "[Server] Loading encryption keys for all nodes...")

    # Load encryption keys for all nodes
    # In a real deployment, this would be securely managed
    node_encryption_keys = {}
    for node_id in ["node-1", "node-2"]:  # Update based on number of nodes
        key_path = f"encryption_keys/{node_id}_key.bin"
        try:
            encryption_key = load_encryption_key(key_path)
            node_encryption_keys[node_id] = encryption_key
            log(INFO, f"[Server] Loaded encryption key for {node_id}")
        except FileNotFoundError:
            log(INFO, f"[Server] Warning: Encryption key not found for {node_id}")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    log(INFO, "[Server] Initializing Encrypted FedAvg strategy...")

    # Initialize Encrypted FedAvg strategy
    strategy = EncryptedFedAvg(
        node_encryption_keys=node_encryption_keys,
        fraction_train=1.0,  # Use all available nodes for training
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,  # Minimum 2 nodes for training
        min_evaluate_nodes=2,  # Minimum 2 nodes for evaluation
        min_available_nodes=2,  # Wait for at least 2 nodes to connect
    )

    # Start strategy, run Encrypted FedAvg for `num_rounds`
    log(INFO, f"[Server] Starting encrypted federated learning for {num_rounds} rounds")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    log(INFO, "[Server] Saving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model_encrypted.pt")
    log(INFO, "[Server] Final model saved successfully")
