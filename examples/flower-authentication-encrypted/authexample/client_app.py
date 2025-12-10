"""authexample: An authenticated Flower / PyTorch app with encrypted weights."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log
from logging import INFO

from authexample.encryption import ModelEncryption, load_encryption_key
from authexample.task import Net, load_data_from_disk
from authexample.task import test as test_fn
from authexample.task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with encrypted weight transmission."""

    # Read from run config
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Get node ID and encryption key path from node config
    node_id = context.node_config["node-id"]
    encryption_key_path = context.node_config["encryption-key"]
    log(INFO, f"[Node {node_id}] Loading encryption key from {encryption_key_path}")

    # Load encryption key and initialize encryption handler
    encryption_key = load_encryption_key(encryption_key_path)
    encryptor = ModelEncryption(encryption_key)

    # Decrypt the received model weights
    log(INFO, f"[Node {node_id}] Decrypting received model weights...")

    try:
        arrays_list = msg.content["arrays"].to_numpy_ndarrays()
        # Check if we have encrypted data (numpy uint8 array)
        if len(arrays_list) > 0 and arrays_list[0].dtype.name == "uint8":
            encrypted_array = arrays_list[0]
            state_dict = encryptor.decrypt_state_dict(encrypted_array)
            log(INFO, f"[Node {node_id}] Successfully decrypted model weights")
        else:
            # Fallback for unencrypted weights
            log(INFO, f"[Node {node_id}] Received unencrypted weights, using directly")
            state_dict = msg.content["arrays"].to_torch_state_dict()
    except Exception:
        # Fallback for unencrypted weights
        log(INFO, f"[Node {node_id}] Received unencrypted weights, using directly")
        state_dict = msg.content["arrays"].to_torch_state_dict()

    # Load the model and initialize it with the decrypted weights
    model = Net()
    model.load_state_dict(state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data_from_disk(dataset_path, batch_size)

    # Call the training function
    log(INFO, f"[Node {node_id}] Training model...")
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        learning_rate,
        device,
    )
    log(INFO, f"[Node {node_id}] Training complete. Loss: {train_loss:.4f}")

    # Encrypt the updated model weights
    log(INFO, f"[Node {node_id}] Encrypting updated model weights...")
    encrypted_weights = encryptor.encrypt_state_dict(model.state_dict())
    log(INFO, f"[Node {node_id}] Model weights encrypted successfully")

    # Construct and return reply Message with encrypted weights
    model_record = ArrayRecord([encrypted_weights])
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    node_config = ConfigRecord({"node_id": node_id})
    content = RecordDict({
        "arrays": model_record, 
        "metrics": metric_record,
        "node_config": node_config,
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data with encrypted weight transmission."""

    # Get node ID and encryption key path from node config
    node_id = context.node_config["node-id"]
    encryption_key_path = context.node_config["encryption-key"]
    log(INFO, f"[Node {node_id}] Loading encryption key for evaluation...")

    # Load encryption key and initialize encryption handler
    encryption_key = load_encryption_key(encryption_key_path)
    encryptor = ModelEncryption(encryption_key)

    # Decrypt the received model weights
    log(INFO, f"[Node {node_id}] Decrypting model weights for evaluation...")

    try:
        arrays_list = msg.content["arrays"].to_numpy_ndarrays()
        # Check if we have encrypted data (numpy uint8 array)
        if len(arrays_list) > 0 and arrays_list[0].dtype.name == "uint8":
            encrypted_array = arrays_list[0]
            state_dict = encryptor.decrypt_state_dict(encrypted_array)
            log(INFO, f"[Node {node_id}] Successfully decrypted model weights")
        else:
            # Fallback for unencrypted weights
            log(INFO, f"[Node {node_id}] Received unencrypted weights, using directly")
            state_dict = msg.content["arrays"].to_torch_state_dict()
    except Exception:
        # Fallback for unencrypted weights
        log(INFO, f"[Node {node_id}] Received unencrypted weights, using directly")
        state_dict = msg.content["arrays"].to_torch_state_dict()

    # Load the model and initialize it with the decrypted weights
    model = Net()
    model.load_state_dict(state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    dataset_path = context.node_config["dataset-path"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data_from_disk(dataset_path, batch_size)

    # Call the evaluation function
    log(INFO, f"[Node {node_id}] Evaluating model...")
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )
    log(
        INFO,
        f"[Node {node_id}] Evaluation complete. Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}",
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    node_config = ConfigRecord({"node_id": node_id})
    content = RecordDict({
        "metrics": metric_record,
        "node_config": node_config,
    })
    return Message(content=content, reply_to=msg)
