"""Flask application for the TRE Data Owner (SuperNode) service.

This microservice provides REST APIs to:
- Start/stop the Flower SuperNode client
- Configure node settings (dataset path, encryption keys)
- Monitor training progress and status
- Manage local data and model artifacts
"""

import logging
import os
import shutil
import sys
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_file

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.cli_wrapper import build_supernode_command
from shared.config import SuperNodeConfig
from shared.process_manager import ProcessManager, ProcessStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global state
process_manager = ProcessManager()
supernode_config: Optional[SuperNodeConfig] = None

# Configuration from environment
NODE_ID = os.getenv("NODE_ID", "1")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
DATASET_PATH = os.getenv("DATASET_PATH", os.path.join(DATA_DIR, "datasets", f"cifar10_part_{NODE_ID}"))
KEYS_DIR = os.getenv("KEYS_DIR", os.path.join(DATA_DIR, "keys"))
ENCRYPTION_KEYS_DIR = os.getenv("ENCRYPTION_KEYS_DIR", os.path.join(DATA_DIR, "encryption_keys"))
CERTIFICATES_DIR = os.getenv("CERTIFICATES_DIR", os.path.join(DATA_DIR, "certificates"))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(DATA_DIR, "models"))

# SuperLink connection settings (from environment)
SUPERLINK_ADDRESS = os.getenv("SUPERLINK_ADDRESS", "superlink:9092")


def api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Create a standardized API response."""
    response = {
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "node_id": NODE_ID,
    }
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
    return jsonify(response), status_code


def require_supernode_running(f):
    """Decorator to check if SuperNode is running."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not process_manager.is_running("supernode"):
            return api_response(
                success=False,
                error="SuperNode is not running. Start it first with POST /supernode/start",
                status_code=400,
            )
        return f(*args, **kwargs)
    return decorated_function


# =======================
# Health & Status Routes
# =======================

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return api_response(
        success=True,
        data={"status": "healthy", "service": "dataowner", "node_id": NODE_ID},
    )


@app.route("/status", methods=["GET"])
def get_status():
    """Get overall service status."""
    supernode_info = process_manager.get_status("supernode")

    # Check if dataset exists
    dataset_exists = os.path.exists(DATASET_PATH)
    dataset_info = None
    if dataset_exists:
        data_file = os.path.join(DATASET_PATH, "data.pt")
        dataset_info = {
            "path": DATASET_PATH,
            "exists": True,
            "data_file_exists": os.path.exists(data_file),
        }

    # Check if keys exist
    private_key_path = os.path.join(KEYS_DIR, f"client_credentials_{NODE_ID}")
    encryption_key_path = os.path.join(ENCRYPTION_KEYS_DIR, f"node-{NODE_ID}_key.bin")

    keys_info = {
        "private_key_exists": os.path.exists(private_key_path),
        "encryption_key_exists": os.path.exists(encryption_key_path),
    }

    return api_response(
        success=True,
        data={
            "supernode": supernode_info.to_dict() if supernode_info else None,
            "dataset": dataset_info,
            "keys": keys_info,
            "config": {
                "node_id": NODE_ID,
                "data_dir": DATA_DIR,
                "superlink_address": SUPERLINK_ADDRESS,
            },
        },
    )


# =======================
# SuperNode Management
# =======================

@app.route("/supernode/start", methods=["POST"])
def start_supernode():
    """Start the Flower SuperNode client.

    Request body (optional):
    {
        "superlink_address": "superlink:9092",
        "insecure": false,
        "root_certificates": "path/to/ca.crt",
        "auth_private_key": "path/to/private_key",
        "node_config": {
            "dataset-path": "path/to/dataset",
            "node-id": "node-1",
            "encryption-key": "path/to/key.bin"
        },
        "clientappio_api_address": "0.0.0.0:9094"
    }
    """
    global supernode_config

    if process_manager.is_running("supernode"):
        return api_response(
            success=False,
            error="SuperNode is already running",
            status_code=400,
        )

    # Parse configuration from request or use defaults
    data = request.get_json() or {}

    # Default node config
    default_node_config = {
        "dataset-path": DATASET_PATH,
        "node-id": f"node-{NODE_ID}",
        "encryption-key": os.path.join(ENCRYPTION_KEYS_DIR, f"node-{NODE_ID}_key.bin"),
    }

    # Merge with provided node config
    node_config = {**default_node_config, **data.get("node_config", {})}

    # Build configuration
    supernode_config = SuperNodeConfig(
        superlink_address=data.get("superlink_address", SUPERLINK_ADDRESS),
        insecure=data.get("insecure", False),
        root_certificates=data.get(
            "root_certificates",
            os.path.join(CERTIFICATES_DIR, "ca.crt") if not data.get("insecure", False) else None,
        ),
        auth_supernode_private_key=data.get(
            "auth_private_key",
            os.path.join(KEYS_DIR, f"client_credentials_{NODE_ID}"),
        ),
        node_config=node_config,
        clientappio_api_address=data.get("clientappio_api_address", "0.0.0.0:9094"),
        max_retries=data.get("max_retries"),
        max_wait_time=data.get("max_wait_time"),
        isolation=data.get("isolation", "subprocess"),
        flwr_dir=data.get("flwr_dir"),
    )

    # Validate required files exist
    validation_errors = []

    if not supernode_config.insecure and supernode_config.root_certificates:
        if not os.path.exists(supernode_config.root_certificates):
            validation_errors.append(f"Root certificates not found: {supernode_config.root_certificates}")

    if supernode_config.auth_supernode_private_key:
        if not os.path.exists(supernode_config.auth_supernode_private_key):
            validation_errors.append(f"Private key not found: {supernode_config.auth_supernode_private_key}")

    if not os.path.exists(node_config.get("dataset-path", "")):
        validation_errors.append(f"Dataset not found: {node_config.get('dataset-path')}")

    if not os.path.exists(node_config.get("encryption-key", "")):
        validation_errors.append(f"Encryption key not found: {node_config.get('encryption-key')}")

    if validation_errors:
        return api_response(
            success=False,
            error="Validation failed: " + "; ".join(validation_errors),
            status_code=400,
        )

    # Build command
    command = build_supernode_command(
        superlink_address=supernode_config.superlink_address,
        root_certificates=supernode_config.root_certificates,
        insecure=supernode_config.insecure,
        auth_supernode_private_key=supernode_config.auth_supernode_private_key,
        node_config=supernode_config.node_config,
        clientappio_api_address=supernode_config.clientappio_api_address,
        flwr_dir=supernode_config.flwr_dir,
        isolation=supernode_config.isolation,
        max_retries=supernode_config.max_retries,
        max_wait_time=supernode_config.max_wait_time,
    )

    try:
        # Start the SuperNode process
        info = process_manager.start_process(
            name="supernode",
            command=command,
        )

        return api_response(
            success=True,
            data={
                "message": "SuperNode started successfully",
                "process": info.to_dict(),
                "config": {
                    "superlink_address": supernode_config.superlink_address,
                    "node_config": supernode_config.node_config,
                },
            },
        )

    except Exception as e:
        logger.error(f"Failed to start SuperNode: {e}")
        return api_response(
            success=False,
            error=str(e),
            status_code=500,
        )


@app.route("/supernode/stop", methods=["POST"])
def stop_supernode():
    """Stop the Flower SuperNode client."""
    if not process_manager.is_running("supernode"):
        return api_response(
            success=False,
            error="SuperNode is not running",
            status_code=400,
        )

    try:
        info = process_manager.stop_process("supernode", timeout=15.0)
        return api_response(
            success=True,
            data={
                "message": "SuperNode stopped successfully",
                "process": info.to_dict(),
            },
        )
    except Exception as e:
        logger.error(f"Failed to stop SuperNode: {e}")
        return api_response(
            success=False,
            error=str(e),
            status_code=500,
        )


@app.route("/supernode/status", methods=["GET"])
def get_supernode_status():
    """Get SuperNode client status."""
    info = process_manager.get_status("supernode")
    if info is None:
        return api_response(
            success=True,
            data={"status": "not_started", "message": "SuperNode has not been started"},
        )
    return api_response(success=True, data=info.to_dict())


@app.route("/supernode/logs", methods=["GET"])
def get_supernode_logs():
    """Get SuperNode client logs."""
    lines = request.args.get("lines", default=100, type=int)
    logs = process_manager.get_logs("supernode", lines)
    return api_response(success=True, data={"logs": logs})


# =======================
# Dataset Management
# =======================

@app.route("/dataset", methods=["GET"])
def get_dataset_info():
    """Get information about the local dataset."""
    if not os.path.exists(DATASET_PATH):
        return api_response(
            success=True,
            data={
                "exists": False,
                "path": DATASET_PATH,
                "message": "Dataset not found. Upload or prepare dataset first.",
            },
        )

    # Get dataset file info
    data_file = os.path.join(DATASET_PATH, "data.pt")
    if os.path.exists(data_file):
        file_stat = os.stat(data_file)
        file_info = {
            "size_bytes": file_stat.st_size,
            "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        }
    else:
        file_info = None

    return api_response(
        success=True,
        data={
            "exists": True,
            "path": DATASET_PATH,
            "data_file": file_info,
        },
    )


@app.route("/dataset/upload", methods=["POST"])
def upload_dataset():
    """Upload a dataset file.

    Expects multipart/form-data with a 'file' field containing data.pt
    """
    if "file" not in request.files:
        return api_response(success=False, error="No file provided", status_code=400)

    file = request.files["file"]
    if file.filename == "":
        return api_response(success=False, error="No file selected", status_code=400)

    try:
        # Create dataset directory
        os.makedirs(DATASET_PATH, exist_ok=True)

        # Save the file
        filepath = os.path.join(DATASET_PATH, "data.pt")
        file.save(filepath)

        return api_response(
            success=True,
            data={
                "message": "Dataset uploaded successfully",
                "path": filepath,
            },
        )
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        return api_response(success=False, error=str(e), status_code=500)


@app.route("/dataset/download", methods=["GET"])
def download_dataset():
    """Download the current dataset file."""
    data_file = os.path.join(DATASET_PATH, "data.pt")
    if not os.path.exists(data_file):
        return api_response(
            success=False,
            error="Dataset file not found",
            status_code=404,
        )

    return send_file(
        data_file,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"data_node_{NODE_ID}.pt",
    )


# =======================
# Key Management
# =======================

@app.route("/keys", methods=["GET"])
def get_keys_info():
    """Get information about available keys."""
    private_key_path = os.path.join(KEYS_DIR, f"client_credentials_{NODE_ID}")
    public_key_path = f"{private_key_path}.pub"
    encryption_key_path = os.path.join(ENCRYPTION_KEYS_DIR, f"node-{NODE_ID}_key.bin")

    return api_response(
        success=True,
        data={
            "auth_keys": {
                "private_key": {
                    "path": private_key_path,
                    "exists": os.path.exists(private_key_path),
                },
                "public_key": {
                    "path": public_key_path,
                    "exists": os.path.exists(public_key_path),
                },
            },
            "encryption_key": {
                "path": encryption_key_path,
                "exists": os.path.exists(encryption_key_path),
            },
        },
    )


@app.route("/keys/upload/auth", methods=["POST"])
def upload_auth_keys():
    """Upload authentication key files.

    Expects multipart/form-data with:
    - 'private_key': Private key file
    - 'public_key': Public key file (optional, will be derived if not provided)
    """
    if "private_key" not in request.files:
        return api_response(success=False, error="private_key file required", status_code=400)

    try:
        os.makedirs(KEYS_DIR, exist_ok=True)

        private_key_path = os.path.join(KEYS_DIR, f"client_credentials_{NODE_ID}")
        request.files["private_key"].save(private_key_path)
        os.chmod(private_key_path, 0o600)

        if "public_key" in request.files:
            public_key_path = f"{private_key_path}.pub"
            request.files["public_key"].save(public_key_path)

        return api_response(
            success=True,
            data={
                "message": "Authentication keys uploaded successfully",
                "private_key": private_key_path,
            },
        )
    except Exception as e:
        logger.error(f"Failed to upload auth keys: {e}")
        return api_response(success=False, error=str(e), status_code=500)


@app.route("/keys/upload/encryption", methods=["POST"])
def upload_encryption_key():
    """Upload encryption key file.

    Expects multipart/form-data with:
    - 'encryption_key': Encryption key file (.bin)
    """
    if "encryption_key" not in request.files:
        return api_response(success=False, error="encryption_key file required", status_code=400)

    try:
        os.makedirs(ENCRYPTION_KEYS_DIR, exist_ok=True)

        encryption_key_path = os.path.join(ENCRYPTION_KEYS_DIR, f"node-{NODE_ID}_key.bin")
        request.files["encryption_key"].save(encryption_key_path)
        os.chmod(encryption_key_path, 0o600)

        return api_response(
            success=True,
            data={
                "message": "Encryption key uploaded successfully",
                "encryption_key": encryption_key_path,
            },
        )
    except Exception as e:
        logger.error(f"Failed to upload encryption key: {e}")
        return api_response(success=False, error=str(e), status_code=500)


@app.route("/keys/download/public", methods=["GET"])
def download_public_key():
    """Download the public key for registration with SuperLink."""
    public_key_path = os.path.join(KEYS_DIR, f"client_credentials_{NODE_ID}.pub")
    if not os.path.exists(public_key_path):
        return api_response(
            success=False,
            error="Public key not found",
            status_code=404,
        )

    return send_file(
        public_key_path,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"client_credentials_{NODE_ID}.pub",
    )


# =======================
# Certificate Management
# =======================

@app.route("/certificates", methods=["GET"])
def get_certificates_info():
    """Get information about available certificates."""
    ca_cert_path = os.path.join(CERTIFICATES_DIR, "ca.crt")

    return api_response(
        success=True,
        data={
            "ca_cert": {
                "path": ca_cert_path,
                "exists": os.path.exists(ca_cert_path),
            },
        },
    )


@app.route("/certificates/upload/ca", methods=["POST"])
def upload_ca_certificate():
    """Upload CA certificate for TLS verification.

    Expects multipart/form-data with:
    - 'ca_cert': CA certificate file
    """
    if "ca_cert" not in request.files:
        return api_response(success=False, error="ca_cert file required", status_code=400)

    try:
        os.makedirs(CERTIFICATES_DIR, exist_ok=True)

        ca_cert_path = os.path.join(CERTIFICATES_DIR, "ca.crt")
        request.files["ca_cert"].save(ca_cert_path)

        return api_response(
            success=True,
            data={
                "message": "CA certificate uploaded successfully",
                "ca_cert": ca_cert_path,
            },
        )
    except Exception as e:
        logger.error(f"Failed to upload CA certificate: {e}")
        return api_response(success=False, error=str(e), status_code=500)


# =======================
# Model Artifacts
# =======================

@app.route("/models", methods=["GET"])
def list_models():
    """List saved model artifacts."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    models = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pt"):
            filepath = os.path.join(MODELS_DIR, filename)
            file_stat = os.stat(filepath)
            models.append({
                "name": filename,
                "path": filepath,
                "size_bytes": file_stat.st_size,
                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            })

    return api_response(success=True, data={"models": models})


@app.route("/models/<filename>", methods=["GET"])
def download_model(filename: str):
    """Download a model artifact."""
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        return api_response(success=False, error="Model not found", status_code=404)

    return send_file(
        filepath,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=filename,
    )


# =======================
# Configuration
# =======================

@app.route("/config", methods=["GET"])
def get_config():
    """Get current node configuration."""
    return api_response(
        success=True,
        data={
            "node_id": NODE_ID,
            "data_dir": DATA_DIR,
            "dataset_path": DATASET_PATH,
            "keys_dir": KEYS_DIR,
            "encryption_keys_dir": ENCRYPTION_KEYS_DIR,
            "certificates_dir": CERTIFICATES_DIR,
            "models_dir": MODELS_DIR,
            "superlink_address": SUPERLINK_ADDRESS,
            "supernode_config": supernode_config.__dict__ if supernode_config else None,
        },
    )


# =======================
# Shutdown Handler
# =======================

@app.teardown_appcontext
def shutdown(exception=None):
    """Clean up on shutdown."""
    process_manager.stop_all()


def create_app():
    """Application factory."""
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(KEYS_DIR, exist_ok=True)
    os.makedirs(ENCRYPTION_KEYS_DIR, exist_ok=True)
    os.makedirs(CERTIFICATES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)
