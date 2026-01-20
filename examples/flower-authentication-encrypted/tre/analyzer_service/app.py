"""Flask application for the TRE Analyzer (SuperLink) service.

This microservice provides REST APIs to:
- Start/stop the Flower SuperLink server
- Register/unregister SuperNodes
- Start/stop/monitor federated learning runs
- Manage TLS certificates and authentication keys
"""

import logging
import os
import sys
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.cli_wrapper import FlowerCLI, build_superlink_command
from shared.config import FederationConfig, SuperLinkConfig
from shared.key_generator import (
    generate_encryption_key,
    generate_ssh_keypair,
    generate_tls_certificates,
)
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
superlink_config: Optional[SuperLinkConfig] = None
federation_config: Optional[FederationConfig] = None
flower_cli: Optional[FlowerCLI] = None

# Configuration from environment
APP_PATH = os.getenv("FLOWER_APP_PATH", "/app/flower-app")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CERTIFICATES_DIR = os.getenv("CERTIFICATES_DIR", os.path.join(DATA_DIR, "certificates"))
KEYS_DIR = os.getenv("KEYS_DIR", os.path.join(DATA_DIR, "keys"))
ENCRYPTION_KEYS_DIR = os.getenv("ENCRYPTION_KEYS_DIR", os.path.join(DATA_DIR, "encryption_keys"))


def api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Create a standardized API response."""
    response = {
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
    return jsonify(response), status_code


def require_superlink_running(f):
    """Decorator to check if SuperLink is running."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not process_manager.is_running("superlink"):
            return api_response(
                success=False,
                error="SuperLink is not running. Start it first with POST /superlink/start",
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
    return api_response(success=True, data={"status": "healthy", "service": "analyzer"})


@app.route("/status", methods=["GET"])
def get_status():
    """Get overall service status."""
    superlink_info = process_manager.get_status("superlink")
    return api_response(
        success=True,
        data={
            "superlink": superlink_info.to_dict() if superlink_info else None,
            "config": {
                "app_path": APP_PATH,
                "data_dir": DATA_DIR,
                "certificates_dir": CERTIFICATES_DIR,
            },
        },
    )


# =======================
# SuperLink Management
# =======================

@app.route("/superlink/start", methods=["POST"])
def start_superlink():
    """Start the Flower SuperLink server.

    Request body (optional):
    {
        "insecure": false,
        "ssl_ca_certfile": "path/to/ca.crt",
        "ssl_certfile": "path/to/server.pem",
        "ssl_keyfile": "path/to/server.key",
        "enable_supernode_auth": true,
        "fleet_api_address": "0.0.0.0:9092",
        "control_api_address": "0.0.0.0:9093",
        "database": "state.db"
    }
    """
    global superlink_config, federation_config, flower_cli

    if process_manager.is_running("superlink"):
        return api_response(
            success=False,
            error="SuperLink is already running",
            status_code=400,
        )

    # Parse configuration from request or use defaults
    data = request.get_json() or {}

    # Build configuration
    superlink_config = SuperLinkConfig(
        ssl_ca_certfile=data.get("ssl_ca_certfile", os.path.join(CERTIFICATES_DIR, "ca.crt")),
        ssl_certfile=data.get("ssl_certfile", os.path.join(CERTIFICATES_DIR, "server.pem")),
        ssl_keyfile=data.get("ssl_keyfile", os.path.join(CERTIFICATES_DIR, "server.key")),
        insecure=data.get("insecure", False),
        enable_supernode_auth=data.get("enable_supernode_auth", True),
        fleet_api_address=data.get("fleet_api_address", "0.0.0.0:9092"),
        control_api_address=data.get("control_api_address", "0.0.0.0:9093"),
        serverappio_api_address=data.get("serverappio_api_address", "0.0.0.0:9091"),
        database=data.get("database"),
        storage_dir=data.get("storage_dir"),
        flwr_dir=data.get("flwr_dir"),
        isolation=data.get("isolation", "subprocess"),
    )

    # Build command
    command = build_superlink_command(
        ssl_ca_certfile=superlink_config.ssl_ca_certfile,
        ssl_certfile=superlink_config.ssl_certfile,
        ssl_keyfile=superlink_config.ssl_keyfile,
        insecure=superlink_config.insecure,
        enable_supernode_auth=superlink_config.enable_supernode_auth,
        fleet_api_address=superlink_config.fleet_api_address,
        control_api_address=superlink_config.control_api_address,
        serverappio_api_address=superlink_config.serverappio_api_address,
        database=superlink_config.database,
        storage_dir=superlink_config.storage_dir,
        flwr_dir=superlink_config.flwr_dir,
        isolation=superlink_config.isolation,
    )

    try:
        # Start the SuperLink process
        info = process_manager.start_process(
            name="superlink",
            command=command,
            cwd=APP_PATH,
        )

        # Initialize federation config for CLI operations
        federation_config = FederationConfig(
            name=data.get("federation_name", "my-federation"),
            address=superlink_config.control_api_address.replace("0.0.0.0", "127.0.0.1"),
            root_certificates=superlink_config.ssl_ca_certfile if not superlink_config.insecure else None,
            insecure=superlink_config.insecure,
        )

        # Initialize CLI wrapper
        flower_cli = FlowerCLI(
            app_path=APP_PATH,
            federation=federation_config.name,
        )

        return api_response(
            success=True,
            data={
                "message": "SuperLink started successfully",
                "process": info.to_dict(),
                "config": {
                    "fleet_api_address": superlink_config.fleet_api_address,
                    "control_api_address": superlink_config.control_api_address,
                },
            },
        )

    except Exception as e:
        logger.error(f"Failed to start SuperLink: {e}")
        return api_response(
            success=False,
            error=str(e),
            status_code=500,
        )


@app.route("/superlink/stop", methods=["POST"])
def stop_superlink():
    """Stop the Flower SuperLink server."""
    if not process_manager.is_running("superlink"):
        return api_response(
            success=False,
            error="SuperLink is not running",
            status_code=400,
        )

    try:
        info = process_manager.stop_process("superlink", timeout=15.0)
        return api_response(
            success=True,
            data={
                "message": "SuperLink stopped successfully",
                "process": info.to_dict(),
            },
        )
    except Exception as e:
        logger.error(f"Failed to stop SuperLink: {e}")
        return api_response(
            success=False,
            error=str(e),
            status_code=500,
        )


@app.route("/superlink/status", methods=["GET"])
def get_superlink_status():
    """Get SuperLink server status."""
    info = process_manager.get_status("superlink")
    if info is None:
        return api_response(
            success=True,
            data={"status": "not_started", "message": "SuperLink has not been started"},
        )
    return api_response(success=True, data=info.to_dict())


@app.route("/superlink/logs", methods=["GET"])
def get_superlink_logs():
    """Get SuperLink server logs."""
    lines = request.args.get("lines", default=100, type=int)
    logs = process_manager.get_logs("superlink", lines)
    return api_response(success=True, data={"logs": logs})


# =======================
# SuperNode Registration
# =======================

@app.route("/supernodes", methods=["GET"])
@require_superlink_running
def list_supernodes():
    """List all registered SuperNodes."""
    if flower_cli is None:
        return api_response(success=False, error="CLI not initialized", status_code=500)

    result = flower_cli.supernode_list()
    return api_response(
        success=result.success,
        data={"output": result.stdout, "raw": result.to_dict()},
        error=result.stderr if not result.success else None,
        status_code=200 if result.success else 500,
    )


@app.route("/supernodes/register", methods=["POST"])
@require_superlink_running
def register_supernode():
    """Register a new SuperNode.

    Request body:
    {
        "public_key_path": "path/to/public_key.pub"
    }

    Or to generate a new key pair:
    {
        "node_id": "1",
        "generate_keys": true
    }
    """
    if flower_cli is None:
        return api_response(success=False, error="CLI not initialized", status_code=500)

    data = request.get_json()
    if not data:
        return api_response(success=False, error="Request body required", status_code=400)

    public_key_path = data.get("public_key_path")

    # Generate keys if requested
    if data.get("generate_keys", False):
        node_id = data.get("node_id")
        if not node_id:
            return api_response(success=False, error="node_id required when generate_keys=true", status_code=400)

        try:
            key_pair = generate_ssh_keypair(
                output_dir=KEYS_DIR,
                name=f"client_credentials_{node_id}",
            )
            public_key_path = key_pair.public_key

            # Also generate encryption key
            encryption_key_path = generate_encryption_key(
                output_path=os.path.join(ENCRYPTION_KEYS_DIR, f"node-{node_id}_key.bin"),
            )

        except Exception as e:
            return api_response(success=False, error=f"Failed to generate keys: {e}", status_code=500)

    if not public_key_path:
        return api_response(
            success=False,
            error="Either public_key_path or generate_keys with node_id required",
            status_code=400,
        )

    result = flower_cli.supernode_register(public_key_path)

    response_data = {"output": result.stdout, "raw": result.to_dict()}
    if data.get("generate_keys", False):
        response_data["keys"] = {
            "public_key": public_key_path,
            "private_key": public_key_path.replace(".pub", ""),
            "encryption_key": os.path.join(ENCRYPTION_KEYS_DIR, f"node-{data.get('node_id')}_key.bin"),
        }

    return api_response(
        success=result.success,
        data=response_data,
        error=result.stderr if not result.success else None,
        status_code=200 if result.success else 500,
    )


@app.route("/supernodes/<node_id>/unregister", methods=["POST"])
@require_superlink_running
def unregister_supernode(node_id: str):
    """Unregister a SuperNode."""
    if flower_cli is None:
        return api_response(success=False, error="CLI not initialized", status_code=500)

    result = flower_cli.supernode_unregister(node_id)
    return api_response(
        success=result.success,
        data={"output": result.stdout, "raw": result.to_dict()},
        error=result.stderr if not result.success else None,
        status_code=200 if result.success else 500,
    )


# =======================
# Run Management
# =======================

@app.route("/runs", methods=["GET"])
@require_superlink_running
def list_runs():
    """List all federated learning runs."""
    if flower_cli is None:
        return api_response(success=False, error="CLI not initialized", status_code=500)

    result = flower_cli.run_list(list_all=True)
    return api_response(
        success=result.success,
        data={"output": result.stdout, "raw": result.to_dict()},
        error=result.stderr if not result.success else None,
        status_code=200 if result.success else 500,
    )


@app.route("/runs", methods=["POST"])
@require_superlink_running
def start_run():
    """Start a new federated learning run.

    Request body (optional):
    {
        "run_config": {
            "num-server-rounds": 5,
            "local-epochs": 2
        }
    }
    """
    if flower_cli is None:
        return api_response(success=False, error="CLI not initialized", status_code=500)

    data = request.get_json() or {}
    run_config = data.get("run_config")

    result = flower_cli.run_start(run_config=run_config)
    return api_response(
        success=result.success,
        data={"output": result.stdout, "raw": result.to_dict()},
        error=result.stderr if not result.success else None,
        status_code=200 if result.success else 500,
    )


@app.route("/runs/<int:run_id>", methods=["GET"])
@require_superlink_running
def get_run(run_id: int):
    """Get details of a specific run."""
    if flower_cli is None:
        return api_response(success=False, error="CLI not initialized", status_code=500)

    result = flower_cli.run_list(run_id=run_id)
    return api_response(
        success=result.success,
        data={"output": result.stdout, "raw": result.to_dict()},
        error=result.stderr if not result.success else None,
        status_code=200 if result.success else 500,
    )


@app.route("/runs/<int:run_id>/stop", methods=["POST"])
@require_superlink_running
def stop_run(run_id: int):
    """Stop a running federated learning run."""
    if flower_cli is None:
        return api_response(success=False, error="CLI not initialized", status_code=500)

    result = flower_cli.run_stop(run_id)
    return api_response(
        success=result.success,
        data={"output": result.stdout, "raw": result.to_dict()},
        error=result.stderr if not result.success else None,
        status_code=200 if result.success else 500,
    )


@app.route("/runs/<int:run_id>/logs", methods=["GET"])
@require_superlink_running
def get_run_logs(run_id: int):
    """Get logs from a federated learning run."""
    if flower_cli is None:
        return api_response(success=False, error="CLI not initialized", status_code=500)

    result = flower_cli.run_logs(run_id)
    return api_response(
        success=result.success,
        data={"output": result.stdout, "raw": result.to_dict()},
        error=result.stderr if not result.success else None,
        status_code=200 if result.success else 500,
    )


# =======================
# Certificate Management
# =======================

@app.route("/certificates/generate", methods=["POST"])
def generate_certificates():
    """Generate TLS certificates for secure communication.

    Request body (optional):
    {
        "days_valid": 365,
        "key_size": 4096
    }
    """
    data = request.get_json() or {}

    try:
        certs = generate_tls_certificates(
            output_dir=CERTIFICATES_DIR,
            days_valid=data.get("days_valid", 365),
            key_size=data.get("key_size", 4096),
        )

        return api_response(
            success=True,
            data={
                "message": "Certificates generated successfully",
                "certificates": {
                    "ca_cert": certs.ca_cert,
                    "ca_key": certs.ca_key,
                    "server_cert": certs.server_cert,
                    "server_key": certs.server_key,
                },
            },
        )
    except Exception as e:
        logger.error(f"Failed to generate certificates: {e}")
        return api_response(success=False, error=str(e), status_code=500)


@app.route("/certificates", methods=["GET"])
def list_certificates():
    """List available certificates."""
    certs = {}
    for filename in ["ca.crt", "ca.key", "server.pem", "server.key"]:
        filepath = os.path.join(CERTIFICATES_DIR, filename)
        certs[filename] = {
            "exists": os.path.exists(filepath),
            "path": filepath,
        }
    return api_response(success=True, data={"certificates": certs})


# =======================
# Key Management
# =======================

@app.route("/keys/generate", methods=["POST"])
def generate_node_keys():
    """Generate authentication and encryption keys for a node.

    Request body:
    {
        "node_id": "1"
    }
    """
    data = request.get_json()
    if not data or "node_id" not in data:
        return api_response(success=False, error="node_id required", status_code=400)

    node_id = data["node_id"]

    try:
        # Generate SSH key pair for authentication
        key_pair = generate_ssh_keypair(
            output_dir=KEYS_DIR,
            name=f"client_credentials_{node_id}",
        )

        # Generate encryption key
        encryption_key_path = generate_encryption_key(
            output_path=os.path.join(ENCRYPTION_KEYS_DIR, f"node-{node_id}_key.bin"),
        )

        return api_response(
            success=True,
            data={
                "message": f"Keys generated for node {node_id}",
                "keys": {
                    "private_key": key_pair.private_key,
                    "public_key": key_pair.public_key,
                    "encryption_key": encryption_key_path,
                },
            },
        )
    except Exception as e:
        logger.error(f"Failed to generate keys: {e}")
        return api_response(success=False, error=str(e), status_code=500)


@app.route("/keys", methods=["GET"])
def list_keys():
    """List available keys."""
    auth_keys = []
    encryption_keys = []

    if os.path.exists(KEYS_DIR):
        for filename in os.listdir(KEYS_DIR):
            if filename.endswith(".pub"):
                auth_keys.append({
                    "name": filename,
                    "path": os.path.join(KEYS_DIR, filename),
                    "private_key_path": os.path.join(KEYS_DIR, filename.replace(".pub", "")),
                })

    if os.path.exists(ENCRYPTION_KEYS_DIR):
        for filename in os.listdir(ENCRYPTION_KEYS_DIR):
            if filename.endswith(".bin"):
                encryption_keys.append({
                    "name": filename,
                    "path": os.path.join(ENCRYPTION_KEYS_DIR, filename),
                })

    return api_response(
        success=True,
        data={
            "auth_keys": auth_keys,
            "encryption_keys": encryption_keys,
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
    os.makedirs(CERTIFICATES_DIR, exist_ok=True)
    os.makedirs(KEYS_DIR, exist_ok=True)
    os.makedirs(ENCRYPTION_KEYS_DIR, exist_ok=True)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
