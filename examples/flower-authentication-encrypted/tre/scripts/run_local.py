#!/usr/bin/env python3
"""Run TRE services locally without Docker for development/testing.

This script starts the Flask services directly for local development.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
TRE_DIR = SCRIPT_DIR.parent
EXAMPLE_DIR = TRE_DIR.parent
DATA_DIR = TRE_DIR / "data"

# Global list to track processes
processes = []


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("\nShutting down services...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error terminating process: {e}")
            proc.kill()
    sys.exit(0)


def setup_environment():
    """Create necessary directories."""
    dirs = [
        DATA_DIR / "certificates",
        DATA_DIR / "keys",
        DATA_DIR / "encryption_keys",
        DATA_DIR / "datasets" / "cifar10_part_1",
        DATA_DIR / "datasets" / "cifar10_part_2",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR}")


def start_analyzer_service(port: int = 5000):
    """Start the analyzer Flask service."""
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(TRE_DIR),
        "FLOWER_APP_PATH": str(EXAMPLE_DIR),
        "DATA_DIR": str(DATA_DIR),
        "CERTIFICATES_DIR": str(DATA_DIR / "certificates"),
        "KEYS_DIR": str(DATA_DIR / "keys"),
        "ENCRYPTION_KEYS_DIR": str(DATA_DIR / "encryption_keys"),
        "FLASK_APP": "analyzer_service.app:create_app",
        "FLASK_ENV": "development",
    })

    cmd = [
        sys.executable, "-m", "flask", "run",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]

    logger.info(f"Starting Analyzer service on port {port}...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(TRE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    processes.append(proc)
    return proc


def start_dataowner_service(node_id: int, port: int):
    """Start a data owner Flask service."""
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(TRE_DIR),
        "NODE_ID": str(node_id),
        "DATA_DIR": str(DATA_DIR),
        "DATASET_PATH": str(DATA_DIR / "datasets" / f"cifar10_part_{node_id}"),
        "KEYS_DIR": str(DATA_DIR / "keys"),
        "ENCRYPTION_KEYS_DIR": str(DATA_DIR / "encryption_keys"),
        "CERTIFICATES_DIR": str(DATA_DIR / "certificates"),
        "MODELS_DIR": str(DATA_DIR / "models"),
        "SUPERLINK_ADDRESS": "127.0.0.1:9092",
        "FLASK_APP": "dataowner_service.app:create_app",
        "FLASK_ENV": "development",
    })

    cmd = [
        sys.executable, "-m", "flask", "run",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]

    logger.info(f"Starting Data Owner {node_id} service on port {port}...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(TRE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    processes.append(proc)
    return proc


def log_output(proc, name: str):
    """Print process output."""
    for line in proc.stdout:
        print(f"[{name}] {line.decode().rstrip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Run TRE services locally for development"
    )
    parser.add_argument(
        "--analyzer-port",
        type=int,
        default=5000,
        help="Analyzer service port (default: 5000)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=2,
        help="Number of data owner nodes (default: 2)",
    )
    parser.add_argument(
        "--dataowner-base-port",
        type=int,
        default=5001,
        help="Base port for data owner services (default: 5001)",
    )
    parser.add_argument(
        "--analyzer-only",
        action="store_true",
        help="Only start the analyzer service",
    )
    parser.add_argument(
        "--dataowner-only",
        type=int,
        metavar="NODE_ID",
        help="Only start a specific data owner service",
    )

    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup environment
    setup_environment()

    # Start services
    if args.dataowner_only:
        # Start only a specific data owner
        port = args.dataowner_base_port + args.dataowner_only - 1
        start_dataowner_service(args.dataowner_only, port)
        logger.info(f"\nData Owner {args.dataowner_only} service running at http://localhost:{port}")
    elif args.analyzer_only:
        # Start only the analyzer
        start_analyzer_service(args.analyzer_port)
        logger.info(f"\nAnalyzer service running at http://localhost:{args.analyzer_port}")
    else:
        # Start all services
        start_analyzer_service(args.analyzer_port)

        for i in range(1, args.num_nodes + 1):
            port = args.dataowner_base_port + i - 1
            start_dataowner_service(i, port)
            time.sleep(1)  # Small delay between starts

        logger.info("\n" + "=" * 60)
        logger.info("All services started!")
        logger.info("=" * 60)
        logger.info(f"\nAnalyzer service:     http://localhost:{args.analyzer_port}")
        for i in range(1, args.num_nodes + 1):
            port = args.dataowner_base_port + i - 1
            logger.info(f"Data Owner {i} service: http://localhost:{port}")
        logger.info("\nPress Ctrl+C to stop all services")

    # Wait for processes
    try:
        while True:
            # Check if any process has died
            for proc in processes:
                if proc.poll() is not None:
                    logger.warning(f"Process {proc.pid} exited with code {proc.returncode}")
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
