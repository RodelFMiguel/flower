#!/usr/bin/env python3
"""Setup script to prepare the TRE environment.

This script:
1. Generates TLS certificates
2. Generates authentication keys for SuperNodes
3. Generates encryption keys for secure weight transmission
4. Prepares dataset partitions
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.key_generator import (
    generate_encryption_key,
    generate_ssh_keypair,
    generate_tls_certificates,
)


def setup_directories(base_dir: str, num_nodes: int = 2):
    """Create necessary directories."""
    dirs = [
        os.path.join(base_dir, "certificates"),
        os.path.join(base_dir, "keys"),
        os.path.join(base_dir, "encryption_keys"),
    ]

    for i in range(1, num_nodes + 1):
        dirs.append(os.path.join(base_dir, "datasets", f"cifar10_part_{i}"))

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

    return dirs


def generate_all_certificates(base_dir: str):
    """Generate TLS certificates."""
    print("\n=== Generating TLS Certificates ===")
    certs_dir = os.path.join(base_dir, "certificates")

    # Create certificate config
    config_content = """[req]
default_bits = 4096
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn

[dn]
C = US
ST = CA
O = TRE Server
CN = localhost

[req_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = analyzer
DNS.3 = superlink
DNS.4 = dataowner_1
DNS.5 = dataowner_2
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
"""
    config_file = os.path.join(certs_dir, "certificate.conf")
    with open(config_file, "w") as f:
        f.write(config_content)

    certs = generate_tls_certificates(
        output_dir=certs_dir,
        config_file=config_file,
        days_valid=365,
    )

    print(f"  CA Certificate: {certs.ca_cert}")
    print(f"  Server Certificate: {certs.server_cert}")
    print(f"  Server Key: {certs.server_key}")

    return certs


def generate_all_keys(base_dir: str, num_nodes: int = 2):
    """Generate authentication and encryption keys for all nodes."""
    print(f"\n=== Generating Keys for {num_nodes} Nodes ===")

    keys_dir = os.path.join(base_dir, "keys")
    encryption_dir = os.path.join(base_dir, "encryption_keys")

    keys_info = []

    for i in range(1, num_nodes + 1):
        print(f"\n  Node {i}:")

        # Generate authentication key pair
        key_pair = generate_ssh_keypair(
            output_dir=keys_dir,
            name=f"client_credentials_{i}",
        )
        print(f"    Auth Private Key: {key_pair.private_key}")
        print(f"    Auth Public Key: {key_pair.public_key}")

        # Generate encryption key
        encryption_key_path = generate_encryption_key(
            output_path=os.path.join(encryption_dir, f"node-{i}_key.bin"),
        )
        print(f"    Encryption Key: {encryption_key_path}")

        keys_info.append({
            "node_id": i,
            "private_key": key_pair.private_key,
            "public_key": key_pair.public_key,
            "encryption_key": encryption_key_path,
        })

    return keys_info


def prepare_datasets(base_dir: str, num_nodes: int = 2):
    """Prepare dataset partitions using the parent example's prepare script."""
    print(f"\n=== Preparing Datasets for {num_nodes} Nodes ===")

    # Check if we're in the tre directory or example root
    example_root = Path(__file__).parent.parent.parent
    prepare_script = example_root / "prepare_dataset_simple.py"

    if not prepare_script.exists():
        print(f"  Warning: Dataset preparation script not found at {prepare_script}")
        print("  You will need to prepare datasets manually.")
        return False

    # Change to example root and run the script
    original_dir = os.getcwd()
    try:
        os.chdir(example_root)

        # Modify the script to output to tre/data/datasets
        datasets_dir = os.path.join(base_dir, "datasets")

        result = subprocess.run(
            [sys.executable, str(prepare_script)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  Error preparing datasets: {result.stderr}")
            return False

        print(result.stdout)

        # Move datasets to tre/data/datasets if needed
        example_datasets = example_root / "datasets"
        if example_datasets.exists():
            for i in range(1, num_nodes + 1):
                src = example_datasets / f"cifar10_part_{i}"
                dst = Path(datasets_dir) / f"cifar10_part_{i}"
                if src.exists() and not dst.exists():
                    shutil.copytree(src, dst)
                    print(f"  Copied dataset partition {i} to {dst}")

        return True

    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Setup TRE environment for Flower federated learning"
    )
    parser.add_argument(
        "--base-dir",
        default="./data",
        help="Base directory for data files (default: ./data)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=2,
        help="Number of data owner nodes (default: 2)",
    )
    parser.add_argument(
        "--skip-certs",
        action="store_true",
        help="Skip certificate generation",
    )
    parser.add_argument(
        "--skip-keys",
        action="store_true",
        help="Skip key generation",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset preparation",
    )

    args = parser.parse_args()

    # Convert to absolute path
    base_dir = os.path.abspath(args.base_dir)

    print(f"Setting up TRE environment in: {base_dir}")
    print(f"Number of nodes: {args.num_nodes}")

    # Setup directories
    setup_directories(base_dir, args.num_nodes)

    # Generate certificates
    if not args.skip_certs:
        generate_all_certificates(base_dir)
    else:
        print("\n=== Skipping Certificate Generation ===")

    # Generate keys
    if not args.skip_keys:
        generate_all_keys(base_dir, args.num_nodes)
    else:
        print("\n=== Skipping Key Generation ===")

    # Prepare datasets
    if not args.skip_datasets:
        prepare_datasets(base_dir, args.num_nodes)
    else:
        print("\n=== Skipping Dataset Preparation ===")

    print("\n=== Setup Complete ===")
    print(f"\nData directory: {base_dir}")
    print("\nNext steps:")
    print("  1. Review generated certificates and keys")
    print("  2. Start services with: docker-compose up -d")
    print("  3. Use orchestrate.py to run federated learning")


if __name__ == "__main__":
    main()
