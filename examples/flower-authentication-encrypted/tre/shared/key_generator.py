"""Key generation utilities for Flower authentication and encryption."""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class KeyPair:
    """SSH key pair paths."""

    private_key: str
    public_key: str


@dataclass
class TLSCertificates:
    """TLS certificate file paths."""

    ca_cert: str
    ca_key: str
    server_cert: str
    server_key: str


def generate_ssh_keypair(
    output_dir: str,
    name: str,
    key_type: str = "ecdsa",
    bits: int = 384,
) -> KeyPair:
    """Generate an SSH key pair for SuperNode authentication.

    Args:
        output_dir: Directory to store the keys
        name: Base name for the key files
        key_type: Key type (ecdsa recommended for Flower)
        bits: Key size in bits

    Returns:
        KeyPair with paths to private and public keys
    """
    os.makedirs(output_dir, exist_ok=True)

    private_key_path = os.path.join(output_dir, name)
    public_key_path = f"{private_key_path}.pub"

    # Remove existing keys if any
    if os.path.exists(private_key_path):
        os.remove(private_key_path)
    if os.path.exists(public_key_path):
        os.remove(public_key_path)

    # Generate the key pair
    cmd = [
        "ssh-keygen",
        "-t", key_type,
        "-b", str(bits),
        "-N", "",  # Empty passphrase
        "-f", private_key_path,
        "-C", "",  # Empty comment
    ]

    logger.info(f"Generating SSH key pair: {name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate SSH key pair: {result.stderr}")

    logger.info(f"Generated key pair: {private_key_path}")

    return KeyPair(private_key=private_key_path, public_key=public_key_path)


def generate_encryption_key(
    output_path: str,
    key_size: int = 32,
) -> str:
    """Generate a symmetric encryption key for model weight encryption.

    Args:
        output_path: Path to store the key
        key_size: Key size in bytes (32 for AES-256)

    Returns:
        Path to the generated key file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate random bytes
    key = os.urandom(key_size)

    with open(output_path, "wb") as f:
        f.write(key)

    logger.info(f"Generated encryption key: {output_path}")
    return output_path


def generate_tls_certificates(
    output_dir: str,
    config_file: Optional[str] = None,
    days_valid: int = 365,
    key_size: int = 4096,
) -> TLSCertificates:
    """Generate TLS certificates for secure communication.

    Args:
        output_dir: Directory to store certificates
        config_file: OpenSSL configuration file (optional)
        days_valid: Certificate validity in days
        key_size: RSA key size in bits

    Returns:
        TLSCertificates with paths to all certificate files
    """
    os.makedirs(output_dir, exist_ok=True)

    ca_key = os.path.join(output_dir, "ca.key")
    ca_cert = os.path.join(output_dir, "ca.crt")
    server_key = os.path.join(output_dir, "server.key")
    server_csr = os.path.join(output_dir, "server.csr")
    server_cert = os.path.join(output_dir, "server.pem")

    # Generate CA key
    logger.info("Generating CA key...")
    subprocess.run(
        ["openssl", "genrsa", "-out", ca_key, str(key_size)],
        check=True,
        capture_output=True,
    )

    # Generate CA certificate
    logger.info("Generating CA certificate...")
    subprocess.run(
        [
            "openssl", "req",
            "-new", "-x509",
            "-key", ca_key,
            "-sha256",
            "-subj", "/C=US/ST=CA/O=TRE CA, Inc.",
            "-days", str(days_valid),
            "-out", ca_cert,
        ],
        check=True,
        capture_output=True,
    )

    # Generate server key
    logger.info("Generating server key...")
    subprocess.run(
        ["openssl", "genrsa", "-out", server_key, str(key_size)],
        check=True,
        capture_output=True,
    )

    # Create default config if not provided
    if not config_file:
        config_file = os.path.join(output_dir, "certificate.conf")
        with open(config_file, "w") as f:
            f.write("""[req]
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
DNS.2 = superlink
DNS.3 = analyzer
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
""")

    # Generate CSR
    logger.info("Generating server CSR...")
    subprocess.run(
        [
            "openssl", "req",
            "-new",
            "-key", server_key,
            "-out", server_csr,
            "-config", config_file,
        ],
        check=True,
        capture_output=True,
    )

    # Generate server certificate
    logger.info("Generating server certificate...")
    subprocess.run(
        [
            "openssl", "x509",
            "-req",
            "-in", server_csr,
            "-CA", ca_cert,
            "-CAkey", ca_key,
            "-CAcreateserial",
            "-out", server_cert,
            "-days", str(days_valid),
            "-sha256",
            "-extfile", config_file,
            "-extensions", "req_ext",
        ],
        check=True,
        capture_output=True,
    )

    # Clean up CSR
    os.remove(server_csr)

    logger.info(f"Generated TLS certificates in: {output_dir}")

    return TLSCertificates(
        ca_cert=ca_cert,
        ca_key=ca_key,
        server_cert=server_cert,
        server_key=server_key,
    )


def setup_node_credentials(
    base_dir: str,
    node_id: str,
) -> Tuple[KeyPair, str]:
    """Set up all credentials for a SuperNode.

    Args:
        base_dir: Base directory for storing credentials
        node_id: Unique identifier for the node

    Returns:
        Tuple of (KeyPair for auth, path to encryption key)
    """
    keys_dir = os.path.join(base_dir, "keys")
    encryption_dir = os.path.join(base_dir, "encryption_keys")

    # Generate authentication keys
    auth_keys = generate_ssh_keypair(
        output_dir=keys_dir,
        name=f"client_credentials_{node_id}",
    )

    # Generate encryption key
    encryption_key_path = generate_encryption_key(
        output_path=os.path.join(encryption_dir, f"node-{node_id}_key.bin"),
    )

    return auth_keys, encryption_key_path
