"""Configuration management for TRE Flower microservices."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SuperLinkConfig:
    """Configuration for SuperLink (Analyzer) service."""

    # Network settings
    fleet_api_address: str = "0.0.0.0:9092"
    control_api_address: str = "0.0.0.0:9093"
    serverappio_api_address: str = "0.0.0.0:9091"

    # TLS settings
    ssl_ca_certfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    insecure: bool = False

    # Authentication
    enable_supernode_auth: bool = True

    # Storage
    database: Optional[str] = None  # SQLite path, None for in-memory
    storage_dir: Optional[str] = None
    flwr_dir: Optional[str] = None

    # Isolation mode
    isolation: str = "subprocess"  # or "process"

    @classmethod
    def from_env(cls) -> "SuperLinkConfig":
        """Create configuration from environment variables."""
        return cls(
            fleet_api_address=os.getenv("SUPERLINK_FLEET_API_ADDRESS", "0.0.0.0:9092"),
            control_api_address=os.getenv("SUPERLINK_CONTROL_API_ADDRESS", "0.0.0.0:9093"),
            serverappio_api_address=os.getenv("SUPERLINK_SERVERAPPIO_API_ADDRESS", "0.0.0.0:9091"),
            ssl_ca_certfile=os.getenv("SUPERLINK_SSL_CA_CERTFILE"),
            ssl_certfile=os.getenv("SUPERLINK_SSL_CERTFILE"),
            ssl_keyfile=os.getenv("SUPERLINK_SSL_KEYFILE"),
            insecure=os.getenv("SUPERLINK_INSECURE", "false").lower() == "true",
            enable_supernode_auth=os.getenv("SUPERLINK_ENABLE_AUTH", "true").lower() == "true",
            database=os.getenv("SUPERLINK_DATABASE"),
            storage_dir=os.getenv("SUPERLINK_STORAGE_DIR"),
            flwr_dir=os.getenv("SUPERLINK_FLWR_DIR"),
            isolation=os.getenv("SUPERLINK_ISOLATION", "subprocess"),
        )


@dataclass
class SuperNodeConfig:
    """Configuration for SuperNode (Data Owner) service."""

    # Connection settings
    superlink_address: str = "127.0.0.1:9092"
    clientappio_api_address: str = "0.0.0.0:9094"

    # TLS settings
    root_certificates: Optional[str] = None
    insecure: bool = False

    # Authentication
    auth_supernode_private_key: Optional[str] = None

    # Node configuration
    node_config: Dict[str, str] = field(default_factory=dict)

    # Retry settings
    max_retries: Optional[int] = None
    max_wait_time: Optional[int] = None

    # Isolation mode
    isolation: str = "subprocess"
    flwr_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SuperNodeConfig":
        """Create configuration from environment variables."""
        node_config = {}
        # Parse node config from environment
        node_config_str = os.getenv("SUPERNODE_NODE_CONFIG", "")
        if node_config_str:
            for pair in node_config_str.split():
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    node_config[key] = value.strip('"\'')

        return cls(
            superlink_address=os.getenv("SUPERNODE_SUPERLINK_ADDRESS", "127.0.0.1:9092"),
            clientappio_api_address=os.getenv("SUPERNODE_CLIENTAPPIO_ADDRESS", "0.0.0.0:9094"),
            root_certificates=os.getenv("SUPERNODE_ROOT_CERTIFICATES"),
            insecure=os.getenv("SUPERNODE_INSECURE", "false").lower() == "true",
            auth_supernode_private_key=os.getenv("SUPERNODE_AUTH_PRIVATE_KEY"),
            node_config=node_config,
            max_retries=int(os.getenv("SUPERNODE_MAX_RETRIES")) if os.getenv("SUPERNODE_MAX_RETRIES") else None,
            max_wait_time=int(os.getenv("SUPERNODE_MAX_WAIT_TIME")) if os.getenv("SUPERNODE_MAX_WAIT_TIME") else None,
            isolation=os.getenv("SUPERNODE_ISOLATION", "subprocess"),
            flwr_dir=os.getenv("SUPERNODE_FLWR_DIR"),
        )


@dataclass
class FederationConfig:
    """Configuration for a Flower federation."""

    name: str
    address: str  # Control API address
    root_certificates: Optional[str] = None
    insecure: bool = False

    @classmethod
    def from_env(cls, prefix: str = "FEDERATION") -> "FederationConfig":
        """Create configuration from environment variables."""
        return cls(
            name=os.getenv(f"{prefix}_NAME", "my-federation"),
            address=os.getenv(f"{prefix}_ADDRESS", "127.0.0.1:9093"),
            root_certificates=os.getenv(f"{prefix}_ROOT_CERTIFICATES"),
            insecure=os.getenv(f"{prefix}_INSECURE", "false").lower() == "true",
        )
