"""CLI wrapper utilities for executing Flower commands."""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a CLI command execution."""

    success: bool
    stdout: str
    stderr: str
    return_code: int
    command: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "command": " ".join(self.command),
        }


class FlowerCLI:
    """Wrapper for Flower CLI commands."""

    def __init__(
        self,
        app_path: str = ".",
        federation: Optional[str] = None,
        timeout: int = 300,
    ):
        """Initialize Flower CLI wrapper.

        Args:
            app_path: Path to the Flower app (pyproject.toml location)
            federation: Default federation name
            timeout: Default command timeout in seconds
        """
        self.app_path = app_path
        self.federation = federation
        self.timeout = timeout

    def _run_command(
        self,
        command: List[str],
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        """Execute a command and return the result.

        Args:
            command: Command and arguments
            timeout: Command timeout in seconds
            cwd: Working directory
            env: Environment variables

        Returns:
            CommandResult with execution details
        """
        timeout = timeout or self.timeout
        cwd = cwd or self.app_path

        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        logger.info(f"Executing: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=process_env,
            )

            return CommandResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                command=command,
            )

        except subprocess.TimeoutExpired as e:
            return CommandResult(
                success=False,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                return_code=-1,
                command=command,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                command=command,
            )

    # =====================
    # SuperNode Management
    # =====================

    def supernode_register(
        self,
        public_key_path: str,
        federation: Optional[str] = None,
        format_output: str = "json",
    ) -> CommandResult:
        """Register a SuperNode with the federation.

        Args:
            public_key_path: Path to the SuperNode's public key file
            federation: Federation name (uses default if not specified)
            format_output: Output format ('default' or 'json')

        Returns:
            CommandResult with registration details
        """
        federation = federation or self.federation
        command = [
            "flwr",
            "supernode",
            "register",
            public_key_path,
            self.app_path,
        ]
        if federation:
            command.append(federation)
        command.extend(["--format", format_output])

        return self._run_command(command)

    def supernode_unregister(
        self,
        node_id: str,
        federation: Optional[str] = None,
    ) -> CommandResult:
        """Unregister a SuperNode from the federation.

        Args:
            node_id: The SuperNode ID to unregister
            federation: Federation name

        Returns:
            CommandResult with unregistration details
        """
        federation = federation or self.federation
        command = [
            "flwr",
            "supernode",
            "unregister",
            node_id,
            self.app_path,
        ]
        if federation:
            command.append(federation)

        return self._run_command(command)

    def supernode_list(
        self,
        federation: Optional[str] = None,
        format_output: str = "json",
    ) -> CommandResult:
        """List all SuperNodes in the federation.

        Args:
            federation: Federation name
            format_output: Output format ('default' or 'json')

        Returns:
            CommandResult with list of SuperNodes
        """
        federation = federation or self.federation
        command = [
            "flwr",
            "supernode",
            "list",
            self.app_path,
        ]
        if federation:
            command.append(federation)
        command.extend(["--format", format_output])

        return self._run_command(command)

    # =====================
    # Run Management
    # =====================

    def run_start(
        self,
        federation: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        format_output: str = "json",
    ) -> CommandResult:
        """Start a Flower run.

        Args:
            federation: Federation name
            run_config: Override run configuration values
            format_output: Output format

        Returns:
            CommandResult with run ID
        """
        federation = federation or self.federation
        command = ["flwr", "run", self.app_path]
        if federation:
            command.append(federation)

        if run_config:
            config_str = " ".join(f'{k}={v}' for k, v in run_config.items())
            command.extend(["--run-config", config_str])

        command.extend(["--format", format_output])

        return self._run_command(command)

    def run_list(
        self,
        federation: Optional[str] = None,
        run_id: Optional[int] = None,
        list_all: bool = False,
        format_output: str = "json",
    ) -> CommandResult:
        """List runs in the federation.

        Args:
            federation: Federation name
            run_id: Specific run ID to get details for
            list_all: List all runs
            format_output: Output format

        Returns:
            CommandResult with run details
        """
        federation = federation or self.federation
        command = ["flwr", "list", self.app_path]
        if federation:
            command.append(federation)

        if run_id:
            command.extend(["--run-id", str(run_id)])
        elif list_all:
            command.append("--runs")

        command.extend(["--format", format_output])

        return self._run_command(command)

    def run_stop(
        self,
        run_id: int,
        federation: Optional[str] = None,
        format_output: str = "json",
    ) -> CommandResult:
        """Stop a running Flower run.

        Args:
            run_id: The run ID to stop
            federation: Federation name
            format_output: Output format

        Returns:
            CommandResult with stop details
        """
        federation = federation or self.federation
        command = ["flwr", "stop", str(run_id), self.app_path]
        if federation:
            command.append(federation)
        command.extend(["--format", format_output])

        return self._run_command(command)

    def run_logs(
        self,
        run_id: int,
        federation: Optional[str] = None,
        show: bool = True,
    ) -> CommandResult:
        """Get logs from a run.

        Args:
            run_id: The run ID to get logs for
            federation: Federation name
            show: If True, show logs once; if False, would stream (not supported in this wrapper)

        Returns:
            CommandResult with logs
        """
        federation = federation or self.federation
        command = ["flwr", "log", str(run_id), self.app_path]
        if federation:
            command.append(federation)
        command.append("--show")

        return self._run_command(command)

    # =====================
    # Build & Install
    # =====================

    def build(self) -> CommandResult:
        """Build a Flower App Bundle (FAB).

        Returns:
            CommandResult with build details
        """
        command = ["flwr", "build", self.app_path]
        return self._run_command(command)

    def install(self, fab_path: str) -> CommandResult:
        """Install a Flower App Bundle.

        Args:
            fab_path: Path to the FAB file

        Returns:
            CommandResult with installation details
        """
        command = ["flwr", "install", fab_path]
        return self._run_command(command)


def build_superlink_command(
    ssl_ca_certfile: Optional[str] = None,
    ssl_certfile: Optional[str] = None,
    ssl_keyfile: Optional[str] = None,
    insecure: bool = False,
    enable_supernode_auth: bool = True,
    fleet_api_address: str = "0.0.0.0:9092",
    control_api_address: str = "0.0.0.0:9093",
    serverappio_api_address: str = "0.0.0.0:9091",
    database: Optional[str] = None,
    storage_dir: Optional[str] = None,
    flwr_dir: Optional[str] = None,
    isolation: str = "subprocess",
) -> List[str]:
    """Build the flower-superlink command with arguments.

    Returns:
        List of command arguments
    """
    command = ["flower-superlink"]

    if insecure:
        command.append("--insecure")
    else:
        if ssl_ca_certfile:
            command.extend(["--ssl-ca-certfile", ssl_ca_certfile])
        if ssl_certfile:
            command.extend(["--ssl-certfile", ssl_certfile])
        if ssl_keyfile:
            command.extend(["--ssl-keyfile", ssl_keyfile])

    if enable_supernode_auth:
        command.append("--enable-supernode-auth")

    command.extend(["--fleet-api-address", fleet_api_address])
    command.extend(["--control-api-address", control_api_address])
    command.extend(["--serverappio-api-address", serverappio_api_address])

    if database:
        command.extend(["--database", database])

    if storage_dir:
        command.extend(["--storage-dir", storage_dir])

    if flwr_dir:
        command.extend(["--flwr-dir", flwr_dir])

    command.extend(["--isolation", isolation])

    return command


def build_supernode_command(
    superlink_address: str = "127.0.0.1:9092",
    root_certificates: Optional[str] = None,
    insecure: bool = False,
    auth_supernode_private_key: Optional[str] = None,
    node_config: Optional[Dict[str, str]] = None,
    clientappio_api_address: str = "0.0.0.0:9094",
    flwr_dir: Optional[str] = None,
    isolation: str = "subprocess",
    max_retries: Optional[int] = None,
    max_wait_time: Optional[int] = None,
) -> List[str]:
    """Build the flower-supernode command with arguments.

    Returns:
        List of command arguments
    """
    command = ["flower-supernode"]

    if insecure:
        command.append("--insecure")
    elif root_certificates:
        command.extend(["--root-certificates", root_certificates])

    command.extend(["--superlink", superlink_address])

    if auth_supernode_private_key:
        command.extend(["--auth-supernode-private-key", auth_supernode_private_key])

    if node_config:
        # Format: 'key1="value1" key2="value2"'
        config_parts = [f'{k}="{v}"' for k, v in node_config.items()]
        config_str = " ".join(config_parts)
        command.extend(["--node-config", config_str])

    command.extend(["--clientappio-api-address", clientappio_api_address])

    if flwr_dir:
        command.extend(["--flwr-dir", flwr_dir])

    command.extend(["--isolation", isolation])

    if max_retries is not None:
        command.extend(["--max-retries", str(max_retries)])

    if max_wait_time is not None:
        command.extend(["--max-wait-time", str(max_wait_time)])

    return command
