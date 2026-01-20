"""Process management utilities for Flower services."""

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    """Status of a managed process."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ProcessInfo:
    """Information about a managed process."""

    pid: Optional[int] = None
    status: ProcessStatus = ProcessStatus.PENDING
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pid": self.pid,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "logs": self.logs[-100:],  # Last 100 log lines
        }


class ProcessManager:
    """Manages subprocess lifecycle for Flower services."""

    def __init__(self, max_log_lines: int = 1000):
        self._processes: Dict[str, subprocess.Popen] = {}
        self._process_info: Dict[str, ProcessInfo] = {}
        self._log_threads: Dict[str, threading.Thread] = {}
        self._max_log_lines = max_log_lines
        self._lock = threading.Lock()

    def start_process(
        self,
        name: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        on_exit: Optional[Callable[[str, int], None]] = None,
    ) -> ProcessInfo:
        """Start a new subprocess.

        Args:
            name: Unique identifier for the process
            command: Command and arguments to execute
            env: Environment variables (merged with current env)
            cwd: Working directory
            on_exit: Callback function when process exits

        Returns:
            ProcessInfo with current status
        """
        with self._lock:
            if name in self._processes and self._processes[name].poll() is None:
                raise ValueError(f"Process '{name}' is already running")

            # Initialize process info
            info = ProcessInfo(status=ProcessStatus.STARTING)
            self._process_info[name] = info

            # Prepare environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)

            try:
                # Start the process
                logger.info(f"Starting process '{name}': {' '.join(command)}")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=process_env,
                    cwd=cwd,
                    text=True,
                    bufsize=1,
                )

                self._processes[name] = process
                info.pid = process.pid
                info.status = ProcessStatus.RUNNING
                info.started_at = datetime.utcnow()

                # Start log collection thread
                log_thread = threading.Thread(
                    target=self._collect_logs,
                    args=(name, process, on_exit),
                    daemon=True,
                )
                log_thread.start()
                self._log_threads[name] = log_thread

                logger.info(f"Process '{name}' started with PID {process.pid}")
                return info

            except Exception as e:
                info.status = ProcessStatus.FAILED
                info.error_message = str(e)
                logger.error(f"Failed to start process '{name}': {e}")
                raise

    def _collect_logs(
        self,
        name: str,
        process: subprocess.Popen,
        on_exit: Optional[Callable[[str, int], None]],
    ):
        """Collect logs from process stdout/stderr."""
        info = self._process_info[name]

        try:
            for line in process.stdout:
                line = line.rstrip()
                with self._lock:
                    info.logs.append(line)
                    # Trim logs if too many
                    if len(info.logs) > self._max_log_lines:
                        info.logs = info.logs[-self._max_log_lines:]
                logger.debug(f"[{name}] {line}")
        except Exception as e:
            logger.error(f"Error collecting logs for '{name}': {e}")

        # Process has exited
        exit_code = process.wait()
        with self._lock:
            info.exit_code = exit_code
            info.stopped_at = datetime.utcnow()
            info.status = ProcessStatus.STOPPED if exit_code == 0 else ProcessStatus.FAILED
            if exit_code != 0:
                info.error_message = f"Process exited with code {exit_code}"

        logger.info(f"Process '{name}' exited with code {exit_code}")

        if on_exit:
            try:
                on_exit(name, exit_code)
            except Exception as e:
                logger.error(f"Error in on_exit callback for '{name}': {e}")

    def stop_process(self, name: str, timeout: float = 10.0) -> ProcessInfo:
        """Stop a running process gracefully.

        Args:
            name: Process identifier
            timeout: Seconds to wait before force killing

        Returns:
            ProcessInfo with final status
        """
        with self._lock:
            if name not in self._processes:
                raise ValueError(f"Process '{name}' not found")

            process = self._processes[name]
            info = self._process_info[name]

            if process.poll() is not None:
                # Already stopped
                return info

            info.status = ProcessStatus.STOPPING

        logger.info(f"Stopping process '{name}' (PID {process.pid})")

        # Try graceful shutdown first
        process.terminate()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Process '{name}' did not stop gracefully, force killing")
            process.kill()
            process.wait()

        with self._lock:
            info = self._process_info[name]

        return info

    def get_status(self, name: str) -> Optional[ProcessInfo]:
        """Get current status of a process."""
        with self._lock:
            if name not in self._process_info:
                return None

            info = self._process_info[name]

            # Update status if process is still referenced
            if name in self._processes:
                process = self._processes[name]
                if info.status == ProcessStatus.RUNNING and process.poll() is not None:
                    # Process has exited but status not yet updated
                    info.exit_code = process.returncode
                    info.stopped_at = datetime.utcnow()
                    info.status = ProcessStatus.STOPPED if process.returncode == 0 else ProcessStatus.FAILED

            return info

    def get_all_status(self) -> Dict[str, ProcessInfo]:
        """Get status of all managed processes."""
        with self._lock:
            result = {}
            for name in self._process_info:
                result[name] = self.get_status(name)
            return result

    def get_logs(self, name: str, lines: int = 100) -> List[str]:
        """Get recent logs from a process."""
        with self._lock:
            if name not in self._process_info:
                return []
            return self._process_info[name].logs[-lines:]

    def is_running(self, name: str) -> bool:
        """Check if a process is currently running."""
        info = self.get_status(name)
        return info is not None and info.status == ProcessStatus.RUNNING

    def cleanup(self, name: str):
        """Remove a stopped process from management."""
        with self._lock:
            if name in self._processes:
                process = self._processes[name]
                if process.poll() is None:
                    raise ValueError(f"Cannot cleanup running process '{name}'")
                del self._processes[name]

            if name in self._process_info:
                del self._process_info[name]

            if name in self._log_threads:
                del self._log_threads[name]

    def stop_all(self, timeout: float = 10.0):
        """Stop all running processes."""
        with self._lock:
            names = list(self._processes.keys())

        for name in names:
            try:
                if self.is_running(name):
                    self.stop_process(name, timeout)
            except Exception as e:
                logger.error(f"Error stopping process '{name}': {e}")
