#!/usr/bin/env python3
"""Orchestration script for TRE Flower federated learning workflow.

This script orchestrates the complete federated learning workflow:
1. Health checks for all services
2. Start SuperLink on the analyzer service
3. Register SuperNodes with the analyzer
4. Start SuperNodes on data owner services
5. Execute federated learning run
6. Monitor progress and retrieve results
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a TRE service."""

    name: str
    url: str
    node_id: Optional[str] = None


class TREOrchestrator:
    """Orchestrates federated learning across TRE microservices."""

    def __init__(
        self,
        analyzer_url: str = "http://localhost:5000",
        dataowner_urls: Optional[List[str]] = None,
        timeout: int = 30,
    ):
        """Initialize the orchestrator.

        Args:
            analyzer_url: URL of the analyzer service
            dataowner_urls: List of data owner service URLs
            timeout: Request timeout in seconds
        """
        self.analyzer = ServiceConfig(name="analyzer", url=analyzer_url)
        self.dataowners: List[ServiceConfig] = []

        if dataowner_urls is None:
            dataowner_urls = ["http://localhost:5001", "http://localhost:5002"]

        for i, url in enumerate(dataowner_urls, 1):
            self.dataowners.append(
                ServiceConfig(name=f"dataowner_{i}", url=url, node_id=str(i))
            )

        self.timeout = timeout

    def _request(
        self,
        method: str,
        url: str,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
    ) -> Dict:
        """Make an HTTP request and return JSON response."""
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                if files:
                    response = requests.post(url, files=files, timeout=self.timeout)
                else:
                    response = requests.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"success": False, "error": str(e)}

    # ==========================================
    # Health Checks
    # ==========================================

    def check_health(self, service: ServiceConfig) -> bool:
        """Check if a service is healthy."""
        url = f"{service.url}/health"
        result = self._request("GET", url)
        return result.get("success", False)

    def check_all_health(self) -> Dict[str, bool]:
        """Check health of all services."""
        health = {}

        logger.info("Checking analyzer health...")
        health["analyzer"] = self.check_health(self.analyzer)
        logger.info(f"  Analyzer: {'OK' if health['analyzer'] else 'FAILED'}")

        for do in self.dataowners:
            logger.info(f"Checking {do.name} health...")
            health[do.name] = self.check_health(do)
            logger.info(f"  {do.name}: {'OK' if health[do.name] else 'FAILED'}")

        return health

    def wait_for_services(self, max_attempts: int = 30, delay: int = 2) -> bool:
        """Wait for all services to become healthy."""
        logger.info("Waiting for all services to become healthy...")

        for attempt in range(max_attempts):
            health = self.check_all_health()
            if all(health.values()):
                logger.info("All services are healthy!")
                return True

            unhealthy = [name for name, ok in health.items() if not ok]
            logger.info(f"Waiting for services: {unhealthy} (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)

        logger.error("Timeout waiting for services")
        return False

    # ==========================================
    # SuperLink Management
    # ==========================================

    def start_superlink(self, config: Optional[Dict] = None) -> Dict:
        """Start the SuperLink on the analyzer service."""
        logger.info("Starting SuperLink...")
        url = f"{self.analyzer.url}/superlink/start"
        result = self._request("POST", url, data=config or {})

        if result.get("success"):
            logger.info("SuperLink started successfully")
        else:
            logger.error(f"Failed to start SuperLink: {result.get('error')}")

        return result

    def stop_superlink(self) -> Dict:
        """Stop the SuperLink on the analyzer service."""
        logger.info("Stopping SuperLink...")
        url = f"{self.analyzer.url}/superlink/stop"
        result = self._request("POST", url)

        if result.get("success"):
            logger.info("SuperLink stopped successfully")
        else:
            logger.error(f"Failed to stop SuperLink: {result.get('error')}")

        return result

    def get_superlink_status(self) -> Dict:
        """Get SuperLink status."""
        url = f"{self.analyzer.url}/superlink/status"
        return self._request("GET", url)

    def wait_for_superlink(self, max_attempts: int = 30, delay: int = 2) -> bool:
        """Wait for SuperLink to be running."""
        logger.info("Waiting for SuperLink to be ready...")

        for attempt in range(max_attempts):
            status = self.get_superlink_status()
            if status.get("success") and status.get("data", {}).get("status") == "running":
                logger.info("SuperLink is ready!")
                return True

            logger.info(f"Waiting for SuperLink... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)

        logger.error("Timeout waiting for SuperLink")
        return False

    # ==========================================
    # SuperNode Registration
    # ==========================================

    def register_supernode(self, node_id: str, generate_keys: bool = False) -> Dict:
        """Register a SuperNode with the analyzer."""
        logger.info(f"Registering SuperNode {node_id}...")
        url = f"{self.analyzer.url}/supernodes/register"

        if generate_keys:
            data = {"node_id": node_id, "generate_keys": True}
        else:
            data = {"public_key_path": f"/app/data/keys/client_credentials_{node_id}.pub"}

        result = self._request("POST", url, data=data)

        if result.get("success"):
            logger.info(f"SuperNode {node_id} registered successfully")
        else:
            logger.error(f"Failed to register SuperNode {node_id}: {result.get('error')}")

        return result

    def register_all_supernodes(self) -> Dict[str, Dict]:
        """Register all SuperNodes."""
        results = {}
        for do in self.dataowners:
            results[do.node_id] = self.register_supernode(do.node_id)
        return results

    def list_supernodes(self) -> Dict:
        """List all registered SuperNodes."""
        url = f"{self.analyzer.url}/supernodes"
        return self._request("GET", url)

    # ==========================================
    # SuperNode (Data Owner) Management
    # ==========================================

    def start_supernode(self, service: ServiceConfig, config: Optional[Dict] = None) -> Dict:
        """Start the SuperNode on a data owner service."""
        logger.info(f"Starting SuperNode on {service.name}...")
        url = f"{service.url}/supernode/start"
        result = self._request("POST", url, data=config or {})

        if result.get("success"):
            logger.info(f"SuperNode on {service.name} started successfully")
        else:
            logger.error(f"Failed to start SuperNode on {service.name}: {result.get('error')}")

        return result

    def stop_supernode(self, service: ServiceConfig) -> Dict:
        """Stop the SuperNode on a data owner service."""
        logger.info(f"Stopping SuperNode on {service.name}...")
        url = f"{service.url}/supernode/stop"
        result = self._request("POST", url)

        if result.get("success"):
            logger.info(f"SuperNode on {service.name} stopped successfully")
        else:
            logger.error(f"Failed to stop SuperNode on {service.name}: {result.get('error')}")

        return result

    def start_all_supernodes(self, config: Optional[Dict] = None) -> Dict[str, Dict]:
        """Start SuperNodes on all data owner services."""
        results = {}
        for do in self.dataowners:
            results[do.name] = self.start_supernode(do, config)
            # Small delay between starts
            time.sleep(1)
        return results

    def stop_all_supernodes(self) -> Dict[str, Dict]:
        """Stop SuperNodes on all data owner services."""
        results = {}
        for do in self.dataowners:
            results[do.name] = self.stop_supernode(do)
        return results

    def get_supernode_status(self, service: ServiceConfig) -> Dict:
        """Get SuperNode status on a data owner service."""
        url = f"{service.url}/supernode/status"
        return self._request("GET", url)

    def wait_for_supernodes(self, max_attempts: int = 30, delay: int = 2) -> bool:
        """Wait for all SuperNodes to be running."""
        logger.info("Waiting for all SuperNodes to be ready...")

        for attempt in range(max_attempts):
            all_running = True

            for do in self.dataowners:
                status = self.get_supernode_status(do)
                if not (status.get("success") and status.get("data", {}).get("status") == "running"):
                    all_running = False
                    break

            if all_running:
                logger.info("All SuperNodes are ready!")
                return True

            logger.info(f"Waiting for SuperNodes... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)

        logger.error("Timeout waiting for SuperNodes")
        return False

    # ==========================================
    # Federated Learning Run
    # ==========================================

    def start_run(self, run_config: Optional[Dict] = None) -> Dict:
        """Start a federated learning run."""
        logger.info("Starting federated learning run...")
        url = f"{self.analyzer.url}/runs"
        result = self._request("POST", url, data={"run_config": run_config} if run_config else {})

        if result.get("success"):
            logger.info("Federated learning run started successfully")
        else:
            logger.error(f"Failed to start run: {result.get('error')}")

        return result

    def list_runs(self) -> Dict:
        """List all runs."""
        url = f"{self.analyzer.url}/runs"
        return self._request("GET", url)

    def get_run(self, run_id: int) -> Dict:
        """Get details of a specific run."""
        url = f"{self.analyzer.url}/runs/{run_id}"
        return self._request("GET", url)

    def stop_run(self, run_id: int) -> Dict:
        """Stop a running federated learning run."""
        logger.info(f"Stopping run {run_id}...")
        url = f"{self.analyzer.url}/runs/{run_id}/stop"
        return self._request("POST", url)

    def get_run_logs(self, run_id: int) -> Dict:
        """Get logs from a run."""
        url = f"{self.analyzer.url}/runs/{run_id}/logs"
        return self._request("GET", url)

    def monitor_run(self, run_id: int, poll_interval: int = 5, max_duration: int = 600) -> bool:
        """Monitor a run until completion."""
        logger.info(f"Monitoring run {run_id}...")
        start_time = time.time()

        while time.time() - start_time < max_duration:
            run_info = self.get_run(run_id)
            if not run_info.get("success"):
                logger.error(f"Failed to get run info: {run_info.get('error')}")
                time.sleep(poll_interval)
                continue

            output = run_info.get("data", {}).get("output", "")
            logger.info(f"Run status: {output}")

            # Check if run is finished (this is a simplified check)
            if "finished" in output.lower() or "completed" in output.lower():
                logger.info(f"Run {run_id} completed!")
                return True

            if "failed" in output.lower() or "error" in output.lower():
                logger.error(f"Run {run_id} failed!")
                return False

            time.sleep(poll_interval)

        logger.warning(f"Monitoring timeout for run {run_id}")
        return False

    # ==========================================
    # Full Workflow
    # ==========================================

    def run_full_workflow(
        self,
        run_config: Optional[Dict] = None,
        skip_registration: bool = False,
    ) -> bool:
        """Execute the complete federated learning workflow.

        Args:
            run_config: Configuration overrides for the run
            skip_registration: Skip SuperNode registration (if already done)

        Returns:
            True if workflow completed successfully
        """
        logger.info("=" * 60)
        logger.info("Starting TRE Federated Learning Workflow")
        logger.info("=" * 60)

        # Step 1: Health checks
        logger.info("\n[Step 1] Health Checks")
        if not self.wait_for_services():
            logger.error("Services health check failed")
            return False

        # Step 2: Start SuperLink
        logger.info("\n[Step 2] Start SuperLink")
        result = self.start_superlink()
        if not result.get("success"):
            logger.error("Failed to start SuperLink")
            return False

        if not self.wait_for_superlink():
            logger.error("SuperLink failed to become ready")
            return False

        # Step 3: Register SuperNodes
        if not skip_registration:
            logger.info("\n[Step 3] Register SuperNodes")
            results = self.register_all_supernodes()
            for node_id, result in results.items():
                if not result.get("success"):
                    logger.error(f"Failed to register SuperNode {node_id}")
                    return False
        else:
            logger.info("\n[Step 3] Skipping SuperNode Registration")

        # Small delay after registration
        time.sleep(2)

        # Step 4: Start SuperNodes
        logger.info("\n[Step 4] Start SuperNodes")
        results = self.start_all_supernodes()
        for name, result in results.items():
            if not result.get("success"):
                logger.error(f"Failed to start SuperNode on {name}")
                return False

        if not self.wait_for_supernodes():
            logger.error("SuperNodes failed to become ready")
            return False

        # Give SuperNodes time to connect to SuperLink
        logger.info("Waiting for SuperNodes to connect to SuperLink...")
        time.sleep(5)

        # Step 5: Start Federated Learning Run
        logger.info("\n[Step 5] Start Federated Learning Run")
        result = self.start_run(run_config)
        if not result.get("success"):
            logger.error("Failed to start federated learning run")
            return False

        # Extract run ID from output (this is simplified)
        run_output = result.get("data", {}).get("output", "")
        logger.info(f"Run started: {run_output}")

        # Step 6: Monitor progress
        logger.info("\n[Step 6] Monitoring Progress")
        # Note: In production, you would parse the run ID and monitor it
        # For now, we'll just wait and check logs periodically

        logger.info("\n" + "=" * 60)
        logger.info("Workflow initiated successfully!")
        logger.info("Monitor logs using:")
        logger.info(f"  curl {self.analyzer.url}/superlink/logs")
        logger.info("=" * 60)

        return True

    def cleanup(self):
        """Stop all services."""
        logger.info("\n" + "=" * 60)
        logger.info("Cleaning up...")
        logger.info("=" * 60)

        # Stop SuperNodes first
        self.stop_all_supernodes()

        # Then stop SuperLink
        self.stop_superlink()

        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate TRE Flower federated learning workflow"
    )
    parser.add_argument(
        "--analyzer-url",
        default="http://localhost:5000",
        help="URL of the analyzer service",
    )
    parser.add_argument(
        "--dataowner-urls",
        nargs="+",
        default=["http://localhost:5001", "http://localhost:5002"],
        help="URLs of data owner services",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Number of federated learning rounds",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs per round",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Skip SuperNode registration",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run cleanup (stop all services)",
    )
    parser.add_argument(
        "--health-check-only",
        action="store_true",
        help="Only run health checks",
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = TREOrchestrator(
        analyzer_url=args.analyzer_url,
        dataowner_urls=args.dataowner_urls,
    )

    try:
        if args.cleanup_only:
            orchestrator.cleanup()
            return 0

        if args.health_check_only:
            health = orchestrator.check_all_health()
            print(json.dumps(health, indent=2))
            return 0 if all(health.values()) else 1

        # Run full workflow
        run_config = {
            "num-server-rounds": args.num_rounds,
            "local-epochs": args.local_epochs,
        }

        success = orchestrator.run_full_workflow(
            run_config=run_config,
            skip_registration=args.skip_registration,
        )

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        orchestrator.cleanup()
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        orchestrator.cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())
