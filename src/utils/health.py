# src/utils/health.py
import socket
import time
from src.utils.logger import get_pipeline_logger

logger = get_pipeline_logger("health_check")


def _check_host_port(host: str, port: int) -> bool:
    """Checks if a TCP port is open and responding on a specific host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def ensure_services_running():
    """Verifies required remote services are reachable."""
    services = {
        "Qdrant": {
            "host": "192.168.100.72",   # server Qdrant
            "port": 6333,
            "action": "Ensure Qdrant is running on the server (Docker/container/service)."
        },
        "Neo4j": {
            "host": "192.168.100.72",   # server Neo4j
            "port": 7687,
            "action": "Ensure Neo4j is running on the server (Docker/container/service)."
        }
    }

    for name, config in services.items():
        host = config["host"]
        port = config["port"]
        action = config["action"]

        while not _check_host_port(host, port):
            logger.warning(f"{name} is not responding on {host}:{port}.")
            print(f"[HEALTH] {name} at {host}:{port} is down.")

            print(f"Action required:\n    {action}\n")
            input(f"Press [Enter] once {name} is accessible to re-verify...")
            time.sleep(1)

        logger.info(f"{name} is online at {host}:{port}.")
        print(f"[HEALTH] {name} is online at {host}:{port}.")
