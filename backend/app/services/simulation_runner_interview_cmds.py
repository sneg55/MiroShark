"""
Simulation runner interview commands — IPC-based single, batch, and global
agent interviews plus graceful env-close.
"""

import os
import json
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger
from .simulation_ipc import SimulationIPCClient

logger = get_logger('miroshark.simulation_runner')

RUN_STATE_DIR = os.path.join(os.path.dirname(__file__), '../../uploads/simulations')


def _require_env_alive(simulation_id: str) -> SimulationIPCClient:
    """Return an IPC client or raise ValueError if env is not running."""
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)
    if not os.path.exists(sim_dir):
        raise ValueError(f"Simulation does not exist: {simulation_id}")
    ipc = SimulationIPCClient(sim_dir)
    if not ipc.check_env_alive():
        raise ValueError(
            f"Simulation environment is not running or has been closed: {simulation_id}"
        )
    return ipc


def interview_agent(
    simulation_id: str,
    agent_id: int,
    prompt: str,
    platform: Optional[str] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Interview a single Agent via IPC."""
    ipc = _require_env_alive(simulation_id)
    logger.info(
        f"Sending Interview command: simulation_id={simulation_id}, "
        f"agent_id={agent_id}, platform={platform}"
    )
    response = ipc.send_interview(
        agent_id=agent_id, prompt=prompt, platform=platform, timeout=timeout
    )
    if response.status.value == "completed":
        return {"success": True, "agent_id": agent_id, "prompt": prompt,
                "result": response.result, "timestamp": response.timestamp}
    return {"success": False, "agent_id": agent_id, "prompt": prompt,
            "error": response.error, "timestamp": response.timestamp}


def interview_agents_batch(
    simulation_id: str,
    interviews: List[Dict[str, Any]],
    platform: Optional[str] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Batch interview multiple Agents via IPC."""
    ipc = _require_env_alive(simulation_id)
    logger.info(
        f"Sending batch Interview command: simulation_id={simulation_id}, "
        f"count={len(interviews)}, platform={platform}"
    )
    response = ipc.send_batch_interview(
        interviews=interviews, platform=platform, timeout=timeout
    )
    if response.status.value == "completed":
        return {"success": True, "interviews_count": len(interviews),
                "result": response.result, "timestamp": response.timestamp}
    return {"success": False, "interviews_count": len(interviews),
            "error": response.error, "timestamp": response.timestamp}


def interview_all_agents(
    simulation_id: str,
    prompt: str,
    platform: Optional[str] = None,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Interview every Agent in the simulation config with the same question."""
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)
    if not os.path.exists(sim_dir):
        raise ValueError(f"Simulation does not exist: {simulation_id}")

    config_path = os.path.join(sim_dir, "simulation_config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Simulation config does not exist: {simulation_id}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    agent_configs = config.get("agent_configs", [])
    if not agent_configs:
        raise ValueError(f"No Agents in simulation config: {simulation_id}")

    interviews = [
        {"agent_id": ac["agent_id"], "prompt": prompt}
        for ac in agent_configs
        if ac.get("agent_id") is not None
    ]
    logger.info(
        f"Sending global Interview command: simulation_id={simulation_id}, "
        f"agent_count={len(interviews)}, platform={platform}"
    )
    return interview_agents_batch(
        simulation_id=simulation_id,
        interviews=interviews,
        platform=platform,
        timeout=timeout,
    )


def close_simulation_env(
    simulation_id: str,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Send a graceful close-environment command without stopping the process."""
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)
    if not os.path.exists(sim_dir):
        raise ValueError(f"Simulation does not exist: {simulation_id}")

    ipc = SimulationIPCClient(sim_dir)
    if not ipc.check_env_alive():
        return {"success": True, "message": "Environment is already closed"}

    logger.info(f"Sending close environment command: simulation_id={simulation_id}")
    try:
        response = ipc.send_close_env(timeout=timeout)
        return {
            "success": response.status.value == "completed",
            "message": "Close environment command sent",
            "result": response.result,
            "timestamp": response.timestamp,
        }
    except TimeoutError:
        return {
            "success": True,
            "message": (
                "Close environment command sent "
                "(response timed out, environment may be shutting down)"
            ),
        }
