"""
Simulation runner interview history — env status checks and SQLite-backed
interview history retrieval.
"""

import os
import json
import sqlite3
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger
from .simulation_ipc import SimulationIPCClient

logger = get_logger('miroshark.simulation_runner')

RUN_STATE_DIR = os.path.join(os.path.dirname(__file__), '../../uploads/simulations')


def check_env_alive(simulation_id: str) -> bool:
    """Return True if the simulation environment can receive Interview commands."""
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)
    if not os.path.exists(sim_dir):
        return False
    return SimulationIPCClient(sim_dir).check_env_alive()


def get_env_status_detail(simulation_id: str) -> Dict[str, Any]:
    """Return detailed status of the simulation environment."""
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)
    status_file = os.path.join(sim_dir, "env_status.json")

    default: Dict[str, Any] = {
        "status": "stopped",
        "twitter_available": False,
        "reddit_available": False,
        "timestamp": None,
    }

    if not os.path.exists(status_file):
        return default

    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
        return {
            "status": status.get("status", "stopped"),
            "twitter_available": status.get("twitter_available", False),
            "reddit_available": status.get("reddit_available", False),
            "timestamp": status.get("timestamp"),
        }
    except (json.JSONDecodeError, OSError):
        return default


def _get_interview_history_from_db(
    db_path: str,
    platform_name: str,
    agent_id: Optional[int] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Read Interview history rows from a single SQLite database."""
    if not os.path.exists(db_path):
        return []

    results: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if agent_id is not None:
            cursor.execute(
                "SELECT user_id, info, created_at FROM trace "
                "WHERE action = 'interview' AND user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (agent_id, limit),
            )
        else:
            cursor.execute(
                "SELECT user_id, info, created_at FROM trace "
                "WHERE action = 'interview' "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )

        for user_id, info_json, created_at in cursor.fetchall():
            try:
                info = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                info = {"raw": info_json}
            results.append({
                "agent_id": user_id,
                "response": info.get("response", info),
                "prompt": info.get("prompt", ""),
                "timestamp": created_at,
                "platform": platform_name,
            })

        conn.close()
    except Exception as e:
        logger.error(f"Failed to read Interview history ({platform_name}): {e}")

    return results


def get_interview_history(
    simulation_id: str,
    platform: Optional[str] = None,
    agent_id: Optional[int] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Return Interview history from the simulation SQLite databases.

    Args:
        platform: "twitter", "reddit", or None (both).
        agent_id: Filter to a single agent.
        limit: Max records per platform.
    """
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)
    platforms = [platform] if platform in ("reddit", "twitter") else ["twitter", "reddit"]

    results: List[Dict[str, Any]] = []
    for p in platforms:
        db_path = os.path.join(sim_dir, f"{p}_simulation.db")
        results.extend(
            _get_interview_history_from_db(db_path, p, agent_id=agent_id, limit=limit)
        )

    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if len(platforms) > 1 and len(results) > limit:
        results = results[:limit]

    return results
