"""
Simulation runner aggregates — timeline summaries, per-agent statistics,
and log cleanup.
"""

import os
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger
from .simulation_runner_log_reader import get_actions

logger = get_logger('miroshark.simulation_runner')

# Run state storage directory
RUN_STATE_DIR = os.path.join(
    os.path.dirname(__file__),
    '../../uploads/simulations'
)


def get_timeline(
    simulation_id: str,
    start_round: int = 0,
    end_round: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Return simulation timeline summarised per round.

    Returns:
        List of per-round summary dicts sorted by round number.
    """
    actions = get_actions(simulation_id, limit=10000)
    rounds: Dict[int, Dict[str, Any]] = {}

    for action in actions:
        rn = action.round_num
        if rn < start_round:
            continue
        if end_round is not None and rn > end_round:
            continue

        if rn not in rounds:
            rounds[rn] = {
                "round_num": rn,
                "twitter_actions": 0,
                "reddit_actions": 0,
                "active_agents": set(),
                "action_types": {},
                "first_action_time": action.timestamp,
                "last_action_time": action.timestamp,
            }

        r = rounds[rn]
        if action.platform == "twitter":
            r["twitter_actions"] += 1
        else:
            r["reddit_actions"] += 1
        r["active_agents"].add(action.agent_id)
        r["action_types"][action.action_type] = r["action_types"].get(action.action_type, 0) + 1
        r["last_action_time"] = action.timestamp

    result = []
    for rn in sorted(rounds.keys()):
        r = rounds[rn]
        result.append({
            "round_num": rn,
            "twitter_actions": r["twitter_actions"],
            "reddit_actions": r["reddit_actions"],
            "total_actions": r["twitter_actions"] + r["reddit_actions"],
            "active_agents_count": len(r["active_agents"]),
            "active_agents": list(r["active_agents"]),
            "action_types": r["action_types"],
            "first_action_time": r["first_action_time"],
            "last_action_time": r["last_action_time"],
        })

    return result


def get_agent_stats(simulation_id: str) -> List[Dict[str, Any]]:
    """
    Return per-agent action statistics, sorted by total actions descending.
    """
    actions = get_actions(simulation_id, limit=10000)
    agent_stats: Dict[int, Dict[str, Any]] = {}

    for action in actions:
        aid = action.agent_id
        if aid not in agent_stats:
            agent_stats[aid] = {
                "agent_id": aid,
                "agent_name": action.agent_name,
                "total_actions": 0,
                "twitter_actions": 0,
                "reddit_actions": 0,
                "action_types": {},
                "first_action_time": action.timestamp,
                "last_action_time": action.timestamp,
            }
        stats = agent_stats[aid]
        stats["total_actions"] += 1
        if action.platform == "twitter":
            stats["twitter_actions"] += 1
        else:
            stats["reddit_actions"] += 1
        stats["action_types"][action.action_type] = (
            stats["action_types"].get(action.action_type, 0) + 1
        )
        stats["last_action_time"] = action.timestamp

    return sorted(agent_stats.values(), key=lambda x: x["total_actions"], reverse=True)


def cleanup_simulation_logs(
    simulation_id: str,
    run_state_cache: dict,
) -> Dict[str, Any]:
    """
    Delete simulation run logs to allow a clean restart.

    Removes run_state.json, action logs, simulation.log, database files, and
    env_status.json.  Does NOT delete simulation_config.json or profile files.

    Returns:
        Dict with keys: success, cleaned_files, errors.
    """
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)

    if not os.path.exists(sim_dir):
        return {"success": True, "message": "Simulation directory does not exist, no cleanup needed"}

    cleaned_files: List[str] = []
    errors: List[str] = []

    for filename in [
        "run_state.json", "simulation.log", "stdout.log", "stderr.log",
        "twitter_simulation.db", "reddit_simulation.db", "env_status.json",
    ]:
        file_path = os.path.join(sim_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                cleaned_files.append(filename)
            except Exception as e:
                errors.append(f"Failed to delete {filename}: {str(e)}")

    for dir_name in ["twitter", "reddit", "polymarket"]:
        actions_file = os.path.join(sim_dir, dir_name, "actions.jsonl")
        if os.path.exists(actions_file):
            try:
                os.remove(actions_file)
                cleaned_files.append(f"{dir_name}/actions.jsonl")
            except Exception as e:
                errors.append(f"Failed to delete {dir_name}/actions.jsonl: {str(e)}")

    run_state_cache.pop(simulation_id, None)

    logger.info(
        f"Simulation log cleanup complete: {simulation_id}, deleted files: {cleaned_files}"
    )
    return {
        "success": len(errors) == 0,
        "cleaned_files": cleaned_files,
        "errors": errors if errors else None,
    }
