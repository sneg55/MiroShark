"""
Simulation runner log reader — low-level JSONL action log parsing and
per-platform action file reading.
"""

import os
import json
from typing import List, Optional

from ..utils.logger import get_logger
from .simulation_runner_types import AgentAction

logger = get_logger('miroshark.simulation_runner')

# Run state storage directory
RUN_STATE_DIR = os.path.join(
    os.path.dirname(__file__),
    '../../uploads/simulations'
)


def read_actions_from_file(
    file_path: str,
    default_platform: Optional[str] = None,
    platform_filter: Optional[str] = None,
    agent_id: Optional[int] = None,
    round_num: Optional[int] = None,
) -> List[AgentAction]:
    """
    Read AgentAction records from a single JSONL action file.

    Args:
        file_path: Path to the JSONL action log.
        default_platform: Platform to assign when the record has no platform field.
        platform_filter: Only return actions for this platform.
        agent_id: Only return actions for this agent.
        round_num: Only return actions for this round.
    """
    if not os.path.exists(file_path):
        return []

    actions: List[AgentAction] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                # Skip non-action event records
                if "event_type" in data:
                    continue

                # Skip records without agent_id
                if "agent_id" not in data:
                    continue

                record_platform = data.get("platform") or default_platform or ""

                if platform_filter and record_platform != platform_filter:
                    continue
                if agent_id is not None and data.get("agent_id") != agent_id:
                    continue
                if round_num is not None and data.get("round") != round_num:
                    continue

                actions.append(AgentAction(
                    round_num=data.get("round", 0),
                    timestamp=data.get("timestamp", ""),
                    platform=record_platform,
                    agent_id=data.get("agent_id", 0),
                    agent_name=data.get("agent_name", ""),
                    action_type=data.get("action_type", ""),
                    action_args=data.get("action_args", {}),
                    result=data.get("result"),
                    success=data.get("success", True),
                ))

            except json.JSONDecodeError:
                continue

    return actions


def get_all_actions(
    simulation_id: str,
    platform: Optional[str] = None,
    agent_id: Optional[int] = None,
    round_num: Optional[int] = None,
) -> List[AgentAction]:
    """
    Return the complete action history for all platforms (no pagination).

    Returns:
        Actions sorted by timestamp descending.
    """
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)
    actions: List[AgentAction] = []

    twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
    if not platform or platform == "twitter":
        actions.extend(read_actions_from_file(
            twitter_log, default_platform="twitter",
            platform_filter=platform, agent_id=agent_id, round_num=round_num,
        ))

    reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
    if not platform or platform == "reddit":
        actions.extend(read_actions_from_file(
            reddit_log, default_platform="reddit",
            platform_filter=platform, agent_id=agent_id, round_num=round_num,
        ))

    polymarket_log = os.path.join(sim_dir, "polymarket", "actions.jsonl")
    if not platform or platform == "polymarket":
        actions.extend(read_actions_from_file(
            polymarket_log, default_platform="polymarket",
            platform_filter=platform, agent_id=agent_id, round_num=round_num,
        ))

    # Fallback: legacy single-file format
    if not actions:
        legacy_log = os.path.join(sim_dir, "actions.jsonl")
        actions = read_actions_from_file(
            legacy_log, default_platform=None,
            platform_filter=platform, agent_id=agent_id, round_num=round_num,
        )

    actions.sort(key=lambda x: x.timestamp, reverse=True)
    return actions


def get_actions(
    simulation_id: str,
    limit: int = 100,
    offset: int = 0,
    platform: Optional[str] = None,
    agent_id: Optional[int] = None,
    round_num: Optional[int] = None,
) -> List[AgentAction]:
    """Return paginated action history."""
    all_actions = get_all_actions(
        simulation_id=simulation_id,
        platform=platform,
        agent_id=agent_id,
        round_num=round_num,
    )
    return all_actions[offset:offset + limit]
