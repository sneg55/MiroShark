"""
Simulation runner state persistence — file I/O for SimulationRunState.
"""

import os
import json
from typing import Dict, Optional
from datetime import datetime

from ..utils.logger import get_logger
from .simulation_runner_types import AgentAction, RunnerStatus, SimulationRunState

logger = get_logger('miroshark.simulation_runner')

# Run state storage directory (mirrors SimulationRunner.RUN_STATE_DIR)
RUN_STATE_DIR = os.path.join(
    os.path.dirname(__file__),
    '../../uploads/simulations'
)


def load_run_state(
    simulation_id: str,
    cache: Dict[str, SimulationRunState],
) -> Optional[SimulationRunState]:
    """Load SimulationRunState from file (or in-memory cache)."""
    if simulation_id in cache:
        return cache[simulation_id]

    state_file = os.path.join(RUN_STATE_DIR, simulation_id, "run_state.json")
    if not os.path.exists(state_file):
        return None

    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus(data.get("runner_status", "idle")),
            current_round=data.get("current_round", 0),
            total_rounds=data.get("total_rounds", 0),
            simulated_hours=data.get("simulated_hours", 0),
            total_simulation_hours=data.get("total_simulation_hours", 0),
            twitter_current_round=data.get("twitter_current_round", 0),
            reddit_current_round=data.get("reddit_current_round", 0),
            polymarket_current_round=data.get("polymarket_current_round", 0),
            twitter_simulated_hours=data.get("twitter_simulated_hours", 0),
            reddit_simulated_hours=data.get("reddit_simulated_hours", 0),
            polymarket_simulated_hours=data.get("polymarket_simulated_hours", 0),
            twitter_running=data.get("twitter_running", False),
            reddit_running=data.get("reddit_running", False),
            polymarket_running=data.get("polymarket_running", False),
            twitter_completed=data.get("twitter_completed", False),
            reddit_completed=data.get("reddit_completed", False),
            polymarket_completed=data.get("polymarket_completed", False),
            twitter_actions_count=data.get("twitter_actions_count", 0),
            reddit_actions_count=data.get("reddit_actions_count", 0),
            polymarket_actions_count=data.get("polymarket_actions_count", 0),
            started_at=data.get("started_at"),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            process_pid=data.get("process_pid"),
        )

        # Load recent actions
        for a in data.get("recent_actions", []):
            state.recent_actions.append(AgentAction(
                round_num=a.get("round_num", 0),
                timestamp=a.get("timestamp", ""),
                platform=a.get("platform", ""),
                agent_id=a.get("agent_id", 0),
                agent_name=a.get("agent_name", ""),
                action_type=a.get("action_type", ""),
                action_args=a.get("action_args", {}),
                result=a.get("result"),
                success=a.get("success", True),
            ))

        cache[simulation_id] = state
        return state

    except Exception as e:
        logger.error(f"Failed to load run state: {str(e)}")
        return None


def save_run_state(
    state: SimulationRunState,
    cache: Dict[str, SimulationRunState],
) -> None:
    """Save SimulationRunState to file and update in-memory cache."""
    sim_dir = os.path.join(RUN_STATE_DIR, state.simulation_id)
    os.makedirs(sim_dir, exist_ok=True)
    state_file = os.path.join(sim_dir, "run_state.json")

    data = state.to_detail_dict()
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    cache[state.simulation_id] = state
