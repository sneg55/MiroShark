"""
Simulation persistence — file I/O, state load/save, directory management helpers.
"""

import os
import json
from typing import Dict, Optional
from datetime import datetime

from ..utils.logger import get_logger
from .simulation_types import SimulationState, SimulationStatus

logger = get_logger('miroshark.simulation')

# Simulation data storage directory
SIMULATION_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    '../../uploads/simulations'
)


def ensure_data_dir() -> None:
    """Ensure the simulation data directory exists."""
    os.makedirs(SIMULATION_DATA_DIR, exist_ok=True)


def get_simulation_dir(simulation_id: str) -> str:
    """Get simulation data directory, creating it if needed."""
    sim_dir = os.path.join(SIMULATION_DATA_DIR, simulation_id)
    os.makedirs(sim_dir, exist_ok=True)
    return sim_dir


def save_simulation_state(
    state: SimulationState,
    cache: Dict[str, SimulationState],
) -> None:
    """Save simulation state to file and update in-memory cache."""
    sim_dir = get_simulation_dir(state.simulation_id)
    state_file = os.path.join(sim_dir, "state.json")

    state.updated_at = datetime.now().isoformat()

    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

    cache[state.simulation_id] = state


def load_simulation_state(
    simulation_id: str,
    cache: Dict[str, SimulationState],
) -> Optional[SimulationState]:
    """Load simulation state from file (or in-memory cache)."""
    if simulation_id in cache:
        return cache[simulation_id]

    sim_dir = get_simulation_dir(simulation_id)
    state_file = os.path.join(sim_dir, "state.json")

    if not os.path.exists(state_file):
        return None

    with open(state_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    state = SimulationState(
        simulation_id=simulation_id,
        project_id=data.get("project_id", ""),
        graph_id=data.get("graph_id", ""),
        enable_twitter=data.get("enable_twitter", True),
        enable_reddit=data.get("enable_reddit", True),
        enable_polymarket=data.get("enable_polymarket", False),
        status=SimulationStatus(data.get("status", "created")),
        entities_count=data.get("entities_count", 0),
        profiles_count=data.get("profiles_count", 0),
        entity_types=data.get("entity_types", []),
        config_generated=data.get("config_generated", False),
        config_reasoning=data.get("config_reasoning", ""),
        current_round=data.get("current_round", 0),
        twitter_status=data.get("twitter_status", "not_started"),
        reddit_status=data.get("reddit_status", "not_started"),
        created_at=data.get("created_at", datetime.now().isoformat()),
        updated_at=data.get("updated_at", datetime.now().isoformat()),
        error=data.get("error"),
    )

    cache[simulation_id] = state
    return state
