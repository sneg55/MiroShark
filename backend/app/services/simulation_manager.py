"""
OASIS Simulation Manager
Manages parallel simulation across Twitter and Reddit platforms
Uses preset scripts + LLM-powered configuration parameter generation
"""

import os
import json
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger
from .entity_reader import EntityReader
from .simulation_config_generator import SimulationConfigGenerator, SimulationParameters

# Re-export types so existing imports still work
from .simulation_types import SimulationStatus, PlatformType, SimulationState  # noqa: F401
from .simulation_persistence import (
    SIMULATION_DATA_DIR,
    ensure_data_dir,
    get_simulation_dir,
    save_simulation_state,
    load_simulation_state,
)
from .simulation_preparation import (
    generate_config,
    generate_profiles_from_configs,
)

logger = get_logger('miroshark.simulation')


class SimulationManager:
    """
    Simulation Manager

    Core features:
    1. Read entities from graph and filter
    2. Generate OASIS Agent Profiles
    3. Use LLM to intelligently generate simulation configuration parameters
    4. Prepare all files needed by preset scripts
    """

    # Simulation data storage directory (kept for backward compat)
    SIMULATION_DATA_DIR = SIMULATION_DATA_DIR

    def __init__(self):
        ensure_data_dir()
        self._simulations: Dict[str, SimulationState] = {}

    def _get_simulation_dir(self, simulation_id: str) -> str:
        return get_simulation_dir(simulation_id)

    def _save_simulation_state(self, state: SimulationState):
        save_simulation_state(state, self._simulations)

    def _load_simulation_state(self, simulation_id: str) -> Optional[SimulationState]:
        return load_simulation_state(simulation_id, self._simulations)

    def create_simulation(
        self,
        project_id: str,
        graph_id: str,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        enable_polymarket: bool = False,
    ) -> SimulationState:
        """Create a new simulation."""
        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"

        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
            enable_polymarket=enable_polymarket,
            status=SimulationStatus.CREATED,
        )

        self._save_simulation_state(state)
        logger.info(f"Created simulation: {simulation_id}, project={project_id}, graph={graph_id}")
        return state

    def prepare_simulation(
        self,
        simulation_id: str,
        simulation_requirement: str,
        document_text: str,
        defined_entity_types: Optional[List[str]] = None,
        use_llm_for_profiles: bool = True,
        progress_callback: Optional[callable] = None,
        parallel_profile_count: int = 3,
        storage: 'GraphStorage' = None,
        target_agents: int = 5,
    ) -> SimulationState:
        """
        Prepare simulation environment (fully automated).

        Steps:
        1. Read and filter entities from graph
        2. Generate simulation config (determines expanded agent roster)
        3. Generate OASIS Agent Profiles from expanded agent configs
        4. Save config and profile files
        """
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation does not exist: {simulation_id}")

        try:
            state.status = SimulationStatus.PREPARING
            self._save_simulation_state(state)
            sim_dir = self._get_simulation_dir(simulation_id)

            # Phase 1: Read and filter entities
            if progress_callback:
                progress_callback("reading", 0, "Connecting to graph...")
            if not storage:
                raise ValueError("storage (GraphStorage) is required for prepare_simulation")

            reader = EntityReader(storage)
            if progress_callback:
                progress_callback("reading", 30, "Reading node data...")

            filtered = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=defined_entity_types,
                enrich_with_edges=True,
            )
            state.entities_count = filtered.filtered_count
            state.entity_types = list(filtered.entity_types)

            if progress_callback:
                progress_callback("reading", 100,
                                  f"Done, {filtered.filtered_count} entities in total",
                                  current=filtered.filtered_count,
                                  total=filtered.filtered_count)

            if filtered.filtered_count == 0:
                state.status = SimulationStatus.FAILED
                state.error = ("No matching entities found, "
                               "please check if the graph is built correctly")
                self._save_simulation_state(state)
                return state

            # Phase 2: Config generation (determines agent roster)
            sim_params = generate_config(state, sim_dir, filtered,
                                         simulation_requirement, document_text,
                                         progress_callback, target_agents)

            # Phase 3: Profile generation (from expanded agent configs)
            generate_profiles_from_configs(state, sim_dir, sim_params, filtered,
                                           storage, simulation_requirement,
                                           use_llm_for_profiles,
                                           progress_callback,
                                           parallel_profile_count)

            state.status = SimulationStatus.READY
            self._save_simulation_state(state)
            logger.info(f"Simulation preparation complete: {simulation_id}, "
                       f"entities={state.entities_count}, profiles={state.profiles_count}")
            return state

        except Exception as e:
            logger.error(f"Simulation preparation failed: {simulation_id}, error={str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            state.status = SimulationStatus.FAILED
            state.error = str(e)
            self._save_simulation_state(state)
            raise

    def get_simulation(self, simulation_id: str) -> Optional[SimulationState]:
        """Get simulation state."""
        return self._load_simulation_state(simulation_id)

    def list_simulations(self, project_id: Optional[str] = None) -> List[SimulationState]:
        """List all simulations."""
        simulations = []
        if os.path.exists(self.SIMULATION_DATA_DIR):
            for sim_id in os.listdir(self.SIMULATION_DATA_DIR):
                sim_path = os.path.join(self.SIMULATION_DATA_DIR, sim_id)
                if sim_id.startswith('.') or not os.path.isdir(sim_path):
                    continue
                state = self._load_simulation_state(sim_id)
                if state and (project_id is None or state.project_id == project_id):
                    simulations.append(state)
        return simulations

    def get_profiles(self, simulation_id: str, platform: str = "reddit") -> List[Dict[str, Any]]:
        """Get simulation Agent Profiles."""
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation does not exist: {simulation_id}")
        sim_dir = self._get_simulation_dir(simulation_id)
        profile_path = os.path.join(sim_dir, f"{platform}_profiles.json")
        if not os.path.exists(profile_path):
            return []
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_simulation_config(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get simulation config."""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_run_instructions(self, simulation_id: str) -> Dict[str, str]:
        """Get run instructions."""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        scripts_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../scripts'))
        return {
            "simulation_dir": sim_dir,
            "scripts_dir": scripts_dir,
            "config_file": config_path,
            "commands": {
                "twitter": (f"python {scripts_dir}/run_twitter_simulation.py "
                            f"--config {config_path}"),
                "reddit": (f"python {scripts_dir}/run_reddit_simulation.py "
                           f"--config {config_path}"),
                "parallel": (f"python {scripts_dir}/run_parallel_simulation.py "
                             f"--config {config_path}"),
            },
            "instructions": (
                f"1. Activate conda environment: conda activate MiroShark\n"
                f"2. Run simulation (scripts located at {scripts_dir}):\n"
                f"   - Twitter: python {scripts_dir}/run_twitter_simulation.py"
                f" --config {config_path}\n"
                f"   - Reddit: python {scripts_dir}/run_reddit_simulation.py"
                f" --config {config_path}\n"
                f"   - Both: python {scripts_dir}/run_parallel_simulation.py"
                f" --config {config_path}"
            ),
        }
