"""
OASIS Simulation Runner — public facade.

Implementation is split across:
  simulation_runner_types.py             RunnerStatus, AgentAction, SimulationRunState
  simulation_runner_state.py             run-state file I/O
  simulation_runner_monitor.py           background monitor thread
  simulation_runner_log_reader.py        JSONL action log parsing + pagination
  simulation_runner_aggregates.py        timeline, agent-stats, log cleanup
  simulation_runner_process.py           subprocess launch, terminate, stop
  simulation_runner_lifecycle.py         server-shutdown cleanup + signal registration
  simulation_runner_interview_cmds.py    IPC interview commands
  simulation_runner_interview_history.py env status + SQLite history
"""

import subprocess
import threading
from queue import Queue
from typing import Any, Dict, List, Optional

from .simulation_runner_types import RunnerStatus, AgentAction, RoundSummary, SimulationRunState
from .simulation_runner_state import load_run_state, save_run_state
from .simulation_runner_process import start_simulation as _start, stop_simulation as _stop
from .simulation_runner_lifecycle import cleanup_all_simulations as _cleanup_all, register_cleanup
from .simulation_runner_log_reader import get_all_actions, get_actions
from .simulation_runner_aggregates import get_timeline, get_agent_stats, cleanup_simulation_logs
from .simulation_runner_interview_cmds import (
    interview_agent, interview_agents_batch, interview_all_agents, close_simulation_env,
)
from .simulation_runner_interview_history import (
    check_env_alive, get_env_status_detail, get_interview_history,
)

_cleanup_registered = False


class SimulationRunner:
    """Thin facade — delegates to focused implementation modules."""

    _run_states: Dict[str, SimulationRunState] = {}
    _processes: Dict[str, subprocess.Popen] = {}
    _action_queues: Dict[str, Queue] = {}
    _monitor_threads: Dict[str, threading.Thread] = {}
    _stdout_files: Dict[str, Any] = {}
    _stderr_files: Dict[str, Any] = {}
    _graph_memory_enabled: Dict[str, bool] = {}
    _cleanup_done: list = [False]   # mutable flag for lifecycle module

    # -- State ----------------------------------------------------------

    @classmethod
    def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        return load_run_state(simulation_id, cls._run_states)

    @classmethod
    def _load_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        return load_run_state(simulation_id, cls._run_states)

    @classmethod
    def _save_run_state(cls, state: SimulationRunState) -> None:
        save_run_state(state, cls._run_states)

    # -- Process control ------------------------------------------------

    @classmethod
    def start_simulation(
        cls,
        simulation_id: str,
        platform: str = "parallel",
        max_rounds: Optional[int] = None,
        enable_graph_memory_update: bool = False,
        graph_id: Optional[str] = None,
        storage: Any = None,
        start_round: int = 0,
        env_only: bool = False,
        enable_cross_platform: bool = True,
    ) -> SimulationRunState:
        existing = cls.get_run_state(simulation_id)
        if existing and existing.runner_status in [RunnerStatus.RUNNING, RunnerStatus.STARTING]:
            raise ValueError(f"Simulation is already running: {simulation_id}")
        return _start(
            simulation_id=simulation_id,
            run_states=cls._run_states,
            processes=cls._processes,
            action_queues=cls._action_queues,
            monitor_threads=cls._monitor_threads,
            stdout_files=cls._stdout_files,
            stderr_files=cls._stderr_files,
            graph_memory_enabled=cls._graph_memory_enabled,
            save_state_fn=cls._save_run_state,
            platform=platform,
            max_rounds=max_rounds,
            enable_graph_memory_update=enable_graph_memory_update,
            graph_id=graph_id,
            storage=storage,
            start_round=start_round,
            env_only=env_only,
            enable_cross_platform=enable_cross_platform,
        )

    @classmethod
    def _terminate_process(cls, process, simulation_id, timeout=10):
        from .simulation_runner_process import terminate_process
        terminate_process(process, simulation_id, timeout)

    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        return _stop(
            simulation_id=simulation_id,
            run_states=cls._run_states,
            processes=cls._processes,
            graph_memory_enabled=cls._graph_memory_enabled,
            save_state_fn=cls._save_run_state,
            get_state_fn=cls.get_run_state,
        )

    # -- Lifecycle ------------------------------------------------------

    @classmethod
    def cleanup_all_simulations(cls) -> None:
        _cleanup_all(
            processes=cls._processes,
            graph_memory_enabled=cls._graph_memory_enabled,
            stdout_files=cls._stdout_files,
            stderr_files=cls._stderr_files,
            action_queues=cls._action_queues,
            get_state_fn=cls.get_run_state,
            save_state_fn=cls._save_run_state,
            done_flag=cls._cleanup_done,
        )

    @classmethod
    def register_cleanup(cls) -> None:
        global _cleanup_registered
        if _cleanup_registered:
            return
        registered = register_cleanup(cls.cleanup_all_simulations)
        if registered:
            _cleanup_registered = True

    # -- Queries --------------------------------------------------------

    @classmethod
    def get_running_simulations(cls) -> List[str]:
        return [sid for sid, p in cls._processes.items() if p.poll() is None]

    @classmethod
    def get_all_actions(cls, simulation_id, platform=None, agent_id=None, round_num=None):
        return get_all_actions(simulation_id, platform=platform,
                               agent_id=agent_id, round_num=round_num)

    @classmethod
    def get_actions(cls, simulation_id, limit=100, offset=0,
                    platform=None, agent_id=None, round_num=None):
        return get_actions(simulation_id, limit=limit, offset=offset,
                           platform=platform, agent_id=agent_id, round_num=round_num)

    @classmethod
    def get_timeline(cls, simulation_id, start_round=0, end_round=None):
        return get_timeline(simulation_id, start_round=start_round, end_round=end_round)

    @classmethod
    def get_agent_stats(cls, simulation_id):
        return get_agent_stats(simulation_id)

    @classmethod
    def cleanup_simulation_logs(cls, simulation_id):
        return cleanup_simulation_logs(simulation_id, cls._run_states)

    # -- Interviews -----------------------------------------------------

    @classmethod
    def check_env_alive(cls, simulation_id):
        return check_env_alive(simulation_id)

    @classmethod
    def get_env_status_detail(cls, simulation_id):
        return get_env_status_detail(simulation_id)

    @classmethod
    def interview_agent(cls, simulation_id, agent_id, prompt, platform=None, timeout=60.0):
        return interview_agent(simulation_id, agent_id, prompt,
                               platform=platform, timeout=timeout)

    @classmethod
    def interview_agents_batch(cls, simulation_id, interviews, platform=None, timeout=120.0):
        return interview_agents_batch(simulation_id, interviews,
                                      platform=platform, timeout=timeout)

    @classmethod
    def interview_all_agents(cls, simulation_id, prompt, platform=None, timeout=180.0):
        return interview_all_agents(simulation_id, prompt,
                                    platform=platform, timeout=timeout)

    @classmethod
    def close_simulation_env(cls, simulation_id, timeout=30.0):
        return close_simulation_env(simulation_id, timeout=timeout)

    @classmethod
    def get_interview_history(cls, simulation_id, platform=None, agent_id=None, limit=100):
        return get_interview_history(simulation_id, platform=platform,
                                     agent_id=agent_id, limit=limit)
