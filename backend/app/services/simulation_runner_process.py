"""
Simulation runner process control — subprocess launch, termination, and stop.
"""

import os
import sys
import json
import signal
import threading
import subprocess
from datetime import datetime
from queue import Queue
from typing import Any, Dict, Optional

from ..utils.logger import get_logger
from .graph_memory_updater import GraphMemoryManager
from .simulation_runner_types import RunnerStatus, SimulationRunState
from .simulation_runner_monitor import monitor_simulation

logger = get_logger('miroshark.simulation_runner')

IS_WINDOWS = sys.platform == 'win32'

RUN_STATE_DIR = os.path.join(os.path.dirname(__file__), '../../uploads/simulations')
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '../../scripts')


def build_cmd(
    config_path: str,
    script_path: str,
    platform: str,
    max_rounds: Optional[int],
    start_round: int,
    env_only: bool,
    enable_cross_platform: bool,
) -> list:
    cmd = [sys.executable, script_path, "--config", config_path]
    if max_rounds is not None and max_rounds > 0:
        cmd.extend(["--max-rounds", str(max_rounds)])
    if start_round > 0:
        cmd.extend(["--start-round", str(start_round)])
    if env_only:
        cmd.append("--env-only")
    if platform == "polymarket":
        cmd.append("--polymarket-only")
    if platform == "parallel" and enable_cross_platform:
        cmd.append("--cross-platform")
    return cmd


def terminate_process(
    process: subprocess.Popen,
    simulation_id: str,
    timeout: int = 10,
) -> None:
    """Cross-platform termination of a process and its children."""
    if IS_WINDOWS:
        logger.info(f"Terminating process tree (Windows): {simulation_id}, pid={process.pid}")
        try:
            subprocess.run(
                ['taskkill', '/PID', str(process.pid), '/T'],
                capture_output=True, timeout=5,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force terminating: {simulation_id}")
                subprocess.run(
                    ['taskkill', '/F', '/PID', str(process.pid), '/T'],
                    capture_output=True, timeout=5,
                )
                process.wait(timeout=5)
        except Exception as e:
            logger.warning(f"taskkill failed, falling back: {e}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    else:
        pgid = os.getpgid(process.pid)
        logger.info(f"Terminating process group (Unix): {simulation_id}, pgid={pgid}")
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"SIGTERM ignored, sending SIGKILL: {simulation_id}")
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=5)


def start_simulation(
    simulation_id: str,
    run_states: Dict[str, SimulationRunState],
    processes: Dict[str, subprocess.Popen],
    action_queues: Dict[str, Queue],
    monitor_threads: Dict[str, threading.Thread],
    stdout_files: Dict[str, Any],
    stderr_files: Dict[str, Any],
    graph_memory_enabled: Dict[str, bool],
    save_state_fn,
    platform: str = "parallel",
    max_rounds: Optional[int] = None,
    enable_graph_memory_update: bool = False,
    graph_id: Optional[str] = None,
    storage: Any = None,
    start_round: int = 0,
    env_only: bool = False,
    enable_cross_platform: bool = True,
) -> SimulationRunState:
    """Launch a simulation subprocess and start the monitor thread."""
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)
    config_path = os.path.join(sim_dir, "simulation_config.json")
    if not os.path.exists(config_path):
        raise ValueError("Simulation config does not exist, please call /prepare first")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)

    # max_rounds from SaaS overrides config-derived round count
    if max_rounds is not None and max_rounds > 0:
        total_rounds = max_rounds
        logger.info(
            f"Using SaaS max_rounds={max_rounds} "
            f"(config default would be {int(total_hours * 60 / minutes_per_round)})"
        )
    else:
        total_rounds = int(total_hours * 60 / minutes_per_round)

    state = SimulationRunState(
        simulation_id=simulation_id,
        runner_status=RunnerStatus.STARTING,
        total_rounds=total_rounds,
        total_simulation_hours=total_hours,
        started_at=datetime.now().isoformat(),
    )
    save_state_fn(state)

    # Graph memory setup
    if enable_graph_memory_update:
        if not graph_id:
            raise ValueError("graph_id is required when graph memory update is enabled")
        try:
            if not storage:
                raise ValueError("Must provide storage when enabling graph memory update")
            GraphMemoryManager.create_updater(simulation_id, graph_id, storage)
            graph_memory_enabled[simulation_id] = True
            logger.info(f"Graph memory update enabled: {simulation_id}, graph_id={graph_id}")
        except Exception as e:
            logger.error(f"Failed to create graph memory updater: {e}")
            graph_memory_enabled[simulation_id] = False
    else:
        graph_memory_enabled[simulation_id] = False

    # Script selection
    platform_map = {
        "twitter": ("run_twitter_simulation.py", "twitter"),
        "reddit": ("run_reddit_simulation.py", "reddit"),
        "polymarket": ("run_parallel_simulation.py", "polymarket"),
    }
    if platform in platform_map:
        script_name, running_attr = platform_map[platform]
        setattr(state, f"{running_attr}_running", True)
    else:
        script_name = "run_parallel_simulation.py"
        state.twitter_running = True
        state.reddit_running = True
        state.polymarket_running = True

    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        raise ValueError(f"Script does not exist: {script_path}")

    action_queues[simulation_id] = Queue()

    cmd = build_cmd(config_path, script_path, platform, max_rounds,
                    start_round, env_only, enable_cross_platform)

    log_mode = 'a' if start_round > 0 else 'w'
    main_log_file = open(
        os.path.join(sim_dir, "simulation.log"), log_mode, encoding='utf-8'
    )

    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        process = subprocess.Popen(
            cmd, cwd=sim_dir,
            stdout=main_log_file, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', bufsize=1, env=env,
            start_new_session=True,
        )
    except Exception as e:
        state.runner_status = RunnerStatus.FAILED
        state.error = str(e)
        save_state_fn(state)
        raise

    stdout_files[simulation_id] = main_log_file
    stderr_files[simulation_id] = None
    state.process_pid = process.pid
    state.runner_status = RunnerStatus.RUNNING
    processes[simulation_id] = process
    save_state_fn(state)

    def _cleanup(sim_id: str) -> None:
        processes.pop(sim_id, None)
        action_queues.pop(sim_id, None)
        for store in (stdout_files, stderr_files):
            fh = store.pop(sim_id, None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass

    t = threading.Thread(
        target=monitor_simulation,
        args=(simulation_id, process, state, graph_memory_enabled, save_state_fn, _cleanup),
        daemon=True,
    )
    t.start()
    monitor_threads[simulation_id] = t

    logger.info(f"Simulation started: {simulation_id}, pid={process.pid}, platform={platform}")
    return state


def stop_simulation(
    simulation_id: str,
    run_states: Dict[str, SimulationRunState],
    processes: Dict[str, subprocess.Popen],
    graph_memory_enabled: Dict[str, bool],
    save_state_fn,
    get_state_fn,
) -> SimulationRunState:
    """Stop a running simulation subprocess."""
    state = get_state_fn(simulation_id)
    if not state:
        raise ValueError(f"Simulation does not exist: {simulation_id}")
    if state.runner_status not in [RunnerStatus.RUNNING, RunnerStatus.PAUSED]:
        raise ValueError(f"Simulation is not running: {simulation_id}")

    state.runner_status = RunnerStatus.STOPPING
    save_state_fn(state)

    process = processes.get(simulation_id)
    if process and process.poll() is None:
        try:
            terminate_process(process, simulation_id)
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.error(f"Failed to terminate process group: {simulation_id}, error={e}")
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()

    state.runner_status = RunnerStatus.STOPPED
    state.twitter_running = False
    state.reddit_running = False
    state.polymarket_running = False
    state.completed_at = datetime.now().isoformat()
    save_state_fn(state)

    if graph_memory_enabled.get(simulation_id, False):
        try:
            GraphMemoryManager.stop_updater(simulation_id)
        except Exception as e:
            logger.error(f"Failed to stop graph memory updater: {e}")
        graph_memory_enabled.pop(simulation_id, None)

    logger.info(f"Simulation stopped: {simulation_id}")
    return state
