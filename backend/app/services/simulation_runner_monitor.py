"""
Simulation runner monitoring — background thread that tails per-platform action logs
and updates SimulationRunState in real time.
"""

import os
import json
import time
import subprocess
import threading
from datetime import datetime
from typing import Dict, Any, Optional

from ..utils.logger import get_logger
from .simulation_runner_types import AgentAction, RunnerStatus, SimulationRunState
from .graph_memory_updater import GraphMemoryManager

logger = get_logger('miroshark.simulation_runner')

# Run state storage directory
RUN_STATE_DIR = os.path.join(
    os.path.dirname(__file__),
    '../../uploads/simulations'
)


def check_all_platforms_completed(state: SimulationRunState) -> bool:
    """
    Check if all enabled platforms have completed simulation.

    Determines whether a platform is enabled by checking whether the corresponding
    actions.jsonl file exists.

    Returns:
        True if all enabled platforms are completed.
    """
    sim_dir = os.path.join(RUN_STATE_DIR, state.simulation_id)
    twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
    reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
    polymarket_log = os.path.join(sim_dir, "polymarket", "actions.jsonl")

    twitter_enabled = os.path.exists(twitter_log)
    reddit_enabled = os.path.exists(reddit_log)
    polymarket_enabled = os.path.exists(polymarket_log)

    if twitter_enabled and not state.twitter_completed:
        return False
    if reddit_enabled and not state.reddit_completed:
        return False
    if polymarket_enabled and not state.polymarket_completed:
        return False

    return twitter_enabled or reddit_enabled or polymarket_enabled


def read_action_log(
    log_path: str,
    position: int,
    state: SimulationRunState,
    platform: str,
    graph_memory_enabled: Dict[str, bool],
) -> int:
    """
    Read new lines from a platform action log file starting at *position*.

    Args:
        log_path: Path to the JSONL action log.
        position: Byte offset of the last read position.
        state: Live run state to mutate.
        platform: "twitter", "reddit", or "polymarket".
        graph_memory_enabled: Shared dict mapping simulation_id -> bool.

    Returns:
        New byte offset after reading.
    """
    enabled = graph_memory_enabled.get(state.simulation_id, False)
    graph_updater = GraphMemoryManager.get_updater(state.simulation_id) if enabled else None

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            f.seek(position)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    action_data = json.loads(line)

                    if "event_type" in action_data:
                        event_type = action_data.get("event_type")

                        if event_type == "simulation_end":
                            if platform == "twitter":
                                state.twitter_completed = True
                                state.twitter_running = False
                                logger.info(
                                    f"Twitter simulation completed: {state.simulation_id}, "
                                    f"total_rounds={action_data.get('total_rounds')}, "
                                    f"total_actions={action_data.get('total_actions')}"
                                )
                            elif platform == "reddit":
                                state.reddit_completed = True
                                state.reddit_running = False
                                logger.info(
                                    f"Reddit simulation completed: {state.simulation_id}, "
                                    f"total_rounds={action_data.get('total_rounds')}, "
                                    f"total_actions={action_data.get('total_actions')}"
                                )
                            elif platform == "polymarket":
                                state.polymarket_completed = True
                                state.polymarket_running = False
                                logger.info(
                                    f"Polymarket simulation completed: {state.simulation_id}, "
                                    f"total_rounds={action_data.get('total_rounds')}, "
                                    f"total_actions={action_data.get('total_actions')}"
                                )

                            if check_all_platforms_completed(state):
                                state.runner_status = RunnerStatus.COMPLETED
                                state.completed_at = datetime.now().isoformat()
                                logger.info(
                                    f"All platform simulations completed: {state.simulation_id}"
                                )

                        elif event_type == "round_end":
                            round_num = action_data.get("round", 0)
                            simulated_hours = action_data.get("simulated_hours", 0)

                            if platform == "twitter":
                                if round_num > state.twitter_current_round:
                                    state.twitter_current_round = round_num
                                state.twitter_simulated_hours = simulated_hours
                            elif platform == "reddit":
                                if round_num > state.reddit_current_round:
                                    state.reddit_current_round = round_num
                                state.reddit_simulated_hours = simulated_hours
                            elif platform == "polymarket":
                                if round_num > state.polymarket_current_round:
                                    state.polymarket_current_round = round_num
                                state.polymarket_simulated_hours = simulated_hours

                            if round_num > state.current_round:
                                state.current_round = round_num
                            state.simulated_hours = max(
                                state.twitter_simulated_hours,
                                state.reddit_simulated_hours,
                                state.polymarket_simulated_hours,
                            )

                        continue

                    action = AgentAction(
                        round_num=action_data.get("round", 0),
                        timestamp=action_data.get("timestamp", datetime.now().isoformat()),
                        platform=platform,
                        agent_id=action_data.get("agent_id", 0),
                        agent_name=action_data.get("agent_name", ""),
                        action_type=action_data.get("action_type", ""),
                        action_args=action_data.get("action_args", {}),
                        result=action_data.get("result"),
                        success=action_data.get("success", True),
                    )
                    state.add_action(action)

                    if action.round_num and action.round_num > state.current_round:
                        state.current_round = action.round_num

                    if graph_updater:
                        graph_updater.add_activity_from_dict(action_data, platform)

                except json.JSONDecodeError:
                    pass
            return f.tell()
    except Exception as e:
        logger.warning(f"Failed to read action log: {log_path}, error={e}")
        return position


def monitor_simulation(
    simulation_id: str,
    process: subprocess.Popen,
    state: SimulationRunState,
    graph_memory_enabled: Dict[str, bool],
    save_state_fn,
    cleanup_fn,
) -> None:
    """
    Background thread: tail per-platform action logs while the subprocess runs,
    then finalise state when the process exits.

    Args:
        simulation_id: Simulation ID.
        process: The running subprocess.
        state: Live run state object (mutated in place).
        graph_memory_enabled: Shared dict mapping simulation_id -> bool.
        save_state_fn: Callable(state) to persist state to disk.
        cleanup_fn: Callable(simulation_id) to release process/thread resources.
    """
    sim_dir = os.path.join(RUN_STATE_DIR, simulation_id)

    twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
    reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
    polymarket_log = os.path.join(sim_dir, "polymarket", "actions.jsonl")

    # If resuming, skip past existing log content to avoid re-reading old simulation_end events
    twitter_pos = os.path.getsize(twitter_log) if os.path.exists(twitter_log) else 0
    reddit_pos = os.path.getsize(reddit_log) if os.path.exists(reddit_log) else 0
    polymarket_pos = os.path.getsize(polymarket_log) if os.path.exists(polymarket_log) else 0

    try:
        while process.poll() is None:
            if os.path.exists(twitter_log):
                twitter_pos = read_action_log(
                    twitter_log, twitter_pos, state, "twitter", graph_memory_enabled
                )
            if os.path.exists(reddit_log):
                reddit_pos = read_action_log(
                    reddit_log, reddit_pos, state, "reddit", graph_memory_enabled
                )
            if os.path.exists(polymarket_log):
                polymarket_pos = read_action_log(
                    polymarket_log, polymarket_pos, state, "polymarket", graph_memory_enabled
                )
            save_state_fn(state)
            time.sleep(2)

        # Final read after process exits
        if os.path.exists(twitter_log):
            read_action_log(twitter_log, twitter_pos, state, "twitter", graph_memory_enabled)
        if os.path.exists(reddit_log):
            read_action_log(reddit_log, reddit_pos, state, "reddit", graph_memory_enabled)
        if os.path.exists(polymarket_log):
            read_action_log(
                polymarket_log, polymarket_pos, state, "polymarket", graph_memory_enabled
            )

        exit_code = process.returncode
        if exit_code == 0:
            state.runner_status = RunnerStatus.COMPLETED
            state.completed_at = datetime.now().isoformat()
            logger.info(f"Simulation completed: {simulation_id}")
        else:
            state.runner_status = RunnerStatus.FAILED
            main_log_path = os.path.join(sim_dir, "simulation.log")
            error_info = ""
            try:
                if os.path.exists(main_log_path):
                    with open(main_log_path, 'r', encoding='utf-8') as f:
                        error_info = f.read()[-2000:]
            except Exception:
                pass
            state.error = f"Process exit code: {exit_code}, error: {error_info}"
            logger.error(f"Simulation failed: {simulation_id}, error={state.error}")

        state.twitter_running = False
        state.reddit_running = False
        state.polymarket_running = False
        save_state_fn(state)

    except Exception as e:
        logger.error(f"Monitor thread exception: {simulation_id}, error={str(e)}")
        state.runner_status = RunnerStatus.FAILED
        state.error = str(e)
        save_state_fn(state)

    finally:
        # Stop graph memory updater
        if graph_memory_enabled.get(simulation_id, False):
            try:
                GraphMemoryManager.stop_updater(simulation_id)
                logger.info(f"Stopped graph memory update: simulation_id={simulation_id}")
            except Exception as e:
                logger.error(f"Failed to stop graph memory updater: {e}")
            graph_memory_enabled.pop(simulation_id, None)

        cleanup_fn(simulation_id)
