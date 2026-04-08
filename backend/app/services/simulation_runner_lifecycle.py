"""
Simulation runner lifecycle — server-shutdown cleanup and signal handler registration.
"""

import os
import sys
import json
import signal
import atexit
import subprocess
from datetime import datetime
from typing import Any, Dict

from ..utils.logger import get_logger
from .graph_memory_updater import GraphMemoryManager
from .simulation_runner_types import RunnerStatus
from .simulation_runner_process import terminate_process

logger = get_logger('miroshark.simulation_runner')

RUN_STATE_DIR = os.path.join(os.path.dirname(__file__), '../../uploads/simulations')


def cleanup_all_simulations(
    processes: Dict[str, subprocess.Popen],
    graph_memory_enabled: Dict[str, bool],
    stdout_files: Dict[str, Any],
    stderr_files: Dict[str, Any],
    action_queues: Dict[str, Any],
    get_state_fn,
    save_state_fn,
    done_flag: list,          # mutable 1-element list used as a flag: [False]
) -> None:
    """
    Terminate all running simulation processes.
    Called on server shutdown via signal handlers and atexit.
    """
    if done_flag[0]:
        return
    done_flag[0] = True

    if not processes and not graph_memory_enabled:
        return

    logger.info("Cleaning up all simulation processes...")

    try:
        GraphMemoryManager.stop_all()
    except Exception as e:
        logger.error(f"Failed to stop graph memory updaters: {e}")
    graph_memory_enabled.clear()

    for simulation_id, process in list(processes.items()):
        try:
            if process.poll() is None:
                logger.info(f"Terminating: {simulation_id}, pid={process.pid}")
                try:
                    terminate_process(process, simulation_id, timeout=5)
                except (ProcessLookupError, OSError):
                    try:
                        process.terminate()
                        process.wait(timeout=3)
                    except Exception:
                        process.kill()

                state = get_state_fn(simulation_id)
                if state:
                    state.runner_status = RunnerStatus.STOPPED
                    state.twitter_running = False
                    state.reddit_running = False
                    state.completed_at = datetime.now().isoformat()
                    state.error = "Server shut down, simulation was terminated"
                    save_state_fn(state)

                try:
                    state_file = os.path.join(RUN_STATE_DIR, simulation_id, "state.json")
                    if os.path.exists(state_file):
                        with open(state_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        data['status'] = 'stopped'
                        data['updated_at'] = datetime.now().isoformat()
                        with open(state_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                except Exception as err:
                    logger.warning(f"Failed to update state.json: {simulation_id}: {err}")
        except Exception as e:
            logger.error(f"Failed to clean up process: {simulation_id}, error={e}")

    for fh in list(stdout_files.values()):
        try:
            if fh:
                fh.close()
        except Exception:
            pass
    stdout_files.clear()

    for fh in list(stderr_files.values()):
        try:
            if fh:
                fh.close()
        except Exception:
            pass
    stderr_files.clear()

    processes.clear()
    action_queues.clear()
    logger.info("Simulation process cleanup complete")


def register_cleanup(cleanup_fn) -> bool:
    """
    Register *cleanup_fn* as an atexit handler and signal handler.

    Returns True if registration succeeded, False if already registered
    or running in a debug parent process.
    """
    is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    is_debug = (
        os.environ.get('FLASK_DEBUG') == '1'
        or os.environ.get('WERKZEUG_RUN_MAIN') is not None
    )
    if is_debug and not is_reloader:
        return False

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    has_sighup = hasattr(signal, 'SIGHUP')
    original_sighup = signal.getsignal(signal.SIGHUP) if has_sighup else None

    def handler(signum=None, frame=None):
        cleanup_fn()
        if signum == signal.SIGINT and callable(original_sigint):
            original_sigint(signum, frame)
        elif signum == signal.SIGTERM and callable(original_sigterm):
            original_sigterm(signum, frame)
        elif has_sighup and signum == signal.SIGHUP:
            if callable(original_sighup):
                original_sighup(signum, frame)
            else:
                sys.exit(0)
        else:
            raise KeyboardInterrupt

    atexit.register(cleanup_fn)
    try:
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        if has_sighup:
            signal.signal(signal.SIGHUP, handler)
    except ValueError:
        logger.warning("Cannot register signal handlers (not in main thread), using atexit only")

    return True
