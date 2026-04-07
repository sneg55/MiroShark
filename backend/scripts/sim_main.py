"""
Entry point for the Wonderwall multi-platform parallel simulation.

Parses CLI args, wires up loggers, dispatches to the appropriate runner
(Twitter-only, Reddit-only, Polymarket-only, or synchronized), then enters
command-waiting mode (IPC) if --no-wait is not set.
"""

import asyncio
import argparse
import os
import signal
from datetime import datetime
from typing import Optional

from action_logger import SimulationLogManager
from cross_platform_digest import CrossPlatformLog
from market_media_bridge import MarketMediaBridge
from sim_ipc import ParallelIPCHandler
from sim_twitter_runner import PlatformSimulation, run_twitter_simulation
from sim_reddit_runner import run_reddit_simulation
from sim_polymarket_runner import run_polymarket_simulation
from sim_sync_runner import run_synchronized_simulation

# These are injected by the caller (run_parallel_simulation) after it defines them
_shutdown_event = None
_cleanup_done = False


def setup_signal_handlers():
    """Register SIGTERM/SIGINT handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nReceived {sig_name} signal, exiting...")
        if not _cleanup_done:
            _cleanup_done = True
            if _shutdown_event:
                _shutdown_event.set()
        else:
            print("Force exiting...")
            import sys
            sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


async def main(
    load_config,
    get_agent_names_from_config,
    create_model,
    get_active_agents_for_round,
    build_social_summary,
    shutdown_event,
):
    """Main async entry point. All business-logic helpers passed as args to avoid circular imports."""
    parser = argparse.ArgumentParser(description='Wonderwall multi-platform parallel simulation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--twitter-only', action='store_true')
    parser.add_argument('--reddit-only', action='store_true')
    parser.add_argument('--polymarket-only', action='store_true')
    parser.add_argument('--max-rounds', type=int, default=None)
    parser.add_argument('--start-round', type=int, default=0)
    parser.add_argument('--env-only', action='store_true', default=False)
    parser.add_argument('--no-wait', action='store_true', default=False)
    parser.add_argument('--cross-platform', action='store_true', default=False)
    args = parser.parse_args()

    import sys
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    simulation_dir = os.path.dirname(args.config) or "."
    wait_for_commands = not args.no_wait

    from sim_logging import init_logging_for_simulation
    init_logging_for_simulation(simulation_dir)

    log_manager = SimulationLogManager(simulation_dir)
    twitter_logger = log_manager.get_twitter_logger()
    reddit_logger = log_manager.get_reddit_logger()
    polymarket_logger = log_manager.get_polymarket_logger()

    log_manager.info("=" * 60)
    log_manager.info("Wonderwall Multi-Platform Parallel Simulation")
    log_manager.info(f"Config: {args.config}  |  ID: {config.get('simulation_id', 'unknown')}")
    log_manager.info("=" * 60)

    time_config = config.get("time_config", {})
    total_hours = time_config.get('total_simulation_hours', 72)
    minutes_per_round = time_config.get('minutes_per_round', 30)
    config_total_rounds = (total_hours * 60) // minutes_per_round
    log_manager.info(f"  {total_hours}h / {minutes_per_round}min-per-round = {config_total_rounds} rounds"
                     + (f" (capped at {args.max_rounds})" if args.max_rounds else ""))
    log_manager.info(f"  Agents: {len(config.get('agent_configs', []))}")

    runner_kwargs = dict(
        create_model=create_model,
        get_agent_names_from_config=get_agent_names_from_config,
        get_active_agents_for_round=get_active_agents_for_round,
        shutdown_event=shutdown_event,
    )

    twitter_result: Optional[PlatformSimulation] = None
    reddit_result: Optional[PlatformSimulation] = None
    polymarket_result: Optional[PlatformSimulation] = None

    xp_log = CrossPlatformLog() if args.cross_platform else None

    if args.twitter_only:
        twitter_result = await run_twitter_simulation(
            config, simulation_dir, twitter_logger, log_manager,
            args.max_rounds, args.start_round, cross_platform_log=xp_log, **runner_kwargs,
        )
    elif args.reddit_only:
        reddit_result = await run_reddit_simulation(
            config, simulation_dir, reddit_logger, log_manager,
            args.max_rounds, args.start_round, cross_platform_log=xp_log, **runner_kwargs,
        )
    elif args.polymarket_only:
        polymarket_result = await run_polymarket_simulation(
            config, simulation_dir, polymarket_logger, log_manager,
            args.max_rounds, args.start_round, cross_platform_log=xp_log, **runner_kwargs,
        )
    else:
        has_twitter = os.path.exists(os.path.join(simulation_dir, "twitter_profiles.csv"))
        has_reddit = os.path.exists(os.path.join(simulation_dir, "reddit_profiles.json"))
        has_polymarket = os.path.exists(os.path.join(simulation_dir, "polymarket_profiles.json"))
        platform_names = [p for p, h in [("Twitter", has_twitter), ("Reddit", has_reddit), ("Polymarket", has_polymarket)] if h]
        log_manager.info(f"Synchronized mode: {', '.join(platform_names)}")

        twitter_result, reddit_result, polymarket_result = await run_synchronized_simulation(
            config=config, simulation_dir=simulation_dir,
            twitter_logger=twitter_logger, reddit_logger=reddit_logger,
            polymarket_logger=polymarket_logger, main_logger=log_manager,
            max_rounds=args.max_rounds, start_round=args.start_round,
            cross_platform_log=xp_log,
            has_twitter=has_twitter, has_reddit=has_reddit, has_polymarket=has_polymarket,
            build_social_summary=build_social_summary,
            **runner_kwargs,
        )

    log_manager.info("Simulation loop complete.")

    if wait_for_commands:
        log_manager.info("Entering command waiting mode (IPC)...")
        ipc_handler = ParallelIPCHandler(
            simulation_dir=simulation_dir,
            twitter_env=twitter_result.env if twitter_result else None,
            twitter_agent_graph=twitter_result.agent_graph if twitter_result else None,
            reddit_env=reddit_result.env if reddit_result else None,
            reddit_agent_graph=reddit_result.agent_graph if reddit_result else None,
        )
        ipc_handler.update_status("alive")
        try:
            while not shutdown_event.is_set():
                if not await ipc_handler.process_commands():
                    break
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=0.5)
                    break
                except asyncio.TimeoutError:
                    pass
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        log_manager.info("Shutting down environment...")
        ipc_handler.update_status("stopped")

    if twitter_result and twitter_result.env:
        await twitter_result.env.close()
    if reddit_result and reddit_result.env:
        await reddit_result.env.close()

    log_manager.info("All done.")
