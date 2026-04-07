"""
Synchronized multi-platform simulation round loop.

All platforms (Twitter, Reddit, Polymarket) step together: round N completes
on ALL platforms before round N+1 starts on ANY platform. This ensures the
Market-Media Bridge data is never stale.

Setup (environment init, seeding) is delegated to sim_sync_setup.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from wonderwall import LLMAction

from action_logger import PlatformActionLogger, SimulationLogManager
from cross_platform_digest import CrossPlatformLog, inject_cross_platform_context
from market_media_bridge import MarketMediaBridge, inject_market_context, inject_sentiment_context
from round_memory import inject_round_memory
from sim_db_fetch import fetch_new_actions_from_db, fetch_polymarket_actions_from_db
from sim_sync_setup import (
    setup_platform_envs, seed_initial_events,
    build_belief_trackers, build_round_memory,
)
from sim_twitter_runner import PlatformSimulation
import os


async def run_synchronized_simulation(
    config: Dict[str, Any],
    simulation_dir: str,
    twitter_logger: Optional[PlatformActionLogger] = None,
    reddit_logger: Optional[PlatformActionLogger] = None,
    polymarket_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None,
    start_round: int = 0,
    cross_platform_log: Optional[CrossPlatformLog] = None,
    has_twitter: bool = False,
    has_reddit: bool = False,
    has_polymarket: bool = False,
    create_model=None,
    get_agent_names_from_config=None,
    get_active_agents_for_round=None,
    build_social_summary=None,
    shutdown_event=None,
) -> Tuple[Optional[PlatformSimulation], Optional[PlatformSimulation], Optional[PlatformSimulation]]:
    """Run all platforms in lock-step, one round at a time.

    Returns (twitter_result, reddit_result, polymarket_result).
    """
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Sync] {msg}")
        print(f"[Sync] {msg}")

    agent_names = get_agent_names_from_config(config)
    model = create_model(config, use_boost=False)
    bridge = MarketMediaBridge()
    log_info("Synchronized mode: all platforms step together per round")

    round_memory = build_round_memory(config)
    log_info("Round Memory: ENABLED")

    twitter_result, reddit_result, polymarket_result = await setup_platform_envs(
        config, simulation_dir, model, agent_names, start_round,
        has_twitter, has_reddit, has_polymarket, log_info,
    )

    twitter_belief, reddit_belief, polymarket_belief = build_belief_trackers(
        config, simulation_dir, has_twitter, has_reddit, has_polymarket,
    )

    for logger in (twitter_logger, reddit_logger, polymarket_logger):
        if logger:
            logger.log_simulation_start(config)

    await seed_initial_events(
        config, start_round, twitter_result, reddit_result, polymarket_result, log_info,
    )

    # Derive DB paths for post-round fetching
    twitter_db = os.path.join(simulation_dir, "twitter_simulation.db")
    reddit_db = os.path.join(simulation_dir, "reddit_simulation.db")
    polymarket_db = os.path.join(simulation_dir, "polymarket_simulation.db")

    time_config = config.get("time_config", {})
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_hours = time_config.get("total_simulation_hours", 72)
    # max_rounds is the target, not a ceiling — SaaS layer controls round count
    if max_rounds is not None and max_rounds > 0:
        total_rounds = max_rounds
    else:
        total_rounds = (total_hours * 60) // minutes_per_round

    tw_rowid = rd_rowid = pm_rowid = 0
    start_time = datetime.now()
    log_info(f"Starting synchronized simulation: {total_rounds} rounds")

    for round_num in range(start_round, total_rounds):
        if shutdown_event and shutdown_event.is_set():
            log_info(f"Shutdown signal at round {round_num + 1}")
            break

        sim_min = round_num * minutes_per_round
        simulated_hour = (sim_min // 60) % 24
        simulated_day = sim_min // (60 * 24) + 1

        round_memory.start_round(round_num, simulated_day, simulated_hour)
        memory_ctx = round_memory.build_context(round_num)
        market_prompt = bridge.get_market_prompt()
        sentiment_prompt = bridge.get_sentiment_prompt()

        platform_tasks = []
        for pr, pname, xp_excl in [
            (twitter_result, "twitter", "twitter"),
            (reddit_result, "reddit", "reddit"),
            (polymarket_result, "polymarket", "polymarket"),
        ]:
            if not pr:
                continue
            active = get_active_agents_for_round(pr.env, config, simulated_hour, round_num)
            if not active:
                continue
            for _, ag in active:
                if memory_ctx:
                    inject_round_memory(ag, memory_ctx)
                if pname != "polymarket" and market_prompt:
                    inject_market_context(ag, market_prompt)
                if pname == "polymarket":
                    if sentiment_prompt:
                        inject_sentiment_context(ag, sentiment_prompt)
                    if market_prompt:
                        inject_market_context(ag, market_prompt)
            if cross_platform_log:
                for aid, ag in active:
                    digest = cross_platform_log.build_digest(aid, exclude_platform=xp_excl)
                    if digest:
                        inject_cross_platform_context(ag, digest)
            if pname == "polymarket" and build_social_summary:
                social_summary = build_social_summary(round_memory, round_num, bridge)
                for _, ag in active:
                    if hasattr(ag, 'env') and hasattr(ag.env, 'extra_observation_context'):
                        ag.env.extra_observation_context = social_summary

            async def _step(p_result=pr, acts=active):
                await p_result.env.step({ag: LLMAction() for _, ag in acts})
                return acts

            platform_tasks.append((pname, _step()))

        platform_results = {}
        if platform_tasks:
            names, coros = zip(*platform_tasks)
            gathered = await asyncio.gather(*coros, return_exceptions=True)
            for name, res in zip(names, gathered):
                if isinstance(res, Exception):
                    log_info(f"[{name}] Round {round_num+1} failed: {res}")
                else:
                    platform_results[name] = res

        # Post-round: fetch, update beliefs, log
        if "twitter" in platform_results and twitter_result:
            acts, tw_rowid = fetch_new_actions_from_db(twitter_db, tw_rowid, agent_names)
            if twitter_belief:
                twitter_belief.after_round(twitter_db, twitter_result.env, platform_results["twitter"], round_num, acts)
                bridge.update_sentiment(twitter_belief.belief_states, acts, round_num, "twitter")
            if cross_platform_log and acts:
                cross_platform_log.record("twitter", acts)
            round_memory.record("twitter", round_num, acts)
            if twitter_logger:
                twitter_logger.log_round_start(round_num + 1, simulated_hour)
                for a in acts:
                    twitter_logger.log_action(round_num=round_num+1, agent_id=a['agent_id'],
                        agent_name=a['agent_name'], action_type=a['action_type'], action_args=a['action_args'])
                twitter_logger.log_round_end(round_num + 1, len(acts))
            twitter_result.total_actions += len(acts)

        if "reddit" in platform_results and reddit_result:
            acts, rd_rowid = fetch_new_actions_from_db(reddit_db, rd_rowid, agent_names)
            if reddit_belief:
                reddit_belief.after_round(reddit_db, reddit_result.env, platform_results["reddit"], round_num, acts)
                bridge.update_sentiment(reddit_belief.belief_states, acts, round_num, "reddit")
            if cross_platform_log and acts:
                cross_platform_log.record("reddit", acts)
            round_memory.record("reddit", round_num, acts)
            if reddit_logger:
                reddit_logger.log_round_start(round_num + 1, simulated_hour)
                for a in acts:
                    reddit_logger.log_action(round_num=round_num+1, agent_id=a['agent_id'],
                        agent_name=a['agent_name'], action_type=a['action_type'], action_args=a['action_args'])
                reddit_logger.log_round_end(round_num + 1, len(acts))
            reddit_result.total_actions += len(acts)

        if "polymarket" in platform_results and polymarket_result:
            bridge.update_prices(polymarket_db, round_num)
            acts, pm_rowid = fetch_polymarket_actions_from_db(polymarket_db, pm_rowid, agent_names)
            if polymarket_belief:
                polymarket_belief.after_round(polymarket_db, polymarket_result.env, platform_results["polymarket"], round_num, acts)
            if cross_platform_log and acts:
                cross_platform_log.record("polymarket", acts)
            round_memory.record("polymarket", round_num, acts)
            if polymarket_logger:
                polymarket_logger.log_round_start(round_num + 1, simulated_hour)
                for a in acts:
                    polymarket_logger.log_action(round_num=round_num+1, agent_id=a['agent_id'],
                        agent_name=a['agent_name'], action_type=a['action_type'], action_args=a['action_args'])
                polymarket_logger.log_round_end(round_num + 1, len(acts))
            polymarket_result.total_actions += len(acts)

        await round_memory.compact_previous_round(round_num)

        if (round_num + 1) % 5 == 0 or round_num == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = (round_num + 1) / total_rounds * 100
            parts = [f"X:{twitter_result.total_actions}"] if twitter_result else []
            if reddit_result:
                parts.append(f"R:{reddit_result.total_actions}")
            if polymarket_result:
                parts.append(f"PM:{polymarket_result.total_actions}")
            log_info(f"Round {round_num+1}/{total_rounds} ({progress:.0f}%) "
                     f"Day {simulated_day} {simulated_hour:02d}:00 — {' '.join(parts)} — {elapsed:.0f}s")

    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Simulation complete! {elapsed:.0f}s total")

    for tracker, name in [(twitter_belief, "Twitter"), (reddit_belief, "Reddit"), (polymarket_belief, "Polymarket")]:
        if tracker:
            log_info(f"[{name}] Trajectory saved: {tracker.save_trajectory()}")
            log_info(f"[{name}] {tracker.get_summary()}")

    for logger in (twitter_logger, reddit_logger, polymarket_logger):
        if logger:
            logger.log_simulation_end(total_rounds, 0)

    return twitter_result, reddit_result, polymarket_result
