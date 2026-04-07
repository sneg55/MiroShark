"""
Polymarket prediction market simulation runner.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import wonderwall
from wonderwall import LLMAction, ManualAction, AgentGraph
from wonderwall.social_agent.agent import SocialAgent
from wonderwall.social_platform.config import UserInfo
from wonderwall.simulations.polymarket import polymarket_simulation

from action_logger import PlatformActionLogger, SimulationLogManager
from belief_integration import BeliefTracker
from cross_platform_digest import CrossPlatformLog, inject_cross_platform_context
from market_media_bridge import MarketMediaBridge, inject_sentiment_context, inject_market_context
from sim_db_fetch import fetch_polymarket_actions_from_db
from sim_twitter_runner import PlatformSimulation


def load_polymarket_profiles(profile_path: str) -> List[Dict[str, Any]]:
    """Load Polymarket agent profiles from a JSON file."""
    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_polymarket_agent_graph(profiles: List[Dict[str, Any]], model) -> AgentGraph:
    """Build an AgentGraph for Polymarket from profile dicts."""
    agent_graph = AgentGraph()
    for profile in profiles:
        agent_id = profile["user_id"]
        display_name = profile.get("display_name") or profile.get("description", "") or profile["name"]
        user_info = UserInfo(
            name=display_name,
            description=profile.get("description", ""),
            profile={
                "other_info": {
                    "user_profile": profile.get("user_profile", ""),
                    "risk_tolerance": profile.get("risk_tolerance", "moderate"),
                }
            },
        )
        agent = SocialAgent(
            agent_id=agent_id, user_info=user_info, model=model,
            agent_graph=agent_graph, simulation=polymarket_simulation,
        )
        agent_graph.add_agent(agent)
    return agent_graph


async def run_polymarket_simulation(
    config: Dict[str, Any],
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None,
    start_round: int = 0,
    cross_platform_log: Optional[CrossPlatformLog] = None,
    market_media_bridge: Optional[MarketMediaBridge] = None,
    create_model=None,
    get_agent_names_from_config=None,
    get_active_agents_for_round=None,
    shutdown_event=None,
) -> PlatformSimulation:
    """Run the Polymarket simulation loop and return a PlatformSimulation."""
    result = PlatformSimulation()

    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Polymarket] {msg}")
        print(f"[Polymarket] {msg}")

    log_info("Initializing...")
    model = create_model(config, use_boost=False)

    profile_path = os.path.join(simulation_dir, "polymarket_profiles.json")
    if not os.path.exists(profile_path):
        log_info(f"Error: Profile file not found: {profile_path}")
        return result

    profiles = load_polymarket_profiles(profile_path)
    result.agent_graph = build_polymarket_agent_graph(profiles, model)

    agent_names = get_agent_names_from_config(config)
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')

    is_resume = start_round > 0
    db_path = os.path.join(simulation_dir, "polymarket_simulation.db")
    if not is_resume and os.path.exists(db_path):
        os.remove(db_path)

    result.env = wonderwall.make(
        agent_graph=result.agent_graph,
        simulation=polymarket_simulation,
        database_path=db_path,
        semaphore=60,
    )
    await result.env.reset()
    log_info("Environment started" + (f" (resuming from round {start_round})" if is_resume else ""))

    if action_logger:
        action_logger.log_simulation_start(config)

    # Seed initial markets (round 0)
    if not is_resume:
        event_config = config.get("event_config", {})
        initial_markets = event_config.get("initial_markets", [])
        if action_logger:
            action_logger.log_round_start(0, 0)
        initial_action_count = 0
        if initial_markets:
            agent_0 = result.env.agent_graph.get_agent(0)
            seed_actions = []
            for market in initial_markets:
                seed_actions.append(ManualAction(
                    action_type="create_market",
                    action_args={
                        "question": market.get("question", ""),
                        "outcome_a": market.get("outcome_a", "YES"),
                        "outcome_b": market.get("outcome_b", "NO"),
                    },
                ))
                initial_action_count += 1
            await result.env.step({agent_0: seed_actions})
            log_info(f"Seeded {len(seed_actions)} initial markets")
            if action_logger:
                for market in initial_markets:
                    action_logger.log_action(
                        round_num=0, agent_id=0,
                        agent_name=agent_names.get(0, "Agent_0"),
                        action_type="CREATE_MARKET",
                        action_args={"question": market.get("question", "")},
                    )
        if action_logger:
            action_logger.log_round_end(0, initial_action_count)

    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    # max_rounds is the target, not a ceiling — SaaS layer controls round count
    if max_rounds is not None and max_rounds > 0:
        total_rounds = max_rounds
    else:
        total_rounds = (total_hours * 60) // minutes_per_round

    belief_tracker = BeliefTracker(config, simulation_dir, "polymarket")
    log_info(f"Belief tracking: {len(belief_tracker.topics)} topics")
    start_time = datetime.now()
    total_actions = 0
    last_rowid = 0

    if start_round > 0:
        log_info(f"Resuming from round {start_round}")

    for round_num in range(start_round, total_rounds):
        if shutdown_event and shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"Shutdown signal, stopping at round {round_num + 1}")
            break

        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1

        active_agents = get_active_agents_for_round(result.env, config, simulated_hour, round_num)
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)
        if not active_agents:
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue

        if cross_platform_log:
            for agent_id, agent in active_agents:
                digest = cross_platform_log.build_digest(agent_id, exclude_platform="polymarket")
                if digest:
                    inject_cross_platform_context(agent, digest)
        if market_media_bridge:
            sentiment_prompt = market_media_bridge.get_sentiment_prompt()
            if sentiment_prompt:
                for _, agent in active_agents:
                    inject_sentiment_context(agent, sentiment_prompt)
            market_prompt = market_media_bridge.get_market_prompt()
            if market_prompt:
                for _, agent in active_agents:
                    inject_market_context(agent, market_prompt)

        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)

        actual_actions, last_rowid = fetch_polymarket_actions_from_db(db_path, last_rowid, agent_names)
        if market_media_bridge:
            market_media_bridge.update_prices(db_path, round_num)
        belief_tracker.after_round(db_path, result.env, active_agents, round_num, actual_actions)
        if cross_platform_log and actual_actions:
            cross_platform_log.record("polymarket", actual_actions)

        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1, agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'], action_type=action_data['action_type'],
                    action_args=action_data['action_args'],
                )
                total_actions += 1
                round_action_count += 1
        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)

        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - "
                     f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")

    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)

    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Simulation loop completed! Elapsed: {elapsed:.1f}s, total actions: {total_actions}")

    traj_path = belief_tracker.save_trajectory()
    log_info(f"Belief trajectory saved: {traj_path}")
    log_info(belief_tracker.get_summary())

    return result
