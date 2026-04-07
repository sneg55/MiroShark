"""
Reddit platform simulation runner.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import wonderwall
from wonderwall import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph

from action_logger import PlatformActionLogger, SimulationLogManager
from belief_integration import BeliefTracker
from cross_platform_digest import CrossPlatformLog, inject_cross_platform_context
from market_media_bridge import MarketMediaBridge, inject_market_context
from sim_constants import REDDIT_ACTIONS
from sim_db_fetch import fetch_new_actions_from_db
from sim_twitter_runner import PlatformSimulation


async def run_reddit_simulation(
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
    """Run the Reddit simulation loop and return a PlatformSimulation."""
    result = PlatformSimulation()

    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Reddit] {msg}")
        print(f"[Reddit] {msg}")

    log_info("Initializing...")
    model = create_model(config, use_boost=True)

    profile_path = os.path.join(simulation_dir, "reddit_profiles.json")
    if not os.path.exists(profile_path):
        log_info(f"Error: Profile file not found: {profile_path}")
        return result

    result.agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path, model=model, available_actions=REDDIT_ACTIONS,
    )

    agent_names = get_agent_names_from_config(config)
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')

    is_resume = start_round > 0
    db_path = os.path.join(simulation_dir, "reddit_simulation.db")
    if not is_resume and os.path.exists(db_path):
        os.remove(db_path)

    result.env = wonderwall.make(
        agent_graph=result.agent_graph,
        platform=wonderwall.DefaultPlatformType.REDDIT,
        database_path=db_path,
        semaphore=60,
    )
    await result.env.reset()
    log_info("Environment started" + (f" (resuming from round {start_round})" if is_resume else ""))

    if action_logger:
        action_logger.log_simulation_start(config)

    total_actions = 0
    last_rowid = 0

    # Execute initial events (skip if resuming)
    if not is_resume:
        event_config = config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])
        if action_logger:
            action_logger.log_round_start(0, 0)
        initial_action_count = 0
        if initial_posts:
            initial_actions = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = result.env.agent_graph.get_agent(agent_id)
                    if agent in initial_actions:
                        if not isinstance(initial_actions[agent], list):
                            initial_actions[agent] = [initial_actions[agent]]
                        initial_actions[agent].append(ManualAction(
                            action_type=ActionType.CREATE_POST,
                            action_args={"content": content},
                        ))
                    else:
                        initial_actions[agent] = ManualAction(
                            action_type=ActionType.CREATE_POST, action_args={"content": content},
                        )
                    if action_logger:
                        action_logger.log_action(
                            round_num=0, agent_id=agent_id,
                            agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                            action_type="CREATE_POST", action_args={"content": content},
                        )
                        total_actions += 1
                        initial_action_count += 1
                except Exception:
                    pass
            if initial_actions:
                await result.env.step(initial_actions)
                log_info(f"Published {len(initial_actions)} initial posts")
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

    belief_tracker = BeliefTracker(config, simulation_dir, "reddit")
    log_info(f"Belief tracking: {len(belief_tracker.topics)} topics")
    start_time = datetime.now()

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
                digest = cross_platform_log.build_digest(agent_id, exclude_platform="reddit")
                if digest:
                    inject_cross_platform_context(agent, digest)
        if market_media_bridge:
            market_prompt = market_media_bridge.get_market_prompt()
            if market_prompt:
                for _, agent in active_agents:
                    inject_market_context(agent, market_prompt)

        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)

        actual_actions, last_rowid = fetch_new_actions_from_db(db_path, last_rowid, agent_names)
        belief_tracker.after_round(db_path, result.env, active_agents, round_num, actual_actions)
        if market_media_bridge:
            market_media_bridge.update_sentiment(
                belief_tracker.belief_states, actual_actions, round_num, "reddit"
            )
        if cross_platform_log and actual_actions:
            cross_platform_log.record("reddit", actual_actions)

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
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")

    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)

    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Simulation loop completed! Elapsed: {elapsed:.1f}s, total actions: {total_actions}")

    traj_path = belief_tracker.save_trajectory()
    log_info(f"Belief trajectory saved: {traj_path}")
    log_info(belief_tracker.get_summary())

    return result
