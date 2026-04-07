"""
Setup helpers for the synchronized multi-platform simulation.

Initialises platform environments, belief trackers, and seeds initial events
(posts + Polymarket markets) before the main round loop starts.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import wonderwall
from wonderwall import ActionType, ManualAction
from wonderwall import generate_twitter_agent_graph, generate_reddit_agent_graph
from wonderwall.simulations.polymarket import polymarket_simulation

from belief_integration import BeliefTracker
from round_memory import RoundMemory
from sim_constants import TWITTER_ACTIONS, REDDIT_ACTIONS
from sim_polymarket_runner import load_polymarket_profiles, build_polymarket_agent_graph
from sim_twitter_runner import PlatformSimulation


async def setup_platform_envs(
    config: Dict[str, Any],
    simulation_dir: str,
    model,
    agent_names: Dict[int, str],
    start_round: int,
    has_twitter: bool,
    has_reddit: bool,
    has_polymarket: bool,
    log_info,
) -> Tuple[Optional[PlatformSimulation], Optional[PlatformSimulation], Optional[PlatformSimulation]]:
    """Initialise all platform environments in parallel and return result containers."""
    twitter_result = reddit_result = polymarket_result = None
    twitter_db = os.path.join(simulation_dir, "twitter_simulation.db")
    reddit_db = os.path.join(simulation_dir, "reddit_simulation.db")
    polymarket_db = os.path.join(simulation_dir, "polymarket_simulation.db")

    if has_twitter:
        twitter_result = PlatformSimulation()
        twitter_result.agent_graph = await generate_twitter_agent_graph(
            profile_path=os.path.join(simulation_dir, "twitter_profiles.csv"),
            model=model, available_actions=TWITTER_ACTIONS,
        )
        for aid, ag in twitter_result.agent_graph.get_agents():
            agent_names.setdefault(aid, getattr(ag, 'name', f'Agent_{aid}'))
        if start_round == 0 and os.path.exists(twitter_db):
            os.remove(twitter_db)
        twitter_result.env = wonderwall.make(
            agent_graph=twitter_result.agent_graph,
            platform=wonderwall.DefaultPlatformType.TWITTER,
            database_path=twitter_db, semaphore=60,
        )
        await twitter_result.env.reset()
        log_info("[Twitter] Environment ready")

    if has_reddit:
        reddit_result = PlatformSimulation()
        reddit_result.agent_graph = await generate_reddit_agent_graph(
            profile_path=os.path.join(simulation_dir, "reddit_profiles.json"),
            model=model, available_actions=REDDIT_ACTIONS,
        )
        for aid, ag in reddit_result.agent_graph.get_agents():
            agent_names.setdefault(aid, getattr(ag, 'name', f'Agent_{aid}'))
        if start_round == 0 and os.path.exists(reddit_db):
            os.remove(reddit_db)
        reddit_result.env = wonderwall.make(
            agent_graph=reddit_result.agent_graph,
            platform=wonderwall.DefaultPlatformType.REDDIT,
            database_path=reddit_db, semaphore=60,
        )
        await reddit_result.env.reset()
        log_info("[Reddit] Environment ready")

    if has_polymarket:
        polymarket_result = PlatformSimulation()
        profiles = load_polymarket_profiles(os.path.join(simulation_dir, "polymarket_profiles.json"))
        polymarket_result.agent_graph = build_polymarket_agent_graph(profiles, model)
        for aid, ag in polymarket_result.agent_graph.get_agents():
            agent_names.setdefault(aid, getattr(ag, 'name', f'Agent_{aid}'))
        if start_round == 0 and os.path.exists(polymarket_db):
            os.remove(polymarket_db)
        polymarket_result.env = wonderwall.make(
            agent_graph=polymarket_result.agent_graph,
            simulation=polymarket_simulation,
            database_path=polymarket_db, semaphore=60,
        )
        await polymarket_result.env.reset()
        log_info("[Polymarket] Environment ready")

    return twitter_result, reddit_result, polymarket_result


async def seed_initial_events(
    config: Dict[str, Any],
    start_round: int,
    twitter_result: Optional[PlatformSimulation],
    reddit_result: Optional[PlatformSimulation],
    polymarket_result: Optional[PlatformSimulation],
    log_info,
) -> None:
    """Seed initial posts and Polymarket markets (round 0 only)."""
    event_config = config.get("event_config", {})

    # Initial social posts
    initial_posts = event_config.get("initial_posts", [])
    if start_round == 0 and initial_posts:
        for pr, pname in [(twitter_result, "twitter"), (reddit_result, "reddit")]:
            if not pr:
                continue
            ia = {}
            for post in initial_posts:
                aid = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    ia[pr.env.agent_graph.get_agent(aid)] = ManualAction(
                        action_type=ActionType.CREATE_POST, action_args={"content": content}
                    )
                except Exception:
                    pass
            if ia:
                await pr.env.step(ia)
                log_info(f"[{pname.capitalize()}] Published {len(ia)} initial posts")

    # Initial Polymarket markets
    initial_markets = event_config.get("initial_markets", [])
    if start_round == 0 and initial_markets and polymarket_result:
        agent_0 = polymarket_result.env.agent_graph.get_agent(0)
        seed_actions = [
            ManualAction(
                action_type="create_market",
                action_args={
                    "question": m.get("question", ""),
                    "outcome_a": m.get("outcome_a", "YES"),
                    "outcome_b": m.get("outcome_b", "NO"),
                    "initial_probability": m.get("initial_probability", 0.5),
                },
            )
            for m in initial_markets
        ]
        if seed_actions:
            await polymarket_result.env.step({agent_0: seed_actions})
            for m in initial_markets:
                log_info(f"  Market: \"{m['question'][:60]}...\" ({m.get('initial_probability', 0.5):.0%})")
            log_info(f"[Polymarket] Seeded {len(seed_actions)} markets")


def build_belief_trackers(
    config: Dict[str, Any],
    simulation_dir: str,
    has_twitter: bool,
    has_reddit: bool,
    has_polymarket: bool,
) -> Tuple[Optional[BeliefTracker], Optional[BeliefTracker], Optional[BeliefTracker]]:
    """Create belief tracker instances for each active platform."""
    return (
        BeliefTracker(config, simulation_dir, "twitter") if has_twitter else None,
        BeliefTracker(config, simulation_dir, "reddit") if has_reddit else None,
        BeliefTracker(config, simulation_dir, "polymarket") if has_polymarket else None,
    )


def build_round_memory(config: Dict[str, Any]) -> RoundMemory:
    """Create a RoundMemory instance, wiring in the LLM client if available."""
    try:
        from app.utils.llm_client import create_llm_client
        memory_llm = create_llm_client()
    except Exception:
        memory_llm = None
    time_config = config.get("time_config", {})
    return RoundMemory(
        llm_client=memory_llm,
        minutes_per_round=time_config.get("minutes_per_round", 60),
    )
