"""
Wonderwall multi-platform parallel simulation — launcher.

Applies the Windows UTF-8 fix before any other imports, then wires up the
business-logic helpers and calls sim_main.main().

Usage:
    python run_parallel_simulation.py --config simulation_config.json
    python run_parallel_simulation.py --config simulation_config.json --cross-platform
    python run_parallel_simulation.py --config simulation_config.json --no-wait
    python run_parallel_simulation.py --config simulation_config.json --twitter-only
    python run_parallel_simulation.py --config simulation_config.json --reddit-only
    python run_parallel_simulation.py --config simulation_config.json --polymarket-only
"""

# ============================================================
# Windows UTF-8 fix — must run before all other imports
# ============================================================
import sys
import os

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    import builtins
    _original_open = builtins.open

    def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None,
                   newline=None, closefd=True, opener=None):
        if encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    builtins.open = _utf8_open

# ============================================================
# Path setup — add backend/ and scripts/ so local modules resolve
# ============================================================
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"Loaded environment config: {_env_file}")
else:
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"Loaded environment config: {_backend_env}")

# ============================================================
# Standard library + third-party imports
# ============================================================
import asyncio
import json
import signal
import warnings
from typing import Dict, Any, List, Optional

# ============================================================
# Business-logic helpers (defined here, injected into sim_main)
# ============================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a JSON simulation config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_agent_names_from_config(config: Dict[str, Any]) -> Dict[int, str]:
    """Return agent_id -> entity_name mapping from simulation config."""
    return {
        cfg["agent_id"]: cfg.get("entity_name", f"Agent_{cfg['agent_id']}")
        for cfg in config.get("agent_configs", [])
        if cfg.get("agent_id") is not None
    }


def create_model(config: Dict[str, Any], use_boost: bool = False):
    """
    Create and return a camel-ai LLM model.

    Supports a primary LLM (LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME)
    and an optional boost LLM (LLM_BOOST_*) for parallel platforms.
    """
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    boost_api_key = os.environ.get("LLM_BOOST_API_KEY", "")
    has_boost = bool(boost_api_key)

    if use_boost and has_boost:
        llm_api_key = boost_api_key
        llm_base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
        llm_model = os.environ.get("LLM_BOOST_MODEL_NAME", "") or os.environ.get("LLM_MODEL_NAME", "")
        label = "[Boost LLM]"
    else:
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        label = "[General LLM]"

    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")
    if llm_api_key:
        os.environ["OPENAI_API_KEY"] = llm_api_key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Missing API Key. Set LLM_API_KEY in .env")
    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url

    print(f"{label} model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else 'default'}...")
    return ModelFactory.create(model_platform=ModelPlatformType.OPENAI, model_type=llm_model)


def get_active_agents_for_round(env, config: Dict[str, Any], current_hour: int, round_num: int) -> List:
    """Return all agents as active for every round.

    The LLM has do_nothing as an available action — it decides whether to act,
    not a coin flip. This eliminates the activity_level, active_hours, and
    agents_per_hour starvation that caused sims to produce near-zero actions.
    """
    agent_configs = config.get("agent_configs", [])
    active_agents = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        try:
            agent = env.agent_graph.get_agent(agent_id)
            active_agents.append((agent_id, agent))
        except Exception:
            pass
    return active_agents


def _build_social_summary_for_traders(round_memory, current_round: int, bridge) -> str:
    """Build a concise social media summary for Polymarket traders' observation prompt."""
    parts = []
    prev_round = current_round - 1
    if prev_round >= 0 and prev_round in round_memory._rounds:
        rec = round_memory._rounds[prev_round]
        for platform in ("twitter", "reddit"):
            actions = rec.platform_actions.get(platform, [])
            content_actions = [
                a for a in actions
                if a.get("action_type") in ("CREATE_POST", "CREATE_COMMENT", "QUOTE_POST")
                and a.get("action_args", {}).get("content")
            ]
            if content_actions:
                parts.append(f"[{platform.title()} — last round]")
                for a in content_actions[:4]:
                    agent = a.get("agent_name", "?")
                    content = a["action_args"]["content"][:150]
                    parts.append(f'  {agent}: "{content}"')
    if bridge and bridge.latest_sentiment and bridge.latest_sentiment.topic_sentiments:
        for topic, data in bridge.latest_sentiment.topic_sentiments.items():
            pos = data.get("positive_pct", 0)
            neg = data.get("negative_pct", 0)
            count = data.get("post_count", 0)
            if count > 0:
                mood = "bullish" if pos > neg + 10 else "bearish" if neg > pos + 10 else "mixed"
                parts.append(f"  Sentiment on \"{topic}\": {mood} ({pos:.0f}% pos, {neg:.0f}% neg, {count} posts)")
    return "\n".join(parts)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    from sim_main import setup_signal_handlers, main as _main
    import sim_main as _sim_main

    _shutdown_event = asyncio.Event()
    _sim_main._shutdown_event = _shutdown_event
    setup_signal_handlers()

    try:
        asyncio.run(_main(
            load_config=load_config,
            get_agent_names_from_config=get_agent_names_from_config,
            create_model=create_model,
            get_active_agents_for_round=get_active_agents_for_round,
            build_social_summary=_build_social_summary_for_traders,
            shutdown_event=_shutdown_event,
        ))
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except SystemExit:
        pass
    finally:
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("Simulation process exited")
