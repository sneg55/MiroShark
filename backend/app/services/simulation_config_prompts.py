"""
Simulation config prompts — LLM-based generation methods for
time, event, and prediction market configurations.
"""

from typing import Dict, Any, List

from ..utils.logger import get_logger
from .entity_reader import EntityNode
from .simulation_config_types import EventConfig
from .simulation_config_rules import get_default_time_config
from .simulation_config_llm import SimulationConfigLLM

logger = get_logger('miroshark.simulation_config')


def generate_time_config(llm: SimulationConfigLLM, context: str,
                         num_entities: int) -> Dict[str, Any]:
    """Generate time configuration via LLM."""
    context_truncated = context[:llm.TIME_CONFIG_CONTEXT_LENGTH]
    max_agents_allowed = max(1, int(num_entities * 0.9))

    prompt = f"""Based on the following simulation requirements, generate a time simulation configuration.

{context_truncated}

## Task
Please generate a time configuration JSON.

### Basic Principles (for reference only):
- Midnight 0-5am almost no activity (coefficient 0.05)
- Morning 6-8am gradually active (coefficient 0.4)
- Work hours 9am-6pm moderate (coefficient 0.7)
- Evening 7-10pm peak (coefficient 1.5)
- After 11pm declining (coefficient 0.5)

### Return JSON format (no markdown)
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Time configuration explanation for this event"
}}

Field descriptions:
- total_simulation_hours (int): 24-168 hours
- minutes_per_round (int): 30-120 minutes, recommended 60
- agents_per_hour_min/max (int): range 1-{max_agents_allowed}
- peak_hours, off_peak_hours, morning_hours, work_hours: hour arrays
- reasoning (string): Brief explanation"""

    system_prompt = (
        "You are a social media simulation architect. Return pure JSON.\n\n"
        "TIMING HEURISTICS:\n"
        "- Breaking news: short rounds (15-30 min), 24-48h, high activity\n"
        "- Product launch: medium rounds (30-60 min), 48-72h, front-loaded\n"
        "- Policy debate: long rounds (60-120 min), 72-168h, steady\n"
        "- Peak: 8-10 AM and 6-9 PM. Quiet: 12-6 AM.\n"
        "- More agents = lower per-agent activity.\n"
        "- Should feel like real-time social media."
    )

    try:
        return llm.call_llm_with_retry(prompt, system_prompt)
    except Exception as e:
        logger.warning(f"Time config LLM generation failed: {e}, using default")
        return get_default_time_config(num_entities)


def generate_event_config(llm: SimulationConfigLLM, context: str,
                          simulation_requirement: str,
                          entities: List[EntityNode]) -> Dict[str, Any]:
    """Generate event configuration via LLM."""
    type_examples: Dict[str, list] = {}
    for e in entities:
        etype = e.get_entity_type() or "Unknown"
        if etype not in type_examples:
            type_examples[etype] = []
        if len(type_examples[etype]) < 3:
            type_examples[etype].append(e.name)

    type_info = "\n".join([
        f"- {t}: {', '.join(examples)}"
        for t, examples in type_examples.items()
    ])
    context_truncated = context[:llm.EVENT_CONFIG_CONTEXT_LENGTH]

    prompt = f"""Based on the following simulation requirements, generate event configuration.

Simulation requirement: {simulation_requirement}

{context_truncated}

## Available Entity Types and Examples
{type_info}

## Task
Generate event configuration JSON with hot topics, narrative direction, and initial posts.
**Important**: poster_type must be from "Available Entity Types" above.

Return JSON format (no markdown):
{{
    "hot_topics": ["keyword1", "keyword2", ...],
    "narrative_direction": "<public opinion direction>",
    "initial_posts": [
        {{"content": "post content", "poster_type": "entity type"}}, ...
    ],
    "reasoning": "<brief explanation>"
}}"""

    system_prompt = (
        "You are a public opinion simulation designer. Return pure JSON.\n\n"
        "EVENT DESIGN HEURISTICS:\n"
        "- Initial posts should feel organic, not like press releases.\n"
        "- First poster should be whoever would realistically learn first.\n"
        "- Schedule 2-3 plot twists mid-simulation.\n"
        "- poster_type must exactly match available entity types.\n"
        "- Narrative direction should have tension."
    )

    try:
        return llm.call_llm_with_retry(prompt, system_prompt)
    except Exception as e:
        logger.warning(f"Event config LLM generation failed: {e}, using default")
        return {
            "hot_topics": [], "narrative_direction": "",
            "initial_posts": [], "reasoning": "Using default configuration",
        }


def generate_prediction_markets(llm: SimulationConfigLLM, context: str,
                                simulation_requirement: str,
                                event_config: EventConfig) -> List[Dict[str, Any]]:
    """Generate prediction markets from simulation requirement."""
    hot_topics = event_config.hot_topics[:5] if event_config.hot_topics else []

    system_prompt = (
        "You are a prediction market designer. Return pure JSON.\n\n"
        "RULES:\n"
        "- Create exactly ONE prediction market as a YES/NO question\n"
        "- It must be SPECIFIC, TIME-BOUND, and RESOLVABLE\n"
        "- Set initial_probability to your best estimate (0.15-0.85)\n"
    )
    user_prompt = f"""Simulation: {simulation_requirement}

Hot topics: {', '.join(hot_topics) if hot_topics else 'N/A'}

Context:
{context[:2000]}

Generate ONE prediction market:
{{
  "markets": [
    {{
      "question": "Will X happen by Y?",
      "outcome_a": "YES", "outcome_b": "NO",
      "initial_probability": 0.65,
      "reasoning": "Why this question and probability"
    }}
  ]
}}"""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = llm.llm.chat_json(messages=messages, temperature=0.5)
        markets = []
        for m in result.get("markets", [])[:1]:
            question = m.get("question", "")
            if not question:
                continue
            prob = max(0.10, min(0.90, float(m.get("initial_probability", 0.5))))
            markets.append({
                "question": question,
                "outcome_a": m.get("outcome_a", "YES"),
                "outcome_b": m.get("outcome_b", "NO"),
                "initial_probability": prob,
                "reasoning": m.get("reasoning", ""),
            })
        logger.info(f"Generated {len(markets)} prediction markets")
        return markets
    except Exception as e:
        logger.warning(f"Prediction market generation failed: {e}")
        return [{
            "question": "Will the scenario have a net positive public reaction?",
            "outcome_a": "YES", "outcome_b": "NO",
            "initial_probability": 0.55,
            "reasoning": "Fallback — auto-generated from simulation requirement",
        }]
