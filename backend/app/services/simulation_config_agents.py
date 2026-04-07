"""
Simulation config agents — LLM-based generation of agent configurations.

Contains both batch (1:1 entity) and multi-perspective (many agents per entity
plus observers) generation strategies.
"""

import json
from typing import Dict, Any, List

from ..utils.logger import get_logger
from .entity_reader import EntityNode
from .simulation_config_types import AgentActivityConfig
from .simulation_config_rules import generate_agent_config_by_rule
from .simulation_config_llm import SimulationConfigLLM

logger = get_logger('miroshark.simulation_config')


def generate_agent_configs_multi(llm: SimulationConfigLLM, context: str,
                                 entities: List[EntityNode],
                                 simulation_requirement: str,
                                 target_agents: int) -> List[AgentActivityConfig]:
    """Generate multi-perspective agent configs with observers.

    When target_agents > len(entities), the LLM allocates multiple
    representatives per important entity plus synthetic observer agents
    (journalists, analysts, bystanders).  Falls back to batch generation
    on failure.
    """
    summary_len = llm.AGENT_SUMMARY_LENGTH
    entity_list = [{
        "name": e.name,
        "type": e.get_entity_type() or "Unknown",
        "summary": (e.summary[:summary_len] + "...")
                  if len(e.summary) > summary_len else e.summary,
    } for e in entities]

    entity_list_json = json.dumps(entity_list, ensure_ascii=False, indent=2)

    prompt = f"""Based on the following simulation scenario, generate {target_agents} diverse agent profiles.

Simulation requirement: {simulation_requirement}

## Entities from knowledge graph
{entity_list_json}

## Rules
1. Important entities should have MULTIPLE representatives with different perspectives.
   Example: "Apple" → a CEO (strategic), an engineer (technical), a PR lead (public messaging).
2. Not all agents should be direct stakeholders. Include OUTSIDE OBSERVERS who watch, analyze,
   and react without being directly involved — journalists, industry analysts, commentators,
   affected bystanders relevant to this specific topic.
3. Each agent must have a UNIQUE name (real or realistic), a clear role, and a distinct viewpoint.
4. Generate exactly {target_agents} agents total.
5. Every agent must be linked to a source entity OR marked as an observer.

Return JSON (no markdown):
{{
    "agents": [
        {{
            "agent_id": 0,
            "entity_name": "<source entity name or 'Observer'>",
            "name": "<unique agent name>",
            "role": "<specific role/title>",
            "activity_level": <0.1-0.9>,
            "posts_per_hour": <float>,
            "comments_per_hour": <float>,
            "response_delay_min": <int minutes>,
            "response_delay_max": <int minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <0.5-3.0>
        }}
    ]
}}"""

    system_prompt = (
        "You are a social media simulation designer. Return pure JSON.\n\n"
        "AGENT DESIGN PRINCIPLES:\n"
        "- Stakeholder agents represent specific people or roles within an entity.\n"
        "- Observer agents are independent voices: journalists cover the story, "
        "analysts evaluate impact, consumers react, regulators watch.\n"
        "- Diversity of stance is critical — not everyone agrees. "
        "Include supporters, opponents, and skeptics.\n"
        "- influence_weight: 2.0-3.0 for executives/institutions, "
        "1.0-2.0 for experts/journalists, 0.5-1.0 for regular people.\n"
        "- stance must reflect realistic positions, not random assignment."
    )

    # Build entity lookup by name for UUID resolution
    entity_by_name: Dict[str, EntityNode] = {e.name.lower(): e for e in entities}

    try:
        result = llm.call_llm_with_retry(prompt, system_prompt)
        agents_raw = result.get("agents", [])
        if not agents_raw:
            raise ValueError("LLM returned empty agents list")

        configs = []
        for idx, agent in enumerate(agents_raw):
            # Resolve source entity
            source_name = agent.get("entity_name", "Observer")
            matched_entity = entity_by_name.get(source_name.lower())

            if matched_entity:
                entity_uuid = matched_entity.uuid
                entity_type = matched_entity.get_entity_type() or "Unknown"
            else:
                entity_uuid = ""
                entity_type = "Observer"

            configs.append(AgentActivityConfig(
                agent_id=idx,
                entity_uuid=entity_uuid,
                entity_name=agent.get("name", f"Agent_{idx}"),
                entity_type=entity_type,
                activity_level=agent.get("activity_level", 0.5),
                posts_per_hour=agent.get("posts_per_hour", 0.5),
                comments_per_hour=agent.get("comments_per_hour", 1.0),
                active_hours=list(range(0, 24)),
                response_delay_min=agent.get("response_delay_min", 5),
                response_delay_max=agent.get("response_delay_max", 60),
                sentiment_bias=agent.get("sentiment_bias", 0.0),
                stance=agent.get("stance", "neutral"),
                influence_weight=agent.get("influence_weight", 1.0),
            ))

        logger.info(
            f"Multi-perspective generation: {len(configs)} agents "
            f"({len([c for c in configs if c.entity_type == 'Observer'])} observers)")
        return configs

    except Exception as e:
        logger.warning(
            f"Multi-perspective agent generation failed: {e}, "
            f"falling back to batch generation")
        # Fallback: batch generation (1:1 entity mapping)
        return generate_agent_configs_batch(
            llm, context, entities, 0, simulation_requirement)


def generate_agent_configs_batch(llm: SimulationConfigLLM, context: str,
                                 entities: List[EntityNode], start_idx: int,
                                 simulation_requirement: str,
                                 ) -> List[AgentActivityConfig]:
    """Batch generate Agent configurations (1:1 entity mapping)."""
    summary_len = llm.AGENT_SUMMARY_LENGTH
    entity_list = [{
        "agent_id": start_idx + i,
        "entity_name": e.name,
        "entity_type": e.get_entity_type() or "Unknown",
        "summary": e.summary[:summary_len] if e.summary else "",
    } for i, e in enumerate(entities)]

    prompt = f"""Generate social media activity configuration for each entity.

Simulation requirement: {simulation_requirement}

## Entity List
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Return JSON (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match>, "activity_level": <0-1>,
            "posts_per_hour": <float>, "comments_per_hour": <float>,
            "active_hours": [<hours>],
            "response_delay_min": <int>, "response_delay_max": <int>,
            "sentiment_bias": <-1 to 1>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <float>
        }}, ...
    ]
}}"""

    system_prompt = (
        "You are a social media behavior analyst. Return pure JSON.\n\n"
        "HEURISTICS:\n"
        "- Institutions: 0.5-1/hr, high influence. Journalists: 2-4/hr.\n"
        "- Activists: 3-5/hr, strong sentiment. Regular people: 0.3-1/hr.\n"
        "- influence_weight: 2-3 institutions, 1-2 experts, 0.5-1 individuals.\n"
    )

    try:
        result = llm.call_llm_with_retry(prompt, system_prompt)
        llm_configs = {
            cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])
        }
    except Exception as e:
        logger.warning(f"Agent config batch LLM failed: {e}, using rule-based")
        llm_configs = {}

    configs = []
    for i, entity in enumerate(entities):
        agent_id = start_idx + i
        cfg = llm_configs.get(agent_id, {})
        if not cfg:
            cfg = generate_agent_config_by_rule(entity)
        configs.append(AgentActivityConfig(
            agent_id=agent_id, entity_uuid=entity.uuid,
            entity_name=entity.name,
            entity_type=entity.get_entity_type() or "Unknown",
            activity_level=cfg.get("activity_level", 0.5),
            posts_per_hour=cfg.get("posts_per_hour", 0.5),
            comments_per_hour=cfg.get("comments_per_hour", 1.0),
            active_hours=cfg.get("active_hours", list(range(9, 23))),
            response_delay_min=cfg.get("response_delay_min", 5),
            response_delay_max=cfg.get("response_delay_max", 60),
            sentiment_bias=cfg.get("sentiment_bias", 0.0),
            stance=cfg.get("stance", "neutral"),
            influence_weight=cfg.get("influence_weight", 1.0),
        ))
    return configs
