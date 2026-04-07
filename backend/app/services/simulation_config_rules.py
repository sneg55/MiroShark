"""
Simulation config rules — rule-based fallback methods, time/event config parsing,
and agent post assignment logic.
"""

from typing import Dict, Any, List

from ..utils.logger import get_logger
from .entity_reader import EntityNode
from .simulation_config_types import (
    AgentActivityConfig,
    TimeSimulationConfig,
    EventConfig,
)

logger = get_logger('miroshark.simulation_config')


def get_default_time_config(num_entities: int) -> Dict[str, Any]:
    """Get default time configuration (typical daily schedule)."""
    return {
        "total_simulation_hours": 72,
        "minutes_per_round": 60,
        "agents_per_hour_min": max(1, num_entities // 15),
        "agents_per_hour_max": max(5, num_entities // 5),
        "peak_hours": [19, 20, 21, 22],
        "off_peak_hours": [0, 1, 2, 3, 4, 5],
        "morning_hours": [6, 7, 8],
        "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "reasoning": "Using default daily schedule configuration (1 hour per round)",
    }


def parse_time_config(result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
    """Parse time configuration result and validate agents_per_hour values."""
    agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
    agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))

    if agents_per_hour_min > num_entities:
        logger.warning(
            f"agents_per_hour_min ({agents_per_hour_min}) exceeds total Agent count "
            f"({num_entities}), corrected")
        agents_per_hour_min = max(1, num_entities // 10)

    if agents_per_hour_max > num_entities:
        logger.warning(
            f"agents_per_hour_max ({agents_per_hour_max}) exceeds total Agent count "
            f"({num_entities}), corrected")
        agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

    if agents_per_hour_min >= agents_per_hour_max:
        agents_per_hour_min = max(1, agents_per_hour_max // 2)
        logger.warning(f"agents_per_hour_min >= max, corrected to {agents_per_hour_min}")

    return TimeSimulationConfig(
        total_simulation_hours=result.get("total_simulation_hours", 72),
        minutes_per_round=result.get("minutes_per_round", 60),
        agents_per_hour_min=agents_per_hour_min,
        agents_per_hour_max=agents_per_hour_max,
        peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
        off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
        off_peak_activity_multiplier=0.05,
        morning_hours=result.get("morning_hours", [6, 7, 8]),
        morning_activity_multiplier=0.4,
        work_hours=result.get("work_hours", list(range(9, 19))),
        work_activity_multiplier=0.7,
        peak_activity_multiplier=1.5,
    )


def parse_event_config(result: Dict[str, Any]) -> EventConfig:
    """Parse event configuration result."""
    return EventConfig(
        initial_posts=result.get("initial_posts", []),
        scheduled_events=[],
        hot_topics=result.get("hot_topics", []),
        narrative_direction=result.get("narrative_direction", ""),
    )


def generate_agent_config_by_rule(entity: EntityNode) -> Dict[str, Any]:
    """Generate single Agent configuration based on rules (Chinese daily schedule)."""
    entity_type = (entity.get_entity_type() or "Unknown").lower()

    if entity_type in ["university", "governmentagency", "ngo"]:
        return {
            "activity_level": 0.2, "posts_per_hour": 0.1, "comments_per_hour": 0.05,
            "active_hours": list(range(9, 18)),
            "response_delay_min": 60, "response_delay_max": 240,
            "sentiment_bias": 0.0, "stance": "neutral", "influence_weight": 3.0,
        }
    elif entity_type in ["mediaoutlet"]:
        return {
            "activity_level": 0.5, "posts_per_hour": 0.8, "comments_per_hour": 0.3,
            "active_hours": list(range(7, 24)),
            "response_delay_min": 5, "response_delay_max": 30,
            "sentiment_bias": 0.0, "stance": "observer", "influence_weight": 2.5,
        }
    elif entity_type in ["professor", "expert", "official"]:
        return {
            "activity_level": 0.4, "posts_per_hour": 0.3, "comments_per_hour": 0.5,
            "active_hours": list(range(8, 22)),
            "response_delay_min": 15, "response_delay_max": 90,
            "sentiment_bias": 0.0, "stance": "neutral", "influence_weight": 2.0,
        }
    elif entity_type in ["student"]:
        return {
            "activity_level": 0.8, "posts_per_hour": 0.6, "comments_per_hour": 1.5,
            "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
            "response_delay_min": 1, "response_delay_max": 15,
            "sentiment_bias": 0.0, "stance": "neutral", "influence_weight": 0.8,
        }
    elif entity_type in ["alumni"]:
        return {
            "activity_level": 0.6, "posts_per_hour": 0.4, "comments_per_hour": 0.8,
            "active_hours": [12, 13, 19, 20, 21, 22, 23],
            "response_delay_min": 5, "response_delay_max": 30,
            "sentiment_bias": 0.0, "stance": "neutral", "influence_weight": 1.0,
        }
    else:
        return {
            "activity_level": 0.7, "posts_per_hour": 0.5, "comments_per_hour": 1.2,
            "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
            "response_delay_min": 2, "response_delay_max": 20,
            "sentiment_bias": 0.0, "stance": "neutral", "influence_weight": 1.0,
        }


def assign_initial_post_agents(
    event_config: EventConfig,
    agent_configs: List[AgentActivityConfig],
) -> EventConfig:
    """Assign suitable publisher Agents to initial posts based on poster_type."""
    if not event_config.initial_posts:
        return event_config

    agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
    for agent in agent_configs:
        etype = agent.entity_type.lower()
        if etype not in agents_by_type:
            agents_by_type[etype] = []
        agents_by_type[etype].append(agent)

    type_aliases = {
        "official": ["official", "university", "governmentagency", "government"],
        "university": ["university", "official"],
        "mediaoutlet": ["mediaoutlet", "media"],
        "student": ["student", "person"],
        "professor": ["professor", "expert", "teacher"],
        "alumni": ["alumni", "person"],
        "organization": ["organization", "ngo", "company", "group"],
        "person": ["person", "student", "alumni"],
    }

    used_indices: Dict[str, int] = {}
    updated_posts = []

    for post in event_config.initial_posts:
        poster_type = post.get("poster_type", "").lower()
        content = post.get("content", "")
        matched_agent_id = None

        if poster_type in agents_by_type:
            agents = agents_by_type[poster_type]
            idx = used_indices.get(poster_type, 0) % len(agents)
            matched_agent_id = agents[idx].agent_id
            used_indices[poster_type] = idx + 1
        else:
            for alias_key, aliases in type_aliases.items():
                if poster_type in aliases or alias_key == poster_type:
                    for alias in aliases:
                        if alias in agents_by_type:
                            agents = agents_by_type[alias]
                            idx = used_indices.get(alias, 0) % len(agents)
                            matched_agent_id = agents[idx].agent_id
                            used_indices[alias] = idx + 1
                            break
                if matched_agent_id is not None:
                    break

        if matched_agent_id is None:
            logger.warning(
                f"No matching Agent for type '{poster_type}', using highest influence")
            if agent_configs:
                sorted_agents = sorted(
                    agent_configs, key=lambda a: a.influence_weight, reverse=True)
                matched_agent_id = sorted_agents[0].agent_id
            else:
                matched_agent_id = 0

        updated_posts.append({
            "content": content,
            "poster_type": post.get("poster_type", "Unknown"),
            "poster_agent_id": matched_agent_id,
        })
        logger.info(
            f"Initial post assignment: poster_type='{poster_type}' -> agent_id={matched_agent_id}")

    event_config.initial_posts = updated_posts
    return event_config
