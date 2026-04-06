"""
LLM-based and rule-based persona generation for OASIS Agent Profile Generator
"""

import json
import random
import time
from typing import Dict, Any, Optional

from ..utils.logger import get_logger
from .oasis_profile_prompts import get_system_prompt, build_individual_persona_prompt, build_group_persona_prompt
from .oasis_profile_json_repair import try_fix_json
from .oasis_profile_constants import MBTI_TYPES, COUNTRIES

logger = get_logger('miroshark.oasis_profile')


def generate_profile_with_llm(
    llm,
    is_individual: bool,
    entity_name: str,
    entity_type: str,
    entity_summary: str,
    entity_attributes: Dict[str, Any],
    context: str,
    fallback_fn,
) -> Dict[str, Any]:
    """Use LLM to generate a detailed persona dict; calls fallback_fn on repeated failure."""
    if is_individual:
        prompt = build_individual_persona_prompt(
            entity_name, entity_type, entity_summary, entity_attributes, context
        )
    else:
        prompt = build_group_persona_prompt(
            entity_name, entity_type, entity_summary, entity_attributes, context
        )

    max_attempts = 3
    last_error = None
    for attempt in range(max_attempts):
        try:
            messages = [
                {"role": "system", "content": get_system_prompt(is_individual)},
                {"role": "user", "content": prompt},
            ]
            content = llm.chat(
                messages=messages,
                temperature=0.7 - (attempt * 0.1),
                response_format={"type": "json_object"},
            )
            try:
                result = json.loads(content)
                if "bio" not in result or not result["bio"]:
                    result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                if "persona" not in result or not result["persona"]:
                    result["persona"] = entity_summary or f"{entity_name} is a {entity_type}."
                return result
            except json.JSONDecodeError as je:
                logger.warning(f"JSON parsing failed (attempt {attempt+1}): {str(je)[:80]}")
                result = try_fix_json(content, entity_name, entity_type, entity_summary)
                if result.get("_fixed"):
                    del result["_fixed"]
                    return result
                last_error = je
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
            last_error = e
            time.sleep(1 * (attempt + 1))

    logger.warning(f"LLM persona generation failed ({max_attempts} attempts): {last_error}, using rule-based")
    return fallback_fn(entity_name, entity_type, entity_summary, entity_attributes)


def generate_profile_rule_based(
    entity_name: str,
    entity_type: str,
    entity_summary: str,
    entity_attributes: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a basic persona using deterministic rules (LLM fallback)."""
    entity_type_lower = entity_type.lower()

    if entity_type_lower in ["student", "alumni"]:
        return {
            "bio": f"{entity_type} with interests in academics and social issues.",
            "persona": f"{entity_name} is a {entity_type.lower()} actively engaged in academic and social discussions.",
            "age": random.randint(18, 30), "gender": random.choice(["male", "female"]),
            "mbti": random.choice(MBTI_TYPES), "country": random.choice(COUNTRIES),
            "profession": "Student", "interested_topics": ["Education", "Social Issues", "Technology"],
        }
    if entity_type_lower in ["publicfigure", "expert", "faculty"]:
        return {
            "bio": "Expert and thought leader in their field.",
            "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights on important matters.",
            "age": random.randint(35, 60), "gender": random.choice(["male", "female"]),
            "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
            "country": random.choice(COUNTRIES),
            "profession": entity_attributes.get("occupation", "Expert"),
            "interested_topics": ["Politics", "Economics", "Culture & Society"],
        }
    if entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
        return {
            "bio": f"Official account for {entity_name}. News and updates.",
            "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse.",
            "age": 30, "gender": "other", "mbti": "ISTJ", "country": "China",
            "profession": "Media", "interested_topics": ["General News", "Current Events", "Public Affairs"],
        }
    if entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
        return {
            "bio": f"Official account of {entity_name}.",
            "persona": f"{entity_name} communicates official positions and engages with stakeholders.",
            "age": 30, "gender": "other", "mbti": "ISTJ", "country": "China",
            "profession": entity_type,
            "interested_topics": ["Public Policy", "Community", "Official Announcements"],
        }
    return {
        "bio": entity_summary[:500] if entity_summary else f"{entity_type}: {entity_name}",
        "persona": entity_summary or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
        "age": random.randint(25, 50), "gender": random.choice(["male", "female"]),
        "mbti": random.choice(MBTI_TYPES), "country": random.choice(COUNTRIES),
        "profession": entity_type, "interested_topics": ["General", "Social Issues"],
    }
