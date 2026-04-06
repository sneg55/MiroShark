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
    """Generate basic persona using rules, with structured soul.md sections"""

    entity_type_lower = entity_type.lower()
    summary_text = entity_summary or f"{entity_name} is a {entity_type.lower()}."

    if entity_type_lower in ["student", "alumni"]:
        persona = (
            f"## SOUL (Identity & Worldview)\n"
            f"{entity_name} is a {entity_type.lower()} engaged in academic and social discussions. "
            f"They tend to see issues through the lens of fairness and personal impact. "
            f"They hold strong opinions but are open to changing their mind when presented with data.\n\n"
            f"## STYLE (Voice & Writing Patterns)\n"
            f"Writes informally with occasional internet slang. Uses rhetorical questions. "
            f"On Twitter, posts quick reactions. On Reddit, writes 2-3 paragraph responses "
            f"drawing on personal experience.\n\n"
            f"## BEHAVIOR (Operating Modes)\n"
            f"Engagement-heavy — likes and comments frequently. Responds to controversy "
            f"and personal stories. Shares articles with brief commentary. "
            f"More active on Reddit than Twitter."
        )
        return {
            "bio": f"{entity_type} with interests in academics and social issues.",
            "persona": persona,
            "age": random.randint(18, 30),
            "gender": random.choice(["male", "female"]),
            "mbti": random.choice(MBTI_TYPES),
            "country": random.choice(COUNTRIES),
            "profession": "Student",
            "interested_topics": ["Education", "Social Issues", "Technology"],
        }

    elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
        persona = (
            f"## SOUL (Identity & Worldview)\n"
            f"{entity_name} is a recognized {entity_type.lower()} who has built authority "
            f"through years of domain expertise. They argue from evidence and professional "
            f"experience. {summary_text}\n\n"
            f"## STYLE (Voice & Writing Patterns)\n"
            f"Writes in a measured, authoritative tone. Uses data and citations. "
            f"On Twitter, posts concise takes with links. On Reddit, writes detailed "
            f"analytical responses. Avoids slang but isn't stuffy.\n\n"
            f"## BEHAVIOR (Operating Modes)\n"
            f"Moderate engagement — posts original takes and responds to substantive "
            f"challenges. Ignores trolls. Will correct misinformation in their domain. "
            f"Shares and comments on others' work in their field."
        )
        return {
            "bio": f"Expert and thought leader in their field.",
            "persona": persona,
            "age": random.randint(35, 60),
            "gender": random.choice(["male", "female"]),
            "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
            "country": random.choice(COUNTRIES),
            "profession": entity_attributes.get("occupation", "Expert"),
            "interested_topics": ["Politics", "Economics", "Culture & Society"],
        }

    elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
        persona = (
            f"## SOUL (Institutional Identity & Position)\n"
            f"{entity_name} is a media entity that reports news and facilitates public "
            f"discourse. Projects an image of objectivity and authority. "
            f"Will not take explicit partisan positions but has editorial leanings.\n\n"
            f"## STYLE (Voice & Tone)\n"
            f"Formal but accessible. Uses third person. Headlines are punchy, body text "
            f"is measured. On Twitter, posts headlines with links. On Reddit, posts "
            f"detailed article summaries.\n\n"
            f"## BEHAVIOR (Operating Modes)\n"
            f"High-frequency broadcaster. Posts breaking news, analysis, and opinion pieces. "
            f"Rarely engages in debates but will post corrections. "
            f"Responds to major controversies with official statements."
        )
        return {
            "bio": f"Official account for {entity_name}. News and updates.",
            "persona": persona,
            "age": 30,
            "gender": "other",
            "mbti": "ISTJ",
            "country": entity_attributes.get("country", "United States"),
            "profession": "Media",
            "interested_topics": ["General News", "Current Events", "Public Affairs"],
        }

    elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
        persona = (
            f"## SOUL (Institutional Identity & Position)\n"
            f"{entity_name} communicates official positions and engages with stakeholders. "
            f"Projects competence and public service. {summary_text}\n\n"
            f"## STYLE (Voice & Tone)\n"
            f"Professional and measured. Uses 'we' and formal language. "
            f"On Twitter, posts announcements and policy updates. On Reddit, "
            f"participates in relevant discussions with official perspective.\n\n"
            f"## BEHAVIOR (Operating Modes)\n"
            f"Moderate posting frequency. Broadcasts announcements and responds to "
            f"direct questions. Deflects controversy with careful language. "
            f"Engages more on Reddit than Twitter."
        )
        return {
            "bio": f"Official account of {entity_name}.",
            "persona": persona,
            "age": 30,
            "gender": "other",
            "mbti": "ISTJ",
            "country": entity_attributes.get("country", "United States"),
            "profession": entity_type,
            "interested_topics": ["Public Policy", "Community", "Official Announcements"],
        }

    else:
        persona = (
            f"## SOUL (Identity & Worldview)\n"
            f"{summary_text} Has opinions shaped by personal experience and "
            f"engages with topics they care about.\n\n"
            f"## STYLE (Voice & Writing Patterns)\n"
            f"Conversational tone. Writes naturally without heavy jargon. "
            f"On Twitter, posts brief takes. On Reddit, writes moderate-length "
            f"comments with personal perspective.\n\n"
            f"## BEHAVIOR (Operating Modes)\n"
            f"Moderately active — engages when topics are relevant to their interests. "
            f"Likes and comments on content they agree with. "
            f"Will push back on takes they disagree with but avoids flame wars."
        )
        return {
            "bio": entity_summary[:500] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": persona,
            "age": random.randint(25, 50),
            "gender": random.choice(["male", "female"]),
            "mbti": random.choice(MBTI_TYPES),
            "country": random.choice(COUNTRIES),
            "profession": entity_type,
            "interested_topics": ["General", "Social Issues"],
        }
