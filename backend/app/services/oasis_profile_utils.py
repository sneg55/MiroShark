"""
Pure utility helpers for OASIS Agent Profile Generator

Contains: username generation, risk tolerance inference,
entity interleaving, and console profile printing.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

from .oasis_profile_types import OasisAgentProfile


def generate_username(name: str) -> str:
    """Generate a unique-ish username from an entity name."""
    username = name.lower().replace(" ", "_")
    username = ''.join(c for c in username if c.isalnum() or c == '_')
    return f"{username}_{random.randint(100, 999)}"


def infer_risk_tolerance(
    entity_type: str,
    mbti: Optional[str],
    profession: Optional[str],
    entity_name: str = "",
) -> str:
    """Infer risk tolerance from entity characteristics for Polymarket profiles."""
    name_lower = (entity_name or "").lower()
    if any(w in name_lower for w in ("hedge fund", "venture", "trading", "capital",
                                      "defi", "prediction market", "polymarket", "augur")):
        return "high"
    if any(w in name_lower for w in ("stablecoin", "usdc", "usdt", "reserve", "treasury")):
        return "low"
    if entity_type and entity_type.lower() in ("governmentagency", "ngo", "institution", "university"):
        return "low"
    if entity_type and entity_type.lower() in ("company", "mediaoutlet", "organization"):
        if entity_name:
            return ["low", "moderate", "high"][hash(entity_name) % 3]
        return "moderate"
    if mbti and len(mbti) == 4:
        if mbti[1] == 'N' and mbti[3] == 'P':
            return "high"
        if mbti[1] == 'S' and mbti[3] == 'J':
            return "low"
    if profession:
        p = profession.lower()
        if any(w in p for w in ("trader", "investor", "entrepreneur", "activist")):
            return "high"
        if any(w in p for w in ("accountant", "official", "administrator", "lawyer")):
            return "low"
    return random.choice(["high", "moderate", "moderate", "low"])


_INDIVIDUAL_SET = {
    "person", "publicfigure", "expert", "faculty", "student", "alumni",
    "journalist", "activist", "politician", "official",
    "cryptofounder", "electionforecaster", "predictionmarketuser",
    "cryptoinfluencer", "regulatoryofficial",
}


def interleave_by_type(entities: list) -> list:
    """Reorder entities to interleave types for diverse early results.

    Instead of [Org, Org, Org, Person, Person], produces
    [Person, Org, Person, Org, Org] — round-robin by type.
    """
    buckets: Dict[str, list] = defaultdict(list)
    for e in entities:
        etype = (e.get_entity_type() or "Entity").lower()
        buckets[etype].append(e)

    individual_keys, group_keys = [], []
    for key in buckets:
        (individual_keys if key in _INDIVIDUAL_SET else group_keys).append(key)

    ordered_keys = individual_keys + group_keys
    for key in buckets:
        if key not in ordered_keys:
            ordered_keys.append(key)

    result = []
    iterators = {k: iter(buckets[k]) for k in ordered_keys}
    while iterators:
        exhausted = []
        for key in ordered_keys:
            if key not in iterators:
                continue
            try:
                result.append(next(iterators[key]))
            except StopIteration:
                exhausted.append(key)
        for key in exhausted:
            del iterators[key]
    return result


def print_generated_profile(entity_name: str, entity_type: str, profile: OasisAgentProfile):
    """Print a generated profile to stdout in a readable format."""
    separator = "-" * 70
    topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'None'
    lines = [
        f"\n{separator}",
        f"[Generated] {entity_name} ({entity_type})",
        separator,
        f"Username: {profile.user_name}",
        "",
        "[Bio]", profile.bio,
        "",
        "[Detailed Persona]", profile.persona,
        "",
        "[Basic Attributes]",
        f"Age: {profile.age} | Gender: {profile.gender} | MBTI: {profile.mbti}",
        f"Profession: {profile.profession} | Country: {profile.country}",
        f"Interested Topics: {topics_str}",
        separator,
    ]
    print("\n".join(lines))
