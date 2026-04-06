"""
OASIS Agent Profile dataclass and social metrics helper
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class OasisAgentProfile:
    """OASIS Agent Profile data structure"""
    # Common fields
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # Optional fields - Reddit style
    karma: int = 1000

    # Optional fields - Twitter style
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500

    # Polymarket-specific fields
    risk_tolerance: str = "moderate"  # "high", "moderate", or "low"

    # Additional persona information
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)

    # Source entity information
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_reddit_format(self) -> Dict[str, Any]:
        """Convert to Reddit platform format"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS library requires field name as username (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }

        # Add additional persona information (if available)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_twitter_format(self) -> Dict[str, Any]:
        """Convert to Twitter platform format"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS library requires field name as username (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }

        # Add additional persona information
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_polymarket_format(self) -> Dict[str, Any]:
        """Convert to Polymarket prediction market format.

        Returns a dict compatible with Wonderwall's UserInfo(profile={"other_info": ...})
        structure, which PolymarketPromptBuilder reads to build trader personas.
        """
        # Build the user_profile text from persona + profession context
        user_profile = self.persona or f"{self.name} participates in prediction markets."
        if self.profession:
            user_profile = f"{self.profession}. {user_profile}"

        return {
            "user_id": self.user_id,
            "name": self.user_name,
            "description": self.bio or f"Prediction market trader: {self.name}",
            "risk_tolerance": self.risk_tolerance,
            "user_profile": user_profile,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to complete dictionary format"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


def social_metrics_for_entity_type(entity_type: str, entity=None) -> Dict[str, int]:
    """Derive social media metrics from entity type and graph structure.

    Replaces the previous random.randint() fallbacks with values grounded in
    the entity's structural role. An institutional media outlet should have
    high follower counts; a student should have modest ones.

    If the entity has related_edges (from the knowledge graph), the degree
    (number of connections) is used as a scaling factor — more connected
    entities get proportionally higher metrics.
    """
    # Use graph degree as a scaling factor (1.0 = baseline, up to ~3.0 for hubs)
    degree = 1
    if entity and hasattr(entity, 'related_edges') and entity.related_edges:
        degree = len(entity.related_edges)
    elif entity and hasattr(entity, 'attributes') and isinstance(entity.attributes, dict):
        degree = entity.attributes.get('degree', 1)
    degree_factor = min(3.0, 1.0 + (degree - 1) * 0.15)

    et = entity_type.lower()

    # Base metrics by entity archetype
    if et in ("mediaoutlet", "socialmediaplatform"):
        base = {"karma": 15000, "friend_count": 200, "follower_count": 50000, "statuses_count": 10000}
    elif et in ("university", "governmentagency", "organization", "ngo"):
        base = {"karma": 8000, "friend_count": 150, "follower_count": 20000, "statuses_count": 5000}
    elif et in ("publicfigure", "expert", "faculty"):
        base = {"karma": 5000, "friend_count": 300, "follower_count": 8000, "statuses_count": 3000}
    elif et in ("student", "alumni"):
        base = {"karma": 800, "friend_count": 200, "follower_count": 300, "statuses_count": 500}
    elif et in ("politician", "official", "regulator"):
        base = {"karma": 6000, "friend_count": 250, "follower_count": 15000, "statuses_count": 4000}
    else:
        base = {"karma": 1500, "friend_count": 120, "follower_count": 500, "statuses_count": 800}

    # Scale by graph degree and add small deterministic jitter from entity name hash
    name_hash = 0
    if entity and hasattr(entity, 'name') and entity.name:
        name_hash = hash(entity.name) % 100  # 0-99 deterministic per entity
    jitter = 0.85 + (name_hash / 100) * 0.30  # 0.85 to 1.15

    return {
        k: max(1, int(v * degree_factor * jitter))
        for k, v in base.items()
    }


# Backward-compatible alias (old name had leading underscore as a module-level function)
_social_metrics_for_entity_type = social_metrics_for_entity_type
