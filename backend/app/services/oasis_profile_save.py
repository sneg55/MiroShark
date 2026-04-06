"""
Save/export utilities for OASIS Agent Profile Generator

Handles writing profiles to Reddit JSON, Twitter CSV, and Polymarket JSON formats.
"""

import csv
import json
from typing import List, Optional

from .oasis_profile_types import OasisAgentProfile
from ..utils.logger import get_logger

logger = get_logger('miroshark.oasis_profile')


def normalize_gender(gender: Optional[str]) -> str:
    """Normalize gender field to OASIS required English format (male / female / other)"""
    if not gender:
        return "other"
    gender_map = {"male": "male", "female": "female", "other": "other"}
    return gender_map.get(gender.lower().strip(), "other")


def save_profiles(
    profiles: List[OasisAgentProfile],
    file_path: str,
    platform: str = "reddit"
):
    """Save profiles to file in the correct format for the given platform.

    Platform format requirements:
    - reddit:      JSON
    - twitter:     CSV
    - polymarket:  JSON
    """
    if platform == "twitter":
        save_twitter_csv(profiles, file_path)
    elif platform == "polymarket":
        save_polymarket_json(profiles, file_path)
    else:
        save_reddit_json(profiles, file_path)


def save_twitter_csv(profiles: List[OasisAgentProfile], file_path: str):
    """Save Twitter profiles as CSV (OASIS official format).

    Required columns: user_id, name, username, user_char, description.
    - user_char: Full persona fed into LLM system prompt
    - description: Short public bio shown on profile
    """
    if not file_path.endswith('.csv'):
        file_path = file_path.replace('.json', '.csv')

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'name', 'username', 'user_char', 'description'])

        for idx, profile in enumerate(profiles):
            user_char = profile.bio
            if profile.persona and profile.persona != profile.bio:
                user_char = f"{profile.bio} {profile.persona}"
            user_char = user_char.replace('\n', ' ').replace('\r', ' ')
            description = profile.bio.replace('\n', ' ').replace('\r', ' ')

            writer.writerow([
                idx,
                profile.name,
                profile.user_name,
                user_char,
                description,
            ])

    logger.info(f"Saved {len(profiles)} Twitter Profiles to {file_path} (OASIS CSV format)")


def save_reddit_json(profiles: List[OasisAgentProfile], file_path: str):
    """Save Reddit profiles as JSON.

    Includes user_id field, which is critical for OASIS agent_graph.get_agent() matching.
    """
    data = []
    for idx, profile in enumerate(profiles):
        item = {
            "user_id": profile.user_id if profile.user_id is not None else idx,
            "username": profile.user_name,
            "name": profile.name,
            "bio": profile.bio[:500] if profile.bio else f"{profile.name}",
            "persona": profile.persona or f"{profile.name} is a participant in social discussions.",
            "karma": profile.karma if profile.karma else 1000,
            "created_at": profile.created_at,
            "age": profile.age if profile.age else 30,
            "gender": normalize_gender(profile.gender),
            "mbti": profile.mbti if profile.mbti else "ISTJ",
            "country": profile.country if profile.country else "China",
        }
        if profile.profession:
            item["profession"] = profile.profession
        if profile.interested_topics:
            item["interested_topics"] = profile.interested_topics
        data.append(item)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Saved {len(profiles)} Reddit Profiles to {file_path} (JSON format, includes user_id field)"
    )


def save_polymarket_json(profiles: List[OasisAgentProfile], file_path: str):
    """Save Polymarket profiles as JSON.

    Each entry matches Wonderwall's UserInfo structure:
    UserInfo(name=..., description=..., profile={"other_info": {"user_profile": ..., "risk_tolerance": ...}})
    """
    data = []
    for idx, profile in enumerate(profiles):
        pm = profile.to_polymarket_format()
        pm["user_id"] = profile.user_id if profile.user_id is not None else idx
        data.append(pm)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(profiles)} Polymarket profiles to {file_path}")
