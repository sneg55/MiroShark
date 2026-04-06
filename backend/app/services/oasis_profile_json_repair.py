"""
JSON repair utilities for OASIS Agent Profile Generator

Handles truncated or malformed LLM JSON outputs.
"""

import re
from typing import Dict, Any

from ..utils.logger import get_logger

logger = get_logger('miroshark.oasis_profile')


def fix_truncated_json(content: str) -> str:
    """Fix truncated JSON (output truncated by max_tokens limit)"""
    content = content.strip()

    # Count unclosed brackets
    open_braces = content.count('{') - content.count('}')
    open_brackets = content.count('[') - content.count(']')

    # Check for unclosed strings
    if content and content[-1] not in '",}]':
        content += '"'

    # Close brackets
    content += ']' * open_brackets
    content += '}' * open_braces

    return content


def try_fix_json(
    content: str,
    entity_name: str,
    entity_type: str,
    entity_summary: str = ""
) -> Dict[str, Any]:
    """Try to fix corrupted JSON. Returns a dict; sets '_fixed': True on success."""

    # 1. First try to fix truncation
    content = fix_truncated_json(content)

    # 2. Try to extract JSON portion
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        json_str = json_match.group()

        # 3. Handle newline issues in strings
        def fix_string_newlines(match):
            s = match.group(0)
            s = s.replace('\n', ' ').replace('\r', ' ')
            s = re.sub(r'\s+', ' ', s)
            return s

        import json
        json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)

        # 4. Try to parse
        try:
            result = json.loads(json_str)
            result["_fixed"] = True
            return result
        except json.JSONDecodeError:
            # 5. More aggressive fix: strip control characters
            try:
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except Exception:
                pass

    # 6. Try to extract partial information from content
    bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
    persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # May be truncated

    bio = bio_match.group(1) if bio_match else (
        entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
    )
    persona = persona_match.group(1) if persona_match else (
        entity_summary or f"{entity_name} is a {entity_type}."
    )

    if bio_match or persona_match:
        logger.info("Extracted partial information from corrupted JSON")
        return {"bio": bio, "persona": persona, "_fixed": True}

    # 7. Complete failure — return basic structure
    logger.warning("JSON fix failed, returning basic structure")
    return {
        "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
        "persona": entity_summary or f"{entity_name} is a {entity_type}.",
    }
