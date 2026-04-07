"""
Simulation config LLM — core LLM client, retry logic, JSON repair,
and context building utilities.
"""

import json
import re
from typing import Dict, Any, List, Optional

from ..config import Config
from ..utils.llm_client import create_llm_client
from ..utils.logger import get_logger
from .entity_reader import EntityNode

logger = get_logger('miroshark.simulation_config')


class SimulationConfigLLM:
    """LLM interaction layer for simulation config generation."""

    MAX_CONTEXT_LENGTH = 50000
    TIME_CONFIG_CONTEXT_LENGTH = 10000
    EVENT_CONFIG_CONTEXT_LENGTH = 8000
    ENTITY_SUMMARY_LENGTH = 300
    AGENT_SUMMARY_LENGTH = 300
    ENTITIES_PER_TYPE_DISPLAY = 20
    AGENTS_PER_BATCH = 15

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name or Config.LLM_MODEL_NAME
        self.base_url = base_url or Config.LLM_BASE_URL
        self.llm = create_llm_client(
            api_key=api_key, base_url=base_url, model=model_name,
        )

    def call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """LLM call with retry, includes JSON fix logic."""
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                content = self.llm.chat(
                    messages=messages,
                    temperature=0.7 - (attempt * 0.1),
                    response_format={"type": "json_object"},
                )
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON parsing failed (attempt {attempt+1}): {str(e)[:80]}")
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    last_error = e
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLM call failed")

    # ---- JSON repair helpers ----

    def _fix_truncated_json(self, content: str) -> str:
        content = content.strip()
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        if content and content[-1] not in '",}]':
            content += '"'
        content += ']' * open_brackets
        content += '}' * open_braces
        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        content = self._fix_truncated_json(content)
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            try:
                return json.loads(json_str)
            except Exception:
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except Exception:
                    pass
        return None

    # ---- context builders ----

    def build_context(self, simulation_requirement: str, document_text: str,
                      entities: List[EntityNode]) -> str:
        """Build LLM context string, truncated to max length."""
        entity_summary = self._summarize_entities(entities)
        context_parts = [
            f"## Simulation Requirement\n{simulation_requirement}",
            f"\n## Entity Information ({len(entities)} entities)\n{entity_summary}",
        ]
        current_length = sum(len(p) for p in context_parts)
        remaining = self.MAX_CONTEXT_LENGTH - current_length - 500
        if remaining > 0 and document_text:
            doc_text = document_text[:remaining]
            if len(document_text) > remaining:
                doc_text += "\n...(document truncated)"
            context_parts.append(f"\n## Original Document Content\n{doc_text}")
        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        lines = []
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} entities)")
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                preview = ((e.summary[:summary_len] + "...")
                           if len(e.summary) > summary_len else e.summary)
                lines.append(f"- {e.name}: {preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... {len(type_entities) - display_count} more")
        return "\n".join(lines)
