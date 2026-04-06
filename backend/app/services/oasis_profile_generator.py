"""
OASIS Agent Profile Generator
Convert entities from the knowledge graph to OASIS simulation platform's required Agent Profile format
"""

import concurrent.futures
from threading import Lock
from typing import Dict, Any, List, Optional

from ..config import Config
from ..utils.llm_client import create_llm_client
from ..utils.logger import get_logger
from .entity_reader import EntityNode
from .web_enrichment import WebEnricher
from ..storage import GraphStorage

from .oasis_profile_constants import (
    MBTI_TYPES, COUNTRIES,
    INDIVIDUAL_ENTITY_TYPES, INDIVIDUAL_TYPE_KEYWORDS, GROUP_ENTITY_TYPES,
)
from .oasis_profile_types import OasisAgentProfile, social_metrics_for_entity_type
from .oasis_profile_llm import generate_profile_with_llm, generate_profile_rule_based
from .oasis_profile_context import build_entity_context
from .oasis_profile_save import save_profiles as _save_profiles
from .oasis_profile_utils import (
    generate_username, infer_risk_tolerance, interleave_by_type, print_generated_profile,
)

logger = get_logger('miroshark.oasis_profile')

# Backward-compatible module-level alias
_social_metrics_for_entity_type = social_metrics_for_entity_type


class OasisProfileGenerator:
    """
    Convert knowledge graph entities to OASIS Agent Profiles.

    Supports individual and institutional entity types, LLM + rule-based generation,
    parallel batch generation, and multi-platform export (Reddit, Twitter, Polymarket).
    """

    MBTI_TYPES = MBTI_TYPES
    COUNTRIES = COUNTRIES
    INDIVIDUAL_ENTITY_TYPES = INDIVIDUAL_ENTITY_TYPES
    INDIVIDUAL_TYPE_KEYWORDS = INDIVIDUAL_TYPE_KEYWORDS
    GROUP_ENTITY_TYPES = GROUP_ENTITY_TYPES

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        storage: Optional[GraphStorage] = None,
        graph_id: Optional[str] = None,
        simulation_requirement: Optional[str] = None,
    ):
        self.model_name = model_name or Config.LLM_MODEL_NAME
        self.llm = create_llm_client(api_key=api_key, base_url=base_url, model=model_name)
        self.storage = storage
        self.graph_id = graph_id
        self.web_enricher = WebEnricher()
        self.simulation_requirement = simulation_requirement or ""

    # ------------------------------------------------------------------
    # Entity type helpers
    # ------------------------------------------------------------------

    def _is_individual_entity(self, entity_type: str) -> bool:
        et = entity_type.lower().replace(" ", "")
        if et in self.INDIVIDUAL_ENTITY_TYPES:
            return True
        for keyword in self.INDIVIDUAL_TYPE_KEYWORDS:
            if keyword in et:
                return True
        return not self._is_group_entity(entity_type)

    def _is_group_entity(self, entity_type: str) -> bool:
        return entity_type.lower().replace(" ", "") in self.GROUP_ENTITY_TYPES

    def set_graph_id(self, graph_id: str):
        self.graph_id = graph_id

    # ------------------------------------------------------------------
    # Single-profile generation
    # ------------------------------------------------------------------

    def generate_profile_from_entity(
        self,
        entity: EntityNode,
        user_id: int,
        use_llm: bool = True,
    ) -> OasisAgentProfile:
        """Generate a single OASIS Agent Profile from a knowledge graph entity."""
        entity_type = entity.get_entity_type() or "Entity"
        name = entity.name
        user_name = generate_username(name)
        context = build_entity_context(
            entity, self.storage, self.graph_id, self.web_enricher, self.simulation_requirement
        )

        if use_llm:
            profile_data = generate_profile_with_llm(
                llm=self.llm,
                is_individual=self._is_individual_entity(entity_type),
                entity_name=name, entity_type=entity_type,
                entity_summary=entity.summary, entity_attributes=entity.attributes,
                context=context,
                fallback_fn=generate_profile_rule_based,
            )
        else:
            profile_data = generate_profile_rule_based(
                entity_name=name, entity_type=entity_type,
                entity_summary=entity.summary, entity_attributes=entity.attributes,
            )

        risk_tolerance = profile_data.get("risk_tolerance") or infer_risk_tolerance(
            entity_type, profile_data.get("mbti"), profile_data.get("profession"), entity_name=name,
        )
        social_defaults = social_metrics_for_entity_type(entity_type, entity)

        return OasisAgentProfile(
            user_id=user_id, user_name=user_name, name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            risk_tolerance=risk_tolerance,
            karma=profile_data.get("karma", social_defaults["karma"]),
            friend_count=profile_data.get("friend_count", social_defaults["friend_count"]),
            follower_count=profile_data.get("follower_count", social_defaults["follower_count"]),
            statuses_count=profile_data.get("statuses_count", social_defaults["statuses_count"]),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 15,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit",
    ) -> List[OasisAgentProfile]:
        """Batch-generate Agent Profiles in parallel, preserving input order."""
        if graph_id:
            self.graph_id = graph_id

        entities = interleave_by_type(entities)
        total = len(entities)
        profiles: List[Optional[OasisAgentProfile]] = [None] * total
        completed_count = [0]
        lock = Lock()

        def save_realtime():
            if not realtime_output_path:
                return
            with lock:
                existing = [p for p in profiles if p is not None]
                if existing:
                    try:
                        _save_profiles(existing, realtime_output_path, output_platform)
                    except Exception as e:
                        logger.warning(f"Failed to save profiles in real time: {e}")

        def generate_one(idx: int, entity: EntityNode):
            entity_type = entity.get_entity_type() or "Entity"
            try:
                profile = self.generate_profile_from_entity(entity=entity, user_id=idx, use_llm=use_llm)
                print_generated_profile(entity.name, entity_type, profile)
                return idx, profile, None
            except Exception as e:
                logger.error(f"Failed to generate persona for {entity.name}: {e}")
                fallback = OasisAgentProfile(
                    user_id=idx, user_name=generate_username(entity.name),
                    name=entity.name, bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or "A participant in social discussions.",
                    source_entity_uuid=entity.uuid, source_entity_type=entity_type,
                )
                return idx, fallback, str(e)

        logger.info(f"Starting parallel generation of {total} Agent personas (parallelism: {parallel_count})...")
        print(f"\n{'='*60}\nStarting Agent persona generation - {total} entities, parallelism: {parallel_count}\n{'='*60}\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            futures = {executor.submit(generate_one, i, e): (i, e) for i, e in enumerate(entities)}
            for future in concurrent.futures.as_completed(futures):
                idx, entity = futures[future]
                entity_type = entity.get_entity_type() or "Entity"
                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile
                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]
                    save_realtime()
                    if progress_callback:
                        progress_callback(current, total, f"Completed {current}/{total}: {entity.name} ({entity_type})")
                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} using fallback: {error}")
                    else:
                        logger.info(f"[{current}/{total}] Generated: {entity.name} ({entity_type})")
                except Exception as e:
                    logger.error(f"Exception processing {entity.name}: {e}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx, user_name=generate_username(entity.name),
                        name=entity.name, bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid, source_entity_type=entity_type,
                    )
                    save_realtime()

        print(f"\n{'='*60}\nPersona generation complete! Generated {len([p for p in profiles if p])} Agents\n{'='*60}\n")
        return profiles

    # ------------------------------------------------------------------
    # Save API (delegates to oasis_profile_save)
    # ------------------------------------------------------------------

    def save_profiles(self, profiles: List[OasisAgentProfile], file_path: str, platform: str = "reddit"):
        """Save profiles to file (reddit JSON / twitter CSV / polymarket JSON)."""
        _save_profiles(profiles, file_path, platform)

    def save_profiles_to_json(self, profiles: List[OasisAgentProfile], file_path: str, platform: str = "reddit"):
        """[Deprecated] Use save_profiles() instead."""
        logger.warning("save_profiles_to_json is deprecated, please use save_profiles method")
        _save_profiles(profiles, file_path, platform)
