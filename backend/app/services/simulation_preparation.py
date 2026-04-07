"""
Simulation preparation helpers — profile generation and config generation phases.
"""

import os
import json
from typing import List, Optional

from ..utils.logger import get_logger
from .entity_reader import EntityReader, FilteredEntities
from .oasis_profile_generator import OasisProfileGenerator
from .simulation_config_generator import SimulationConfigGenerator
from .simulation_types import SimulationState, SimulationStatus

logger = get_logger('miroshark.simulation')


def generate_profiles(state, sim_dir, filtered, storage,
                      simulation_requirement, use_llm_for_profiles,
                      progress_callback, parallel_profile_count):
    """Phase 2: Generate Agent Profiles."""
    total_entities = len(filtered.entities)

    if progress_callback:
        progress_callback("generating_profiles", 0, "Starting generation...",
                          current=0, total=total_entities)

    generator = OasisProfileGenerator(
        storage=storage, graph_id=state.graph_id,
        simulation_requirement=simulation_requirement,
    )

    def profile_progress(current, total, msg):
        if progress_callback:
            progress_callback("generating_profiles", int(current / total * 100),
                              msg, current=current, total=total, item_name=msg)

    realtime_output_path = None
    realtime_platform = "reddit"
    if state.enable_reddit:
        realtime_output_path = os.path.join(sim_dir, "reddit_profiles.json")
    elif state.enable_twitter:
        realtime_output_path = os.path.join(sim_dir, "twitter_profiles.csv")
        realtime_platform = "twitter"

    profiles = generator.generate_profiles_from_entities(
        entities=filtered.entities, use_llm=use_llm_for_profiles,
        progress_callback=profile_progress, graph_id=state.graph_id,
        parallel_count=parallel_profile_count,
        realtime_output_path=realtime_output_path,
        output_platform=realtime_platform,
    )

    state.profiles_count = len(profiles)

    if progress_callback:
        progress_callback("generating_profiles", 95, "Saving Profile files...",
                          current=total_entities, total=total_entities)

    if state.enable_reddit:
        generator.save_profiles(profiles=profiles,
                                file_path=os.path.join(sim_dir, "reddit_profiles.json"),
                                platform="reddit")
    if state.enable_twitter:
        generator.save_profiles(profiles=profiles,
                                file_path=os.path.join(sim_dir, "twitter_profiles.csv"),
                                platform="twitter")
    if state.enable_polymarket:
        generator.save_profiles(profiles=profiles,
                                file_path=os.path.join(sim_dir, "polymarket_profiles.json"),
                                platform="polymarket")

    if progress_callback:
        progress_callback("generating_profiles", 100,
                          f"Done, {len(profiles)} Profiles in total",
                          current=len(profiles), total=len(profiles))


def generate_config(state, sim_dir, filtered, simulation_requirement,
                    document_text, progress_callback, target_agents):
    """Phase 3: LLM-powered simulation config generation."""
    if progress_callback:
        progress_callback("generating_config", 0,
                          "Analyzing simulation requirements...",
                          current=0, total=3)

    config_generator = SimulationConfigGenerator()

    if progress_callback:
        progress_callback("generating_config", 30,
                          "Calling LLM to generate config...",
                          current=1, total=3)

    sim_params = config_generator.generate_config(
        simulation_id=state.simulation_id,
        project_id=state.project_id,
        graph_id=state.graph_id,
        simulation_requirement=simulation_requirement,
        document_text=document_text,
        entities=filtered.entities,
        enable_twitter=state.enable_twitter,
        enable_reddit=state.enable_reddit,
        target_agents=target_agents,
    )

    if progress_callback:
        progress_callback("generating_config", 70, "Saving config files...",
                          current=2, total=3)

    config_path = os.path.join(sim_dir, "simulation_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(sim_params.to_json())

    state.config_generated = True
    state.config_reasoning = sim_params.generation_reasoning

    if progress_callback:
        progress_callback("generating_config", 100, "Config generation complete",
                          current=3, total=3)
