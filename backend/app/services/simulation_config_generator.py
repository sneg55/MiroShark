"""
Simulation configuration intelligent generator — thin orchestrator.

Uses a step-by-step generation strategy:
1. Generate time configuration
2. Generate event configuration
3. Batch generate Agent configurations
4. Generate platform configuration
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable

from ..utils.logger import get_logger
from .entity_reader import EntityNode

# Re-export types so existing imports still work
from .simulation_config_types import (  # noqa: F401
    CHINA_TIMEZONE_CONFIG,
    AgentActivityConfig,
    TimeSimulationConfig,
    EventConfig,
    PlatformConfig,
    SimulationParameters,
)
from .simulation_config_rules import (
    parse_time_config,
    parse_event_config,
    assign_initial_post_agents,
)
from .simulation_config_llm import SimulationConfigLLM
from .simulation_config_prompts import (
    generate_time_config,
    generate_event_config,
    generate_prediction_markets,
    generate_agent_configs_batch,
)

logger = get_logger('miroshark.simulation_config')


class SimulationConfigGenerator:
    """
    Simulation configuration intelligent generator.

    Delegates to SimulationConfigLLM for LLM calls and
    simulation_config_rules for parsing/fallback logic.
    """

    AGENTS_PER_BATCH = 15

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self._llm = SimulationConfigLLM(
            api_key=api_key, base_url=base_url, model_name=model_name,
        )

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    @property
    def base_url(self) -> str:
        return self._llm.base_url

    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        target_agents: int = 5,
    ) -> SimulationParameters:
        """
        Intelligently generate complete simulation configuration.

        Args:
            simulation_id: Simulation ID
            project_id: Project ID
            graph_id: Graph ID
            simulation_requirement: Simulation requirement description
            document_text: Original document content
            entities: Filtered entity list
            enable_twitter: Whether to enable Twitter
            enable_reddit: Whether to enable Reddit
            progress_callback: Progress callback(current_step, total_steps, message)
            target_agents: Target number of agents for the simulation

        Returns:
            SimulationParameters
        """
        logger.info(
            f"Starting config generation: simulation_id={simulation_id}, "
            f"entity_count={len(entities)}")

        num_entities = len(entities)
        num_batches = math.ceil(num_entities / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # Build base context
        context = self._llm.build_context(
            simulation_requirement, document_text, entities)

        reasoning_parts = []

        # Step 1: Time configuration
        report_progress(1, "Generating time configuration...")
        time_result = generate_time_config(self._llm, context, num_entities)
        time_config = parse_time_config(time_result, num_entities)
        reasoning_parts.append(
            f"Time config: {time_result.get('reasoning', 'success')}")

        # Step 2: Event configuration
        report_progress(2, "Generating event configuration and hot topics...")
        event_result = generate_event_config(
            self._llm, context, simulation_requirement, entities)
        event_config = parse_event_config(event_result)
        reasoning_parts.append(
            f"Event config: {event_result.get('reasoning', 'success')}")

        # Step 2b: Prediction markets
        report_progress(3, "Generating prediction markets...")
        markets = generate_prediction_markets(
            self._llm, context, simulation_requirement, event_config)
        event_config.initial_markets = markets
        reasoning_parts.append(f"Prediction markets: {len(markets)} generated")

        # Steps 3-N: Batch generate Agent configurations (parallel)
        all_agent_configs = self._generate_agents_parallel(
            context, entities, simulation_requirement,
            num_batches, report_progress)
        reasoning_parts.append(
            f"Agent configs: Successfully generated {len(all_agent_configs)}")

        # Assign publisher Agents to initial posts
        event_config = assign_initial_post_agents(event_config, all_agent_configs)
        assigned = len([p for p in event_config.initial_posts
                        if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"Initial post assignment: {assigned} posts assigned")

        # Final step: Platform configuration
        report_progress(total_steps, "Generating platform configuration...")
        twitter_config, reddit_config = self._build_platform_configs(
            enable_twitter, enable_reddit)

        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts),
        )

        logger.info(
            f"Config generation complete: {len(params.agent_configs)} Agent configs")
        return params

    def _generate_agents_parallel(self, context, entities,
                                  simulation_requirement,
                                  num_batches, report_progress):
        """Run agent config generation in parallel batches."""
        max_parallel = min(3, num_batches)
        report_progress(
            3, f"Generating Agent configs ({num_batches} batches, "
               f"{max_parallel} parallel)...")

        def _gen_batch(batch_idx):
            start = batch_idx * self.AGENTS_PER_BATCH
            end = min(start + self.AGENTS_PER_BATCH, len(entities))
            batch = entities[start:end]
            return batch_idx, generate_agent_configs_batch(
                self._llm, context, batch, start, simulation_requirement)

        results_by_idx = {}
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(_gen_batch, bi): bi for bi in range(num_batches)}
            for future in as_completed(futures):
                batch_idx, batch_configs = future.result()
                results_by_idx[batch_idx] = batch_configs
                done = len(results_by_idx)
                report_progress(3 + done - 1,
                                f"Agent config batch {done}/{num_batches} done")

        all_configs = []
        for idx in range(num_batches):
            all_configs.extend(results_by_idx[idx])
        return all_configs

    @staticmethod
    def _build_platform_configs(enable_twitter, enable_reddit):
        """Build platform-specific configurations."""
        twitter_config = None
        reddit_config = None
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter", recency_weight=0.4,
                popularity_weight=0.3, relevance_weight=0.3,
                viral_threshold=10, echo_chamber_strength=0.5)
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit", recency_weight=0.3,
                popularity_weight=0.4, relevance_weight=0.3,
                viral_threshold=15, echo_chamber_strength=0.6)
        return twitter_config, reddit_config
