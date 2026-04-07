"""
Simulation configuration type definitions — dataclasses for config parameters.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


# China timezone activity configuration (Beijing Time)
CHINA_TIMEZONE_CONFIG = {
    "dead_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "peak_hours": [19, 20, 21, 22],
    "night_hours": [23],
    "activity_multipliers": {
        "dead": 0.05,
        "morning": 0.4,
        "work": 0.7,
        "peak": 1.5,
        "night": 0.5,
    }
}


@dataclass
class AgentActivityConfig:
    """Activity configuration for a single Agent"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    activity_level: float = 0.5
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    response_delay_min: int = 5
    response_delay_max: int = 60
    sentiment_bias: float = 0.0
    stance: str = "neutral"
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Time simulation configuration (based on Chinese daily schedule)"""
    total_simulation_hours: int = 72
    minutes_per_round: int = 60
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05

    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    work_hours: List[int] = field(
        default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Event configuration"""
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    hot_topics: List[str] = field(default_factory=list)
    narrative_direction: str = ""
    initial_markets: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PlatformConfig:
    """Platform-specific configuration.

    NOTE ON DEFAULT VALUES: The defaults below are heuristic starting points,
    NOT calibrated against real platform data. They exist because the OASIS
    simulation framework requires these parameters. Users should treat them as
    tunable scenario assumptions, not empirical constants.

    - viral_threshold: In a simulation of ~50-200 agents, 10-15 interactions is
      roughly the top 5-10% of posts.
    - echo_chamber_strength: 0.0 = no clustering, 1.0 = fully siloed. The defaults
      (0.5 Twitter, 0.6 Reddit) reflect the assumption that Reddit's subreddit
      structure creates slightly stronger echo chambers than Twitter's timeline.
    - Recommendation weights: These control the feed algorithm's trade-off between
      fresh content, popular content, and content relevant to the agent's interests.
    """
    platform: str  # twitter or reddit

    recency_weight: float = 0.4
    popularity_weight: float = 0.3
    relevance_weight: float = 0.3
    viral_threshold: int = 10
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Complete simulation parameter configuration"""
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    event_config: EventConfig = field(default_factory=EventConfig)
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    llm_model: str = ""
    llm_base_url: str = ""

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
