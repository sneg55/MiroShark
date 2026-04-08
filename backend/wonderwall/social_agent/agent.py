# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager
from camel.prompts import TextPrompt
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole

from wonderwall.social_agent.agent_action import SocialAction
from wonderwall.social_agent.agent_actions_mixin import AgentActionsMixin
from wonderwall.social_agent.agent_environment import SocialEnvironment
from wonderwall.social_platform import Channel
from wonderwall.social_platform.config import UserInfo
from wonderwall.social_platform.typing import ActionType

if TYPE_CHECKING:
    from wonderwall.social_agent import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")

    if not agent_log.handlers:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(
            f"./log/social.agent-{str(now)}.log")
        file_handler.setLevel("DEBUG")
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
        agent_log.addHandler(file_handler)

ALL_SOCIAL_ACTIONS = [action.value for action in ActionType]


class SocialAgent(AgentActionsMixin, ChatAgent):
    r"""Agent that participates in an OASIS simulation.

    Supports both the legacy social-media workflow (``SocialAction`` +
    ``SocialEnvironment``) and the new generic simulation framework
    (any ``BaseAction`` + ``BaseEnvironment`` via ``SimulationConfig``).
    """

    def __init__(self,
                 agent_id: int,
                 user_info: UserInfo,
                 user_info_template: TextPrompt | None = None,
                 channel: Channel | None = None,
                 model: Optional[Union[BaseModelBackend,
                                       List[BaseModelBackend],
                                       ModelManager]] = None,
                 agent_graph: "AgentGraph" = None,
                 available_actions: list[ActionType] = None,
                 tools: Optional[List[Union[FunctionTool, Callable]]] = None,
                 max_iteration: int = 1,
                 interview_record: bool = False,
                 # --- New: generic simulation support ---
                 simulation=None):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.channel = channel or Channel()

        # ------------------------------------------------------------------
        # Build action/environment/prompt from SimulationConfig if provided
        # ------------------------------------------------------------------
        if simulation is not None:
            from wonderwall.simulations.base import SimulationConfig
            if isinstance(simulation, SimulationConfig):
                action_instance = simulation.action_cls(agent_id, self.channel)
                self.env = simulation.environment_cls(action_instance)
                if user_info_template is not None:
                    system_message_content = (
                        self.user_info.to_custom_system_message(
                            user_info_template))
                else:
                    system_message_content = (
                        simulation.prompt_builder.build_system_prompt(
                            user_info))
                # Default actions from SimulationConfig if none specified
                if available_actions is None and simulation.default_actions:
                    available_actions = simulation.default_actions
            else:
                raise ValueError(
                    f"simulation must be a SimulationConfig, got "
                    f"{type(simulation)}")
        else:
            # Legacy path: social media
            self.env = SocialEnvironment(
                SocialAction(agent_id, self.channel))
            if user_info_template is None:
                system_message_content = self.user_info.to_system_message()
            else:
                system_message_content = (
                    self.user_info.to_custom_system_message(
                        user_info_template))

        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=system_message_content,
        )

        if not available_actions:
            agent_log.info("No available actions defined, using all actions.")
            self.action_tools = self.env.action.get_openai_function_list()
        else:
            all_tools = self.env.action.get_openai_function_list()
            all_possible_actions = [tool.func.__name__ for tool in all_tools]

            for action in available_actions:
                action_name = action.value if isinstance(
                    action, ActionType) else action
                if action_name not in all_possible_actions:
                    agent_log.warning(
                        f"Action {action_name} is not supported. Supported "
                        f"actions are: {', '.join(all_possible_actions)}")
            self.action_tools = [
                tool for tool in all_tools if tool.func.__name__ in [
                    a.value if isinstance(a, ActionType) else a
                    for a in available_actions
                ]
            ]
        all_tools = (tools or []) + (self.action_tools or [])
        super().__init__(
            system_message=system_message,
            model=model,
            scheduling_strategy='random_model',
            tools=all_tools,
        )
        self.max_iteration = max_iteration
        self.interview_record = interview_record
        self.agent_graph = agent_graph
        self.test_prompt = (
            "\n"
            "Helen is a successful writer who usually writes popular western "
            "novels. Now, she has an idea for a new novel that could really "
            "make a big impact. If it works out, it could greatly "
            "improve her career. But if it fails, she will have spent "
            "a lot of time and effort for nothing.\n"
            "\n"
            "What do you think Helen should do?")

    async def perform_action_by_llm(self):
        # Get environment observation:
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform actions after observing the "
                f"platform environment. Use the available tools to take "
                f"action. Don't limit yourself to just one type of action. "
                f"Here is your current environment: {env_prompt}"))
        try:
            agent_log.info(
                f"Agent {self.social_agent_id} observing environment: "
                f"{env_prompt}")
            response = await self.astep(user_msg)
            if not response.info.get('tool_calls'):
                agent_log.warning(
                    f"Agent {self.social_agent_id} returned no tool calls, "
                    f"falling back to do_nothing")
                await self.env.action.do_nothing()
                return response
            for tool_call in response.info['tool_calls']:
                action_name = tool_call.tool_name
                args = tool_call.args
                agent_log.info(f"Agent {self.social_agent_id} performed "
                               f"action: {action_name} with args: {args}")
                if action_name not in ALL_SOCIAL_ACTIONS:
                    agent_log.info(
                        f"Agent {self.social_agent_id} get the result: "
                        f"{tool_call.result}")

                return response
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            return e

    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        r"""Remove edge if action is unfollow or add edge
        if action is follow to the agent graph.
        """
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} unfollowed Agent {followee_id}")
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} followed Agent {followee_id}")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.social_agent_id}, "
                f"model_type={self.model_type.value})")
