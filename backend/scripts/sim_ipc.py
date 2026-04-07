"""
IPC handler for receiving interview and control commands while the simulation
environment is kept alive after the main loop completes.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from sim_constants import IPC_COMMANDS_DIR, IPC_RESPONSES_DIR, ENV_STATUS_FILE

try:
    from wonderwall import ActionType, ManualAction
except ImportError:
    ActionType = None
    ManualAction = None


class CommandType:
    """Command type string constants."""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class ParallelIPCHandler:
    """Dual-platform IPC command handler (Twitter + Reddit)."""

    def __init__(self, simulation_dir, twitter_env=None, twitter_agent_graph=None,
                 reddit_env=None, reddit_agent_graph=None):
        self.simulation_dir = simulation_dir
        self.twitter_env = twitter_env
        self.twitter_agent_graph = twitter_agent_graph
        self.reddit_env = reddit_env
        self.reddit_agent_graph = reddit_agent_graph
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)

    def update_status(self, status: str) -> None:
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status, "pid": os.getpid(),
                "twitter_available": self.twitter_env is not None,
                "reddit_available": self.reddit_env is not None,
                "timestamp": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)

    def poll_command(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.commands_dir):
            return None
        files = sorted(
            (os.path.join(self.commands_dir, f), os.path.getmtime(os.path.join(self.commands_dir, f)))
            for f in os.listdir(self.commands_dir) if f.endswith('.json')
        )
        for filepath, _ in files:
            try:
                with open(filepath, 'r', encoding='utf-8') as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def send_response(self, command_id: str, status: str,
                      result: Dict = None, error: str = None) -> None:
        response = {"command_id": command_id, "status": status,
                    "result": result, "error": error,
                    "timestamp": datetime.now().isoformat()}
        with open(os.path.join(self.responses_dir, f"{command_id}.json"), 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        try:
            os.remove(os.path.join(self.commands_dir, f"{command_id}.json"))
        except OSError:
            pass

    def _get_env_and_graph(self, platform: str):
        if platform == "twitter" and self.twitter_env:
            return self.twitter_env, self.twitter_agent_graph, "twitter"
        elif platform == "reddit" and self.reddit_env:
            return self.reddit_env, self.reddit_agent_graph, "reddit"
        return None, None, None

    async def _interview_single_platform(self, agent_id: int, prompt: str, platform: str) -> Dict:
        env, agent_graph, actual_platform = self._get_env_and_graph(platform)
        if not env or not agent_graph:
            return {"platform": platform, "error": f"{platform} platform is not available"}
        try:
            agent = agent_graph.get_agent(agent_id)
            await env.step({agent: ManualAction(action_type=ActionType.INTERVIEW,
                                                action_args={"prompt": prompt})})
            result = self._get_interview_result(agent_id, actual_platform)
            result["platform"] = actual_platform
            return result
        except Exception as e:
            return {"platform": platform, "error": str(e)}

    async def handle_interview(self, command_id: str, agent_id: int,
                               prompt: str, platform: str = None) -> bool:
        if platform in ("twitter", "reddit"):
            result = await self._interview_single_platform(agent_id, prompt, platform)
            if "error" in result:
                self.send_response(command_id, "failed", error=result["error"])
                return False
            self.send_response(command_id, "completed", result=result)
            return True

        if not self.twitter_env and not self.reddit_env:
            self.send_response(command_id, "failed", error="No simulation environment available")
            return False

        tasks, platform_names = [], []
        if self.twitter_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "twitter"))
            platform_names.append("twitter")
        if self.reddit_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "reddit"))
            platform_names.append("reddit")

        import asyncio
        platform_results = await asyncio.gather(*tasks)
        results = {"agent_id": agent_id, "prompt": prompt, "platforms": {}}
        success_count = 0
        for name, res in zip(platform_names, platform_results):
            results["platforms"][name] = res
            if "error" not in res:
                success_count += 1

        if success_count > 0:
            self.send_response(command_id, "completed", result=results)
            return True
        errors = [f"{p}: {r.get('error', 'unknown')}" for p, r in results["platforms"].items()]
        self.send_response(command_id, "failed", error="; ".join(errors))
        return False

    async def handle_batch_interview(self, command_id: str, interviews: List[Dict],
                                     platform: str = None) -> bool:
        twitter_ivs, reddit_ivs = [], []
        both = []
        for iv in interviews:
            p = iv.get("platform", platform)
            if p == "twitter":
                twitter_ivs.append(iv)
            elif p == "reddit":
                reddit_ivs.append(iv)
            else:
                both.append(iv)
        if both:
            if self.twitter_env:
                twitter_ivs.extend(both)
            if self.reddit_env:
                reddit_ivs.extend(both)

        results: Dict[str, Any] = {}

        async def _run_batch(ivs, env, graph, platform_name):
            actions = {}
            for iv in ivs:
                aid = iv.get("agent_id")
                try:
                    agent = graph.get_agent(aid)
                    actions[agent] = ManualAction(
                        action_type=ActionType.INTERVIEW,
                        action_args={"prompt": iv.get("prompt", "")},
                    )
                except Exception as e:
                    print(f"  Warning: Cannot get {platform_name} Agent {aid}: {e}")
            if actions:
                await env.step(actions)
            for iv in ivs:
                aid = iv.get("agent_id")
                r = self._get_interview_result(aid, platform_name)
                r["platform"] = platform_name
                results[f"{platform_name}_{aid}"] = r

        try:
            if twitter_ivs and self.twitter_env:
                await _run_batch(twitter_ivs, self.twitter_env, self.twitter_agent_graph, "twitter")
        except Exception as e:
            print(f"  Twitter batch interview failed: {e}")
        try:
            if reddit_ivs and self.reddit_env:
                await _run_batch(reddit_ivs, self.reddit_env, self.reddit_agent_graph, "reddit")
        except Exception as e:
            print(f"  Reddit batch interview failed: {e}")

        if results:
            self.send_response(command_id, "completed",
                               result={"interviews_count": len(results), "results": results})
            return True
        self.send_response(command_id, "failed", error="No successful interviews")
        return False

    def _get_interview_result(self, agent_id: int, platform: str) -> Dict[str, Any]:
        db_path = os.path.join(self.simulation_dir, f"{platform}_simulation.db")
        result: Dict[str, Any] = {"agent_id": agent_id, "response": None, "timestamp": None}
        if not os.path.exists(db_path):
            return result
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id, info, created_at FROM trace "
                "WHERE action = ? AND user_id = ? ORDER BY created_at DESC LIMIT 1",
                (ActionType.INTERVIEW.value, agent_id),
            )
            row = cursor.fetchone()
            if row:
                _, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    result["response"] = info.get("response", info)
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json
            conn.close()
        except Exception as e:
            print(f"  Failed to read interview result: {e}")
        return result

    async def process_commands(self) -> bool:
        command = self.poll_command()
        if not command:
            return True
        cid = command.get("command_id")
        ctype = command.get("command_type")
        args = command.get("args", {})
        print(f"\nReceived IPC command: {ctype}, id={cid}")
        if ctype == CommandType.INTERVIEW:
            await self.handle_interview(cid, args.get("agent_id", 0),
                                        args.get("prompt", ""), args.get("platform"))
            return True
        elif ctype == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(cid, args.get("interviews", []),
                                              args.get("platform"))
            return True
        elif ctype == CommandType.CLOSE_ENV:
            print("Received close environment command")
            self.send_response(cid, "completed", result={"message": "Environment is shutting down"})
            return False
        else:
            self.send_response(cid, "failed", error=f"Unknown command type: {ctype}")
            return True
