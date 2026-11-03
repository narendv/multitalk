"""The Planner and Analyzer LLM agents."""

from __future__ import annotations

from pathlib import Path

from ..config import PROMPTS_DIR
from .client import ChatClient, Usage


def _read(agent: str, name: str) -> str:
    return (PROMPTS_DIR / agent / f"{name}.txt").read_text(encoding="utf-8")


def _load(agent: str, name: str) -> str:
    return _read(agent, name).strip()


def _fill(template: str, **values: str) -> str:
    """Fill {name} placeholders. str.format would break on the JSON in scenes."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


class Conversation:
    def __init__(self, system_prompt: str) -> None:
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user(self, content: str) -> None:
        if content:
            self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        if content:
            self.messages.append({"role": "assistant", "content": content})

    def last_assistant_message(self) -> str:
        last = self.messages[-1]
        return last["content"] if last["role"] == "assistant" else self.messages[-2]["content"]

    def save(self, path: str | Path) -> None:
        text = "".join(f"{m['role']}:\n {m['content']}\n\n" for m in self.messages)
        Path(path).write_text(text, encoding="utf-8")


class _Agent:
    agent = ""

    def __init__(self, client: ChatClient, usage: Usage) -> None:
        self._client = client
        self._usage = usage
        self.conversation = Conversation(_load(self.agent, "system"))

    def _respond(self) -> str:
        completion = self._client.complete(self.conversation.messages)
        self._usage.add(completion)
        self.conversation.add_assistant(completion.text)
        return self.conversation.last_assistant_message()


class Planner(_Agent):
    agent = "planner"

    def call(self, call_type: str, task: str, environment: str, feedback: str | None = None) -> str:
        """Send the first query, or feedback from 'analyzer', 'human',
        'simulator' or 'perception', and return the reply."""
        if call_type == "first":
            parts = [
                _load("planner", name)
                for name in ("prompt_function", "prompt_environment", "prompt_output_format")
            ]
            parts.append(_fill(_load("planner", "query"), objects=environment, task=task).strip())
            self.conversation.add_user("\n".join(parts))
        elif call_type == "perception":
            self.conversation.add_user(_fill(_read("planner", "feedback_perception"), objects=feedback))
        else:
            self.conversation.add_user(
                _fill(_load("planner", f"feedback_{call_type}"), feedback=feedback)
            )
        return self._respond()


class Analyzer(_Agent):
    agent = "analyzer"

    def call(
        self,
        call_type: str,
        task: str,
        environment: str,
        plan: str,
        clarifications: str,
        updated_environment: str | None = None,
    ) -> str:
        """Review a plan. 'first' sends the full context; 'feedback' sends the new plan,
        preceded by the rescanned environment if there is one."""
        clarifications = clarifications.rstrip("\n") if clarifications else "None"
        if call_type == "first":
            parts = [
                _load("analyzer", name)
                for name in (
                    "prompt_input_format",
                    "prompt_critic",
                    "observed_environment",
                    "prompt_restrictions",
                    "prompt_output_format",
                )
            ]
            self.conversation.add_user("\n".join(parts))
            self.conversation.add_user(
                _fill(
                    _read("analyzer", "query"),
                    task=task,
                    plan=plan,
                    clarifications=clarifications,
                    objects=environment,
                    semantic_relations="None",
                )
            )
        else:
            if updated_environment is not None:
                self.conversation.add_user(
                    _fill(_read("analyzer", "updated_environment"), objects=updated_environment)
                )
            self.conversation.add_user(
                _fill(_read("analyzer", "feedback"), plan=plan, clarifications=clarifications)
            )
        return self._respond()
