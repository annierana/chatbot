"""Interactive chatbot backed by Groq's OpenAI-compatible endpoint."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI, APIError


DEFAULT_MODEL = "openai/gpt-oss-20b"
SUMMARY_EVERY_N = 5          # summarize every N user turns
KEEP_RECENT = 5              # messages kept verbatim after summarization
SUMMARY_MAX_TOKENS = 256     # hard cap on summary length
MAX_HISTORY_MESSAGES = 40    # absolute safety bound


@dataclass
class TokenStats:
    input: int = 0
    output: int = 0
    total: int = 0

    def add_usage(self, usage: Any | None) -> None:
        if usage is None:
            return
        self.input += getattr(usage, "prompt_tokens", 0) or 0
        self.output += getattr(usage, "completion_tokens", 0) or 0
        self.total += getattr(usage, "total_tokens", 0) or 0

    def __str__(self) -> str:
        return (
            f"Input tokens: {self.input}\n"
            f"Output tokens: {self.output}\n"
            f"Total tokens: {self.total}"
        )


@dataclass
class ChatBot:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str = "https://api.groq.com/openai/v1"

    _messages: list[dict[str, str]] = field(default_factory=list, init=False)
    _stats: TokenStats = field(default_factory=TokenStats, init=False)
    _iteration: int = field(default=0, init=False)
    _llm: OpenAI = field(init=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )
        self._llm = OpenAI(api_key=key, base_url=self.base_url)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def ask(self, message: str) -> str:
        """Send one user message and return the assistant's reply."""
        self._messages.append({"role": "user", "content": message})

        response = self._chat_completion(self._messages)
        reply = response.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": reply})
        self._stats.add_usage(response.usage)

        self._iteration += 1
        if self._iteration % SUMMARY_EVERY_N == 0:
            self._summarize_history()

        return reply

    def stats(self) -> TokenStats:
        return self._stats

    def reset(self) -> None:
        self._messages.clear()
        self._stats = TokenStats()
        self._iteration = 0

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _chat_completion(self, messages: list[dict[str, str]], **kwargs: Any):
        try:
            return self._llm.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        except APIError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    def _summarize_history(self) -> None:
        """Collapse the oldest non-system messages into one system summary."""
        system_msgs = [m for m in self._messages if m["role"] == "system"]
        convo = [m for m in self._messages if m["role"] != "system"]

        # Nothing to summarize if we already have few messages.
        if len(convo) <= KEEP_RECENT:
            return

        early, recent = convo[:-KEEP_RECENT], convo[-KEEP_RECENT:]

        # Hard safety valve: even after summarization, cap total history.
        if len(self._messages) > MAX_HISTORY_MESSAGES:
            recent = recent[-KEEP_RECENT:]

        try:
            response = self._chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize the conversation below concisely, "
                            "preserving key facts, decisions, and user "
                            "preferences. Do not add commentary."
                        ),
                    },
                    *early,
                ],
                max_tokens=SUMMARY_MAX_TOKENS,
            )
        except RuntimeError:
            # Never let summarization break the conversation.
            return

        self._stats.add_usage(response.usage)
        summary_text = (response.choices[0].message.content or "").strip()

        summary_msg = {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary_text}",
        }

        # Preserve original system prompts, then insert summary, then recent.
        self._messages = [*system_msgs, summary_msg, *recent]


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #

def main() -> None:
    load_dotenv()
    bot = ChatBot()

    while True:
        user_input = input("Ask something (type Q to quit): ").strip()
        if user_input.lower() == "q":
            print("Goodbye!")
            break
        if not user_input:
            continue

        try:
            reply = bot.ask(user_input)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            continue

        print(reply)
        print(bot.stats())


if __name__ == "__main__":
    main()