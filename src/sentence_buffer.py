from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from .glossary import is_weak_japanese_fragment


@dataclass
class BufferResult:
    ready: bool
    text: str
    reason: str


def normalize_asr_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("。。", "。").replace("？？", "？").replace("！！", "！")
    return text


def is_near_duplicate(prev: str, current: str) -> bool:
    prev = normalize_asr_text(prev)
    current = normalize_asr_text(current)

    if not prev or not current:
        return False

    if prev == current:
        return True

    if current in prev and len(prev) - len(current) <= 4:
        return True

    if prev in current and len(current) - len(prev) <= 4:
        return True

    return False


class SentenceBuffer:
    """
    Generic sentence buffer for real-time interpretation.

    This class must not contain fixed test sentences.
    It uses language-structure rules only:
      - Hold Japanese quote particles like "...と。" until the next clause arrives.
      - Hold statistic fragments like "答えた人は26%" if no quoted content exists yet.
      - Suppress weak fragments such as "でした。" alone.
    """

    def __init__(
        self,
        direction: str = "ja2ko",
        max_chars: int = 140,
        max_wait_sec: float = 4.0,
        min_chars_to_translate: int = 6,
    ):
        self.direction = direction
        self.max_chars = max_chars
        self.max_wait_sec = max_wait_sec
        self.min_chars_to_translate = min_chars_to_translate
        self.buffer = ""
        self.last_update_time: Optional[float] = None

    def add(self, text: str) -> BufferResult:
        now = time.perf_counter()
        text = normalize_asr_text(text)

        if not text:
            return BufferResult(False, "", "empty")

        if self.last_update_time is None:
            self.last_update_time = now

        self.buffer = self._merge(self.buffer, text)
        self.last_update_time = now

        if len(self.buffer) < self.min_chars_to_translate:
            return BufferResult(False, "", "too_short")

        if self.direction == "ja2ko":
            if is_weak_japanese_fragment(self.buffer):
                return BufferResult(False, "", "weak_fragment")

            if self._ends_with_open_japanese_quote_marker(self.buffer):
                return BufferResult(False, "", "open_quote_marker")

            if self._is_statistic_answer_without_context(self.buffer):
                return BufferResult(False, "", "statistic_fragment_waiting_context")

        if self._is_sentence_complete(self.buffer):
            return BufferResult(True, self.flush(), "sentence_end")

        if len(self.buffer) >= self.max_chars:
            return BufferResult(True, self.flush(), "max_chars")

        return BufferResult(False, "", "buffering")

    def force_flush_if_timeout(self) -> BufferResult:
        if not self.buffer or self.last_update_time is None:
            return BufferResult(False, "", "empty")

        elapsed = time.perf_counter() - self.last_update_time

        if self.direction == "ja2ko":
            if is_weak_japanese_fragment(self.buffer):
                return BufferResult(False, "", "weak_fragment_timeout")

            # Hold quote markers a bit longer. Translating "...と。" alone is usually wrong.
            if self._ends_with_open_japanese_quote_marker(self.buffer):
                return BufferResult(False, "", "open_quote_marker_timeout")

            if self._is_statistic_answer_without_context(self.buffer):
                return BufferResult(False, "", "statistic_fragment_timeout")

        if elapsed >= self.max_wait_sec and len(self.buffer) >= self.min_chars_to_translate:
            return BufferResult(True, self.flush(), "timeout")

        return BufferResult(False, "", "waiting")

    def flush(self) -> str:
        text = normalize_asr_text(self.buffer)
        self.buffer = ""
        self.last_update_time = None
        return text

    def _merge(self, prev: str, current: str) -> str:
        prev = normalize_asr_text(prev)
        current = normalize_asr_text(current)

        if not prev:
            return current

        if not current:
            return prev

        if current == prev or current in prev:
            return prev

        # Generic overlap removal.
        tail_max = min(len(prev), len(current), 32)
        for n in range(tail_max, 3, -1):
            if prev[-n:] == current[:n]:
                return normalize_asr_text(prev + current[n:])

        if self.direction == "ja2ko":
            return normalize_asr_text(prev + " " + current)

        return normalize_asr_text(prev + " " + current)

    def _is_sentence_complete(self, text: str) -> bool:
        text = normalize_asr_text(text)

        if self.direction == "ja2ko":
            if self._ends_with_open_japanese_quote_marker(text):
                return False

            if self._is_statistic_answer_without_context(text):
                return False

            if text.endswith(("。", "？", "?", "！", "!")):
                return True

            endings = (
                "です", "ます", "ました", "ません", "でした", "だった",
                "あります", "ありません", "と思います", "してください",
                "でしょうか", "ですか", "ますか", "終えました",
                "発表しました", "述べました", "%でした", "%です",
            )
            return text.endswith(endings)

        if text.endswith((".", "?", "!", "？", "！")):
            return True

        endings = (
            "요", "다", "니다", "습니까", "니까", "입니다", "합니다",
            "했습니다", "하겠습니다", "겠습니다", "됩니다", "됐습니다",
            "까요", "네요",
        )
        return text.endswith(endings)

    @staticmethod
    def _ends_with_open_japanese_quote_marker(text: str) -> bool:
        t = normalize_asr_text(text)
        # Generic quoted-clause continuation:
        #   ...と
        #   ...と。
        #   ...と、
        #   ...って
        #   ...って。
        return bool(re.search(r"(?:、)?(?:と|って)(?:。|、)?$", t))

    @staticmethod
    def _is_statistic_answer_without_context(text: str) -> bool:
        t = normalize_asr_text(text)
        # Generic survey/statistic fragment with no preceding quoted content.
        # Example fragment: "答えた人は26%。"
        has_stat = re.search(
            r"(?:答えた|回答した|選んだ|支持した|反対した|賛成した)人は\s*\d+(?:\.\d+)?\s*%?",
            t,
        ) is not None
        has_context = re.search(
            r"(?:と|って)(?:答えた|回答した|選んだ|支持した|反対した|賛成した)人は",
            t,
        ) is not None
        return has_stat and not has_context and len(t) <= 40
