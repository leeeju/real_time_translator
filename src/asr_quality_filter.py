from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ASRQualityDecision:
    accepted: bool
    reason: str


class ASRQualityFilter:
    """
    Conservative filter for obviously bad ASR fragments.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_text_len: int = 2,
        min_japanese_ratio: float = 0.30,
        max_repeated_char_ratio: float = 0.50,
        max_compression_ratio: float = 3.5,
        min_avg_logprob: float = -1.50,
        max_no_speech_prob: float = 0.85,
    ):
        self.enabled = enabled
        self.min_text_len = min_text_len
        self.min_japanese_ratio = min_japanese_ratio
        self.max_repeated_char_ratio = max_repeated_char_ratio
        self.max_compression_ratio = max_compression_ratio
        self.min_avg_logprob = min_avg_logprob
        self.max_no_speech_prob = max_no_speech_prob

    def accept_text(self, text: str, direction: str) -> ASRQualityDecision:
        if not self.enabled:
            return ASRQualityDecision(True, "disabled")

        text = text.strip()

        if len(text) < self.min_text_len:
            return ASRQualityDecision(False, "too_short")

        if self._repeated_char_ratio(text) > self.max_repeated_char_ratio:
            return ASRQualityDecision(False, "repeated_chars")

        if direction == "ja2ko":
            ratio = self._japanese_char_ratio(text)
            if ratio < self.min_japanese_ratio:
                return ASRQualityDecision(False, f"low_japanese_ratio:{ratio:.2f}")

        return ASRQualityDecision(True, "text_ok")

    def accept_segments(self, segments: Iterable) -> ASRQualityDecision:
        if not self.enabled:
            return ASRQualityDecision(True, "disabled")

        checked = 0
        bad = 0

        for seg in segments:
            checked += 1

            compression_ratio = getattr(seg, "compression_ratio", None)
            avg_logprob = getattr(seg, "avg_logprob", None)
            no_speech_prob = getattr(seg, "no_speech_prob", None)

            if compression_ratio is not None and compression_ratio > self.max_compression_ratio:
                bad += 1
                continue

            if avg_logprob is not None and avg_logprob < self.min_avg_logprob:
                bad += 1
                continue

            if no_speech_prob is not None and no_speech_prob > self.max_no_speech_prob:
                bad += 1
                continue

        if checked == 0:
            return ASRQualityDecision(False, "no_segments")

        if bad == checked:
            return ASRQualityDecision(False, "all_segments_low_quality")

        return ASRQualityDecision(True, "segments_ok")

    @staticmethod
    def _japanese_char_ratio(text: str) -> float:
        if not text:
            return 0.0

        jp = 0
        for ch in text:
            if ("\u3040" <= ch <= "\u30ff") or ("\u3400" <= ch <= "\u9fff"):
                jp += 1

        return jp / max(len(text), 1)

    @staticmethod
    def _repeated_char_ratio(text: str) -> float:
        if not text:
            return 0.0

        counts = {}
        for ch in text:
            if ch.isspace():
                continue
            counts[ch] = counts.get(ch, 0) + 1

        if not counts:
            return 0.0

        return max(counts.values()) / max(sum(counts.values()), 1)

