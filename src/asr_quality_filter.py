from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass
class ASRQualityDecision:
    accepted: bool
    reason: str


class ASRQualityFilter:
    """
    Conservative filter for obviously bad ASR fragments.

    Conversation-mode design:
      - Do not overfit to radio/news scripts.
      - Keep short but valid utterances such as はい, いいです, そうです.
      - Drop obvious non-speech noises such as コンッ, カンッ.
      - Avoid dropping useful single ASR segments only because Whisper's
        segment confidence metadata is temporarily unstable.
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
        short_noise_filter_enabled: bool = True,
        leading_fragment_filter_enabled: bool = True,
        prompt_leak_filter_enabled: bool = True,
    ):
        self.enabled = enabled
        self.min_text_len = min_text_len
        self.min_japanese_ratio = min_japanese_ratio
        self.max_repeated_char_ratio = max_repeated_char_ratio
        self.max_compression_ratio = max_compression_ratio
        self.min_avg_logprob = min_avg_logprob
        self.max_no_speech_prob = max_no_speech_prob

        self.short_noise_filter_enabled = short_noise_filter_enabled
        self.leading_fragment_filter_enabled = leading_fragment_filter_enabled
        self.prompt_leak_filter_enabled = prompt_leak_filter_enabled

    def accept_text(self, text: str, direction: str) -> ASRQualityDecision:
        if not self.enabled:
            return ASRQualityDecision(True, "disabled")

        text = text.strip()

        if len(text) < self.min_text_len:
            return ASRQualityDecision(False, "too_short")

        if self.prompt_leak_filter_enabled and self._looks_like_prompt_leak(text):
            return ASRQualityDecision(False, "prompt_leak")

        if direction == "ja2ko":
            if self.short_noise_filter_enabled and self._is_short_noise_fragment(text):
                return ASRQualityDecision(False, "short_noise_fragment")

            # if self.leading_fragment_filter_enabled and self._is_leading_particle_fragment(text):
            #     return ASRQualityDecision(False, "leading_particle_fragment")

        if self._repeated_char_ratio(text) > self.max_repeated_char_ratio:
            return ASRQualityDecision(False, "repeated_chars")

        if direction == "ja2ko":
            ratio = self._japanese_char_ratio(text)
            if ratio < self.min_japanese_ratio:
                return ASRQualityDecision(False, f"low_japanese_ratio:{ratio:.2f}")

        return ASRQualityDecision(True, "text_ok")

    def accept_segments(self, segments: Iterable) -> ASRQualityDecision:
        """
        Whisper segment metadata is useful, but in short real-time chunks it can
        be unstable. Therefore:
          - no segment         -> reject
          - one bad segment    -> soft accept; text filter decides
          - all bad multi-seg  -> reject
          - mixed segments     -> accept
        """
        if not self.enabled:
            return ASRQualityDecision(True, "disabled")

        checked = 0
        bad = 0
        reasons = []

        for seg in segments:
            checked += 1

            compression_ratio = getattr(seg, "compression_ratio", None)
            avg_logprob = getattr(seg, "avg_logprob", None)
            no_speech_prob = getattr(seg, "no_speech_prob", None)

            if compression_ratio is not None and compression_ratio > self.max_compression_ratio:
                bad += 1
                reasons.append("compression_ratio")
                continue

            if avg_logprob is not None and avg_logprob < self.min_avg_logprob:
                bad += 1
                reasons.append("avg_logprob")
                continue

            if no_speech_prob is not None and no_speech_prob > self.max_no_speech_prob:
                bad += 1
                reasons.append("no_speech_prob")
                continue

        if checked == 0:
            return ASRQualityDecision(False, "no_segments")

        if bad == checked:
            if checked == 1:
                return ASRQualityDecision(True, "single_segment_low_quality_soft_accept")
            reason = ",".join(sorted(set(reasons))) if reasons else "unknown"
            return ASRQualityDecision(False, f"all_segments_low_quality:{reason}")

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

    @staticmethod
    def _is_short_noise_fragment(text: str) -> bool:
        """
        Drop short non-speech fragments caused by clicks, bumps, keyboard sounds,
        or loopback artifacts. Keep real conversational fillers such as あの,
        えっと, はい, うん.
        """
        t = text.strip()
        t = t.strip("。.!?！？、, 　")

        noise_tokens = {
            "コン",
            "コンッ",
            "カン",
            "カンッ",
            "ドン",
            "ドンッ",
            "パン",
            "パンッ",
            "ガサ",
            "ガサッ",
            "ザッ",
            "ノイズ",
            "雑音",
        }

        if t in noise_tokens:
            return True

        # Very short katakana-only impulse sounds.
        if len(t) <= 4 and re.fullmatch(r"[\u30a0-\u30ffーッ]+", t):
            safe_fillers = {"えっと", "エット", "あの", "アノ"}
            if t not in safe_fillers:
                return True

        return False

    @staticmethod
    def _is_leading_particle_fragment(text: str) -> bool:
        """
        Drop short leftover fragments that begin with particles and are unlikely
        to be standalone utterances.

        Conservative list:
          - Do not drop は/が/に/で because they may begin valid words or phrases.
          - Drop の..., を..., から..., まで..., より... only when short.
        """
        t = text.strip()
        t = t.strip("。.!?！？、, 　")

        if len(t) > 28:
            return False

        return bool(re.match(r"^(?:の|を|から|まで|より)", t))

    @staticmethod
    def _looks_like_prompt_leak(text: str) -> bool:
        """
        Guard against ASR initial_prompt text leaking into transcription.
        This should rarely trigger when initial_prompt is null, but it prevents
        previous prompt text from becoming translated output.
        """
        t = text.strip()

        prompt_like_patterns = (
            "日本語の日常会話です",
            "質問、説明、確認、依頼",
            "相づち、短い返事",
            "会議での自然な発言",
            "会議での自然な発話",
        )

        return any(p in t for p in prompt_like_patterns)
