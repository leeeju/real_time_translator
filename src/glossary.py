from __future__ import annotations

import re
from typing import Dict, Mapping


class Glossary:
    """
    Lightweight ASR correction + generic Japanese normalizer.

    No fixed test sentences should be placed here.
    Use:
      - user-editable glossary terms
      - punctuation cleanup
      - generic number/percent normalization
      - generic quote/statistic joining
    """

    def __init__(
        self,
        enabled: bool = True,
        ja_asr_corrections: Mapping[str, str] | None = None,
        ko_asr_corrections: Mapping[str, str] | None = None,
    ):
        self.enabled = enabled
        self.ja_asr_corrections: Dict[str, str] = dict(ja_asr_corrections or {})
        self.ko_asr_corrections: Dict[str, str] = dict(ko_asr_corrections or {})

    def apply_asr_corrections(self, text: str, direction: str) -> str:
        if not text:
            return text

        out = text.strip()

        if self.enabled:
            corrections = self._corrections_for_direction(direction)
            for src, dst in corrections.items():
                out = out.replace(src, dst)

        if direction == "ja2ko":
            out = normalize_japanese_for_translation(out)
        elif direction.startswith("ko2"):
            out = normalize_korean_for_translation(out)
        else:
            out = normalize_generic_for_translation(out)

        return out.strip()

    def apply_translation_postprocess(self, text: str, direction: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _corrections_for_direction(self, direction: str) -> Mapping[str, str]:
        if direction.startswith("ja2"):
            return self.ja_asr_corrections
        if direction.startswith("ko2"):
            return self.ko_asr_corrections
        return {}


def normalize_japanese_for_translation(text: str) -> str:
    """
    Generic Japanese cleanup before NLLB translation.

    This normalizer should not hard-code one specific experiment sentence.
    """
    text = _squash_spaces(text)

    # Generic repeated phrase cleanup caused by overlap.
    # Example: "来年、来年の夏" -> "来年の夏"
    text = _remove_short_overlap_repetition(text)

    # Normalize percent expressions.
    text = re.sub(r"(\d+)\s*パー\s*1\s*セント", r"\1%", text)
    text = re.sub(r"(\d+)\s*パー\s*セント", r"\1%", text)
    text = re.sub(r"(\d+)\s*パーセント", r"\1%", text)
    text = re.sub(r"(\d+)\s*パース", r"\1%", text)
    text = re.sub(r"(\d+)\s*％", r"\1%", text)

    # Generic quote/statistic joining:
    # "A、と。 答えた人は26%。" -> "Aと答えた人は26%でした。"
    report_verbs = r"(?:答えた|回答した|選んだ|支持した|反対した|賛成した|述べた|話した)"
    text = re.sub(r"[、,]\s*(?:と|って)\s*。?\s*(" + report_verbs + r"人は)", r"と\1", text)
    text = re.sub(r"\s*(?:と|って)\s*。?\s*(" + report_verbs + r"人は)", r"と\1", text)

    # Remove spaces between Japanese characters.
    text = re.sub(r"(?<=[\u3040-\u30ff\u3400-\u9fff])\s+(?=[\u3040-\u30ff\u3400-\u9fff])", "", text)

    # Remove awkward spaces around percent and endings.
    text = re.sub(r"(\d+)%\s+(でした|です|となりました|になりました)", r"\1%\2", text)
    text = re.sub(r"(\d+)%\s*。", r"\1%。", text)

    # Generic survey/statistic restoration.
    text = re.sub(
        r"((?:と|って)(?:答えた|回答した|選んだ|支持した|反対した|賛成した)人は\s*\d+(?:\.\d+)?%)。?$",
        r"\1でした。",
        text,
    )

    # Add comma after conjunctive verbs only when the next token looks like a noun/kanji phrase.
    # Avoid corrupting conjugations such as 発表した, 発表しました, 上昇した.
    # text = re.sub(r"(上昇し)(?=[\u3400-\u9fff])", r"\1、", text)
    # text = re.sub(r"(下落し)(?=[\u3400-\u9fff])", r"\1、", text)
    # text = re.sub(r"(発表し)(?=[\u3400-\u9fff])", r"\1、", text)
    # text = re.sub(r"(述べ)(?=[\u3400-\u9fff])", r"\1、", text)
    # text = re.sub(r"(話し)(?=[\u3400-\u9fff])", r"\1、", text)
    # text = re.sub(r"し\s+(?=[\u3040-\u30ff\u3400-\u9fff])", "し、", text)

    # Add period for common completed Japanese endings.
    if text and not text.endswith(("。", "？", "?", "！", "!", ".")):
        if text.endswith((
            "です", "ます", "ました", "ません", "でした", "だった",
            "終えました", "発表しました", "述べました", "%でした", "%です",
        )):
            text += "。"

    return text.strip()


def normalize_korean_for_translation(text: str) -> str:
    text = _squash_spaces(text)
    if text and not text.endswith((".", "?", "!", "。", "？", "！")):
        if text.endswith(("요", "다", "니다", "습니다", "했습니다", "하겠습니다", "입니다")):
            text += "."
    return text.strip()


def normalize_generic_for_translation(text: str) -> str:
    text = _squash_spaces(text)
    if text and not text.endswith((".", "?", "!", "。", "？", "！")):
        text += "."
    return text.strip()


def is_weak_japanese_fragment(text: str) -> bool:
    t = text.strip().replace("。", "").replace(".", "")
    weak = {
        "です", "ます", "でした", "ました", "なります",
        "あります", "ありません", "でしたね",
    }

    if t in weak:
        return True

    if len(t) <= 5 and t.endswith(("です", "ます", "でした", "ました")):
        return True

    return False


def _remove_short_overlap_repetition(text: str) -> str:
    """
    Generic cleanup for overlap-induced repetition.

    It avoids fixed phrase rules. It checks comma-separated adjacent chunks and
    removes a short prefix when the next phrase starts with the same prefix.

    Example:
      来年、来年の夏に...
      -> 来年の夏に...
    """
    text = re.sub(r"\s+", " ", text)

    # Pattern: short Japanese phrase + comma + same phrase followed by more text.
    m = re.match(r"^([\u3040-\u30ff\u3400-\u9fff]{2,8})[、,]\s*(\1[\u3040-\u30ff\u3400-\u9fff].*)$", text)
    if m:
        return m.group(2)

    return text


def _squash_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
