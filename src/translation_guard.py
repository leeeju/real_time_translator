from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GuardedTranslation:
    handled: bool
    text: str = ""
    reason: str = ""


class TranslationGuard:

    def __init__(
        self,
        enabled: bool = True,
        quote_continuation_enabled: bool = True,
        statistic_pattern_enabled: bool = True,
        preserve_percent: bool = True,
        # Backward compatibility with previous pipeline/config versions.
        survey_pattern_enabled: Optional[bool] = None,
    ):
        self.enabled = enabled
        self.quote_continuation_enabled = quote_continuation_enabled

        # Older realtime_pipeline.py may pass survey_pattern_enabled.
        # Treat it as an alias of statistic_pattern_enabled.
        if survey_pattern_enabled is not None:
            statistic_pattern_enabled = survey_pattern_enabled

        self.statistic_pattern_enabled = statistic_pattern_enabled
        self.preserve_percent = preserve_percent

    def try_translate(
        self,
        source_text: str,
        direction: str,
        translate_fn: Callable[[str, str], str],
    ) -> GuardedTranslation:
        if not self.enabled or direction != "ja2ko":
            return GuardedTranslation(False)

        if self.statistic_pattern_enabled:
            result = self._try_statistic_pattern(source_text, translate_fn)
            if result.handled:
                return result

        return GuardedTranslation(False)

    def _try_statistic_pattern(
        self,
        source_text: str,
        translate_fn: Callable[[str, str], str],
    ) -> GuardedTranslation:
        text = source_text.strip()

        report_verbs = r"(?:答えた|回答した|選んだ|支持した|反対した|賛成した)"
        m = re.search(
            r"(.+?)(?:と|って)(" + report_verbs + r")人は\s*(\d+(?:\.\d+)?)\s*%(?:でした|です|となりました|になりました|。)?",
            text,
        )

        if not m:
            return GuardedTranslation(False)

        option_ja = m.group(1).strip(" 「」『』、。")
        report_verb = m.group(2)
        percent = m.group(3)

        if len(option_ja) < 2:
            return GuardedTranslation(False)

        option_ko = self._translate_option_phrase(option_ja, translate_fn)
        report_ko = self._report_verb_to_korean(report_verb)

        ko = f"“{option_ko}”라고 {report_ko} 사람은 {percent}%였습니다."
        return GuardedTranslation(True, ko, "generic_statistic_pattern")

    @staticmethod
    def _translate_option_phrase(option_ja: str, translate_fn: Callable[[str, str], str]) -> str:
        """
        Translate the quoted answer option using NLLB, then lightly adapt it
        for embedded quotation form.

        No fixed test sentence is hard-coded here.
        """
        option_ko = translate_fn(option_ja, "ja2ko").strip()
        option_ko = option_ko.rstrip(".。")

        replacements = {
            "해야 합니다": "하는 것이 좋다",
            "좋습니다": "좋다",
            "입니다": "이다",
            "입니다.": "이다",
        }

        for src, dst in replacements.items():
            option_ko = option_ko.replace(src, dst)

        return option_ko

    @staticmethod
    def _report_verb_to_korean(report_verb: str) -> str:
        if report_verb in ("答えた", "回答した"):
            return "답한"
        if report_verb == "選んだ":
            return "선택한"
        if report_verb == "支持した":
            return "지지한"
        if report_verb == "反対した":
            return "반대한"
        if report_verb == "賛成した":
            return "찬성한"
        return "응답한"
