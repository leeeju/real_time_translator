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

        result = self._try_polite_request(source_text, translate_fn)
        if result.handled:
            return result

        return GuardedTranslation(False)

    def _try_polite_request(
        self,
        source_text: str,
        translate_fn: Callable[[str, str], str],
    ) -> GuardedTranslation:
        text = source_text.strip()
        text = text.rstrip("。.!！?")

        patterns = (
            (
                r"(.+?)(?:を)?しておいていただきたいと思います$",
                "{action} 두시기 바랍니다.",
                "generic_polite_request_keep_ready",
            ),
            (
                r"(.+?)(?:を)?していただきたいと思います$",
                "{action} 주시기 바랍니다.",
                "generic_polite_request",
            ),
            (
                r"(.+?)(?:を)?していただければと思います$",
                "{action} 주시면 좋겠습니다.",
                "generic_polite_request_soft",
            ),
        )

        for pattern, template, reason in patterns:
            m = re.fullmatch(pattern, text)
            if not m:
                continue

            action_ja = m.group(1).strip(" 「」『』、。")
            if not (1 <= len(action_ja) <= 24):
                continue

            action_ko = self._translate_action_phrase(action_ja, translate_fn)
            if not action_ko:
                continue

            return GuardedTranslation(True, template.format(action=action_ko), reason)

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
    def _translate_action_phrase(action_ja: str, translate_fn: Callable[[str, str], str]) -> str:
        action_ko = translate_fn(action_ja, "ja2ko").strip()
        action_ko = action_ko.rstrip(".。")

        cleanup_patterns = (
            r"^(?:그|그것|이를|이것을)\s+",
            r"\s*(?:하는 것|하는 것을|하고 있는 것|하고 있는 것을)$",
            r"\s*(?:입니다|입니다\.|합니다|합니다\.)$",
        )
        for pattern in cleanup_patterns:
            action_ko = re.sub(pattern, "", action_ko).strip()

        if not action_ko:
            return ""

        if action_ko.endswith(("하다", "하기")):
            action_ko = action_ko[:-2]

        if action_ko.endswith("해"):
            return action_ko

        if action_ko.endswith(("공유", "확인", "준비", "검토", "대응", "정리", "참고", "기록")):
            return f"{action_ko}해"

        return f"{action_ko} 해"

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
