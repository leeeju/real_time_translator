from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


LANG_MAP = {
    "ja2ko": {
        "asr_lang": "ja",
        "nllb_src": "jpn_Jpan",
        "nllb_tgt": "kor_Hang",
        "src_label": "JA",
        "tgt_label": "KO",
    },
    "ko2ja": {
        "asr_lang": "ko",
        "nllb_src": "kor_Hang",
        "nllb_tgt": "jpn_Jpan",
        "src_label": "KO",
        "tgt_label": "JA",
    },
}


@dataclass
class TranslationResult:
    text: str
    elapsed: float


class NLLBTranslator:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        cache_dir: str | Path = "models",
        device: str = "cuda",
        dtype: str = "fp16",
        max_new_tokens: int = 64,
        num_beams: int = 2,
        repetition_penalty: float = 1.18,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 0.95,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = self._select_device(device)
        self.torch_dtype = self._select_dtype(self.device, dtype)

        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty

        print("=== Load NLLB Translator ===")
        print(f"model                 : {self.model_name}")
        print(f"device                : {self.device}")
        print(f"dtype                 : {self.torch_dtype}")
        print(f"cache dir             : {self.cache_dir}")
        print(f"max_new_tokens        : {self.max_new_tokens}")
        print(f"num_beams             : {self.num_beams}")
        print(f"repetition_penalty    : {self.repetition_penalty}")
        print(f"no_repeat_ngram_size  : {self.no_repeat_ngram_size}")
        print()

        self.tokenizers = {}

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def translate(self, text: str, direction: str = "ja2ko") -> TranslationResult:
        if direction not in LANG_MAP:
            raise ValueError(f"Unsupported direction: {direction}")

        src_lang = LANG_MAP[direction]["nllb_src"]
        tgt_lang = LANG_MAP[direction]["nllb_tgt"]

        tokenizer = self._get_tokenizer(src_lang)
        tokenizer.src_lang = src_lang

        t0 = time.perf_counter()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=192,
        ).to(self.device)

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        output_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=False,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            length_penalty=self.length_penalty,
        )

        translated = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - t0

        return TranslationResult(text=translated, elapsed=elapsed)

    def _get_tokenizer(self, src_lang: str):
        if src_lang not in self.tokenizers:
            self.tokenizers[src_lang] = AutoTokenizer.from_pretrained(
                self.model_name,
                src_lang=src_lang,
                cache_dir=str(self.cache_dir),
            )
        return self.tokenizers[src_lang]

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if requested == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA unavailable. Falling back to CPU.")
            return "cpu"

        return requested

    @staticmethod
    def _select_dtype(device: str, requested_dtype: str):
        if requested_dtype == "auto":
            return torch.float16 if device == "cuda" else torch.float32

        if requested_dtype == "fp16":
            return torch.float16

        return torch.float32
