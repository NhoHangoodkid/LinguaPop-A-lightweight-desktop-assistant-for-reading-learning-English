"""
Translation engine: load HuggingFace tokenizer + CTranslate2 model ONCE,
then each translation only calls translate() -> fast.

Design does not hard-code EN->VI: each engine is a pair (model, tokenizer).
To add VI->EN later = create another TranslationEngine, without modifying logic.

Length processing: engine translates each sentence independently via translate_batch (parallel batch).
When receiving str, engine automatically splits sentences using regex to ensure Marian always receives single sentences.
Capture can pass a complete list of sentences or a str.
"""
import logging
import re

import ctranslate2
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Sentence splitting regex: ends with . ! ? (optionally followed by closing quotes/brackets), followed by
# whitespace or end of string. Keeps the punctuation at the end of each sentence (lookbehind).
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+')


class TranslationEngine:
    def __init__(self, model_dir: str, tokenizer_dir: str, device: str = "cpu", max_input_length: int = 128, num_beams: int = 4, num_hypotheses: int = 1, max_decode_length: int = 128, repetition_penalty: float = 1.2, no_repeat_ngram_size: int = 3, length_penalty: float = 1.0, inter_threads: int = 1):
        # Load once at app startup
        self.translator = ctranslate2.Translator(
            model_dir, device=device, inter_threads=inter_threads,
            compute_type="float32")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, clean_up_tokenization_spaces=True)
        self.max_input_length = max_input_length
        self.num_beams = num_beams
        self.num_hypotheses = num_hypotheses
        self.max_decode_length = max_decode_length
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
    
    # Internal
    def _split_sentences(self, text: str) -> list[str]:
        """
        Safety net: when receiving str, split into single sentences list before translating.
        Marian is trained on single sentences — passing a whole paragraph will cause truncation.
        """
        text = text.strip()
        if not text:
            return []
        parts = _SENT_BOUNDARY.split(text)
        return [p.strip() for p in parts if p and p.strip()]

    def _src_tokens(self, text: str):
        """SOURCE tokens (list[str]) to feed into the model."""
        enc = self.tokenizer(text, truncation=True, max_length=self.max_input_length)
        
        full_len = self._n_src_tokens(text)
        if full_len > self.max_input_length:
            logger.warning("Input %d tokens > %d, trailing part truncated.", 
                           full_len, self.max_input_length)
                           
        return self.tokenizer.convert_ids_to_tokens(enc["input_ids"])

    def _n_src_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def _decode(self, tokens) -> str:
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def _strip_eos(self, tokens):
        """Remove </s> at the end if present — DO NOT force the model to stop midway when used as prefix."""
        eos = self.tokenizer.eos_token
        out = list(tokens)
        while out and out[-1] == eos:
            out.pop()
        return out

    def _run(self, src_tokens, prefix=None, decode_len=None):
        """Call CT2 for 1 sentence. Returns the ORIGINAL TOKEN sequence of the translation (undecoded)."""
        kwargs = dict(
            beam_size=self.num_beams,
            max_decoding_length=decode_len or self.max_decode_length,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            length_penalty=self.length_penalty,
        )
        if prefix is not None:
            res = self.translator.translate_batch(
                [src_tokens], target_prefix=[prefix], **kwargs)
        else:
            res = self.translator.translate_batch([src_tokens], **kwargs)
        return res[0].hypotheses[0]

    def translate_with_context(self, sentences) -> list:
        """
        Sliding window of 2 sentences using target_prefix.
        Translates sentence N in the context of sentence N-1; returns a list of translations for EACH sentence (in order).
        """
        sentences = [s.strip() for s in sentences if s and s.strip()]
        if not sentences:
            return []

        out_texts = []
        prev_tokens = None

        for i, sent in enumerate(sentences):
            # --- First sentence, or no prefix -> translate independently ---
            if prev_tokens is None:
                hyp = self._run(self._src_tokens(sent))
                out_texts.append(self._decode(hyp))
                prev_tokens = hyp
                continue

            pair_text = sentences[i - 1] + " " + sent

            # --- Check exception: source token budget. If exceeded -> fallback to independent translation ---
            if self._n_src_tokens(pair_text) > self.max_input_length:
                hyp = self._run(self._src_tokens(sent))
                out_texts.append(self._decode(hyp))
                prev_tokens = hyp
                continue

            # --- Translate pair [S_{n-1}+S_n], forcing prefix = T_{n-1} ---
            prefix = self._strip_eos(prev_tokens)
            decode_len = len(prefix) + self.max_decode_length
            full = self._run(self._src_tokens(pair_text),
                             prefix=prefix, decode_len=decode_len)

            # --- Handle token overlap: slice off the prefix tokens at the beginning of the generated result ---
            t_n_tokens = full[len(prefix):]
            t_n = self._decode(t_n_tokens)

            # --- Fallback: model stopped early, didn't generate more -> translate independently ---
            if not t_n:
                hyp = self._run(self._src_tokens(sent))
                t_n = self._decode(hyp)
                prev_tokens = hyp
            else:
                prev_tokens = t_n_tokens

            out_texts.append(t_n)

        return out_texts
    
    # Public API
    def translate(self, sentences) -> list:
        """
        Accepts a string (str) or list[dict] from capture.split_sentences.
        Returns list[str] containing a single element (a complete paragraph keeping the original context).
        """
        if isinstance(sentences, str):
            # Fallback: import split_sentences from capture if a str is passed
            try:
                from capture import split_sentences
                sentences = split_sentences(sentences)
            except ImportError:
                sentences = [{"text": sentences, "delimiter": ""}]
                
        # Normalize to list[dict] for backward compatibility (if any)
        items = []
        for s in sentences:
            if isinstance(s, str):
                items.append({"text": s, "delimiter": " "})
            else:
                items.append(s)

        if not items:
            return []

        # Filter sentences that actually have content to feed into the model
        texts_to_translate = [item["text"] for item in items if item["text"].strip()]
        
        if texts_to_translate:
            translations = self.translate_with_context(texts_to_translate)
        else:
            translations = []

        # Assemble results with original context
        final_parts = []
        t_idx = 0
        for item in items:
            if item["text"].strip():
                final_parts.append(translations[t_idx])
                t_idx += 1
            else:
                # Empty sentence, only exists due to regex or extra whitespace, keep it as is
                final_parts.append(item["text"])
            final_parts.append(item["delimiter"])

        return ["".join(final_parts).strip()]


class EngineManager:
    """Manages multiple engines, selected by language pair. Currently only en-vi is supported."""

    def __init__(self):
        self._engines = {}
        self._default = None

    def register(self, name: str, engine: TranslationEngine, is_default: bool = False):
        self._engines[name] = engine
        if is_default or self._default is None:
            self._default = name

    def translate(self, sentences, engine_name: str = None) -> list:
        name = engine_name or self._default
        if name not in self._engines:
            raise KeyError(f"Engine '{name}' not found")
        return self._engines[name].translate(sentences)