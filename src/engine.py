"""
Engine dich: load model CTranslate2 + tokenizer HuggingFace MOT LAN,
sau do moi lan dich chi goi translate() -> nhanh.

Thiet ke khong hard-code EN->VI: moi engine la mot cap (model, tokenizer).
Them VI->EN sau nay = tao them mot TranslationEngine khac, khong sua logic.

Xu ly do dai: engine dich tung cau doc lap qua translate_batch (batch song song).
Khi nhan str, engine tu tach cau bang regex de dam bao Marian luon nhan cau don.
Capture co the dua list cau hoan chinh hoac str deu duoc.
"""
import logging
import re

import ctranslate2
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Regex tach cau: ket thuc bang . ! ? (co the kem dau dong/ngoac), theo sau la
# khoang trang hoac cuoi chuoi.  Giu dau cau o cuoi moi cau (lookbehind).
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+')


class TranslationEngine:
    def __init__(self, model_dir: str, tokenizer_dir: str,
                 device: str = "cpu", max_input_length: int = 128,
                 num_beams: int = 4, num_hypotheses: int = 1,
                 max_decode_length: int = 128,
                 repetition_penalty: float = 1.2,
                 no_repeat_ngram_size: int = 3,
                 length_penalty: float = 1.0,
                 inter_threads: int = 1):
        # Load 1 lan luc khoi dong app
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """
        Luoi an toan: khi nhan str, tach thanh list cau don truoc khi dich.
        Marian duoc train tren cau don — dua nguyen doan vao se bi dich cut.
        """
        text = text.strip()
        if not text:
            return []
        parts = _SENT_BOUNDARY.split(text)
        return [p.strip() for p in parts if p and p.strip()]

    def _src_tokens(self, text: str):
        """Token NGUON (list[str]) de dua vao model."""
        enc = self.tokenizer(text, truncation=True, max_length=self.max_input_length)
        
        full_len = self._n_src_tokens(text)
        if full_len > self.max_input_length:
            logger.warning("Dau vao %d token > %d, bi cat bot phan duoi.", 
                           full_len, self.max_input_length)
                           
        return self.tokenizer.convert_ids_to_tokens(enc["input_ids"])

    def _n_src_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def _decode(self, tokens) -> str:
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def _strip_eos(self, tokens):
        """Bo </s> o cuoi neu co — KHONG ep model dung giua chung khi lam prefix."""
        eos = self.tokenizer.eos_token
        out = list(tokens)
        while out and out[-1] == eos:
            out.pop()
        return out

    def _run(self, src_tokens, prefix=None, decode_len=None):
        """Goi CT2 1 cau. Tra ve dãy TOKEN GOC cua ban dich (chua decode)."""
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
        Cua so truot 2 cau bang target_prefix.
        Cau N dich trong ngu canh cau N-1; tra ve list ban dich TUNG cau (cung thu tu).
        """
        sentences = [s.strip() for s in sentences if s and s.strip()]
        if not sentences:
            return []

        out_texts = []
        prev_tokens = None

        for i, sent in enumerate(sentences):
            # --- Cau dau, hoac khong co prefix -> dich co lap ---
            if prev_tokens is None:
                hyp = self._run(self._src_tokens(sent))
                out_texts.append(self._decode(hyp))
                prev_tokens = hyp
                continue

            pair_text = sentences[i - 1] + " " + sent

            # --- BAY 3: canh ngan sach token nguon. Vuot gioi han -> lui ve co lap ---
            if self._n_src_tokens(pair_text) > self.max_input_length:
                hyp = self._run(self._src_tokens(sent))
                out_texts.append(self._decode(hyp))
                prev_tokens = hyp
                continue

            # --- Dich cap [S_{n-1}+S_n], ep prefix = T_{n-1} ---
            prefix = self._strip_eos(prev_tokens)
            decode_len = len(prefix) + self.max_decode_length
            full = self._run(self._src_tokens(pair_text),
                             prefix=prefix, decode_len=decode_len)

            # --- BAY token-level: cat dung SO TOKEN cua prefix o dau ---
            t_n_tokens = full[len(prefix):]
            t_n = self._decode(t_n_tokens)

            # --- Fallback: model dung som, khong sinh them -> dich co lap ---
            if not t_n:
                hyp = self._run(self._src_tokens(sent))
                t_n = self._decode(hyp)
                prev_tokens = hyp
            else:
                prev_tokens = t_n_tokens

            out_texts.append(t_n)

        return out_texts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, sentences) -> list:
        """
        Nhan 1 chuoi (str) hoac list[dict] tu capture.split_sentences.
        Tra ve list[str] gom 1 phan tu duy nhat (doan van hoan chinh giu nguyen context).
        """
        if isinstance(sentences, str):
            # Fallback: import split_sentences tu capture neu duoc truyen str
            try:
                from capture import split_sentences
                sentences = split_sentences(sentences)
            except ImportError:
                sentences = [{"text": sentences, "delimiter": ""}]
                
        # Dong nhat ve dang list[dict] de tuong thich code cu (neu co)
        items = []
        for s in sentences:
            if isinstance(s, str):
                items.append({"text": s, "delimiter": " "})
            else:
                items.append(s)

        if not items:
            return []

        # Loc cac cau thuc su co noi dung de nap vao model
        texts_to_translate = [item["text"] for item in items if item["text"].strip()]
        
        if texts_to_translate:
            translations = self.translate_with_context(texts_to_translate)
        else:
            translations = []

        # Ghep ket qua voi context goc
        final_parts = []
        t_idx = 0
        for item in items:
            if item["text"].strip():
                final_parts.append(translations[t_idx])
                t_idx += 1
            else:
                # Cau rong, chi xuat hien do regex hoac khoang trang du, giu nguyen
                final_parts.append(item["text"])
            final_parts.append(item["delimiter"])

        return ["".join(final_parts).strip()]


class EngineManager:
    """Quan ly nhieu engine, chon theo cap ngon ngu. Hien tai chi co en-vi."""

    def __init__(self):
        self._engines = {}
        self._default = None

    def register(self, name: str, engine: TranslationEngine,
                 is_default: bool = False):
        self._engines[name] = engine
        if is_default or self._default is None:
            self._default = name

    def translate(self, sentences, engine_name: str = None) -> list:
        name = engine_name or self._default
        if name not in self._engines:
            raise KeyError(f"Chua co engine '{name}'")
        return self._engines[name].translate(sentences)