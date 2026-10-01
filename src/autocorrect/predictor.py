"""Load the trained corrector and fix the typos in a piece of text.

The model is the character-level sequence-to-sequence corrector from
Prototype 2 (Ghost-Type-Corrector). It runs here with **NumPy only** - the web
app does not need TensorFlow at all - using beam search plus a dictionary gate:

* a word already in the dictionary is never changed, and
* a proposed correction is only used if it is itself a real word.

This endpoint corrects a whole passage by correcting each word in turn and
putting the spacing and capitalization back.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights.npz")
TOKENIZER_PATH = os.path.join(BASE_DIR, "data", "tokenizer_config.json")
DICTIONARY_PATH = os.path.join(BASE_DIR, "data", "dictionary.txt")

_WORD = re.compile(r"[A-Za-z]+")


class Corrector:
    def __init__(self, weights_path=WEIGHTS_PATH, tokenizer_path=TOKENIZER_PATH,
                 dictionary_path=DICTIONARY_PATH):
        print("INFO: Loading corrector (NumPy, no TensorFlow)...")
        try:
            w = np.load(weights_path)
            self.enc_emb, self.enc_k, self.enc_rk, self.enc_b = (
                w["enc_emb"], w["enc_k"], w["enc_rk"], w["enc_b"])
            self.dec_emb, self.dec_k, self.dec_rk, self.dec_b = (
                w["dec_emb"], w["dec_k"], w["dec_rk"], w["dec_b"])
            self.den_k, self.den_b = w["den_k"], w["den_b"]
            self.U = self.enc_rk.shape[0]

            cfg = json.loads(open(tokenizer_path, encoding="utf-8").read())
            self.c2i = cfg["char_to_index"]
            self.i2c = {int(k): v for k, v in cfg["index_to_char"].items()}
            self.maxlen = cfg["max_seq_length"]
            self.start, self.end, self.pad = (
                cfg["start_token_index"], cfg["end_token_index"], cfg["pad_token_index"])
            self.dictionary = {
                line.strip() for line in open(dictionary_path, encoding="utf-8") if line.strip()}
            self.loaded = True
            print(f"INFO: corrector ready ({len(self.dictionary):,}-word dictionary)")
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the UI
            print(f"CRITICAL ERROR: failed to load the corrector: {exc}")
            self.loaded = False

    # -- neural core (mirrors Ghost-Type-Corrector's inference) ----------------

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _log_softmax(v):
        v = v - v.max()
        e = np.exp(v)
        return np.log(e / e.sum())

    def _step(self, x, h, c, k, rk, b):
        u = self.U
        z = x @ k + h @ rk + b
        i = self._sigmoid(z[:u])
        f = self._sigmoid(z[u:2 * u])
        g = np.tanh(z[2 * u:3 * u])
        o = self._sigmoid(z[3 * u:])
        c2 = f * c + i * g
        return o * np.tanh(c2), c2

    def _encode(self, word):
        h = np.zeros(self.U, dtype=np.float32)
        c = np.zeros(self.U, dtype=np.float32)
        for t in [self.start] + [self.c2i.get(ch, self.pad) for ch in word] + [self.end]:
            h, c = self._step(self.enc_emb[t], h, c, self.enc_k, self.enc_rk, self.enc_b)
        return h, c

    def _candidates(self, word, beam_width=5, max_candidates=8):
        h0, c0 = self._encode(word)
        beams = [(0.0, [self.start], h0, c0)]
        finished = []
        for _ in range(self.maxlen):
            nxt = []
            for score, seq, h, c in beams:
                nh, nc = self._step(self.dec_emb[seq[-1]], h, c, self.dec_k, self.dec_rk, self.dec_b)
                logp = self._log_softmax(nh @ self.den_k + self.den_b)
                for t in np.argsort(logp)[-beam_width:]:
                    t = int(t)
                    if t in (self.end, self.pad):
                        finished.append((score + logp[t], seq))
                    else:
                        nxt.append((score + logp[t], seq + [t], nh, nc))
            if not nxt:
                break
            nxt.sort(key=lambda b: -b[0] / max(len(b[1]), 1))
            beams = nxt[:beam_width]
        pool = finished + [(s, sq) for s, sq, _, _ in beams]
        pool.sort(key=lambda b: -b[0] / max(len(b[1]), 1))
        words, seen = [], set()
        for _, seq in pool:
            txt = "".join(self.i2c.get(t, "") for t in seq if t not in (self.start, self.end, self.pad))
            txt = txt.replace("\t", "").replace("\n", "")
            if txt and txt not in seen:
                seen.add(txt)
                words.append(txt)
            if len(words) >= max_candidates:
                break
        return words

    def correct_word(self, word):
        lower = word.lower()
        # Leave very short words alone: "a" and "i" are real words the
        # dictionary omits (it only keeps words of length >= 2), and correcting
        # one- or two-letter tokens is all risk and no reward.
        if len(lower) < 3 or not lower.isalpha() or lower in self.dictionary:
            return word
        for cand in self._candidates(lower):
            if cand != lower and cand in self.dictionary:
                return _match_case(word, cand)
        return word

    # -- text-level API used by the web route ----------------------------------

    def predict(self, text):
        if not getattr(self, "loaded", False):
            return "Error: the corrector is not loaded."
        return _WORD.sub(lambda m: self.correct_word(m.group(0)), text)


def _match_case(original, corrected):
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


print("INFO: Initializing global Corrector instance...")
corrector_instance = Corrector()
