# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-10-01

Make the web app actually correct typos. The idea is unchanged — a Flask GUI
serving the Phase-2 Seq2Seq corrector — but the model it served was the wrong
kind, and it needed TensorFlow to run.

### Fixed
- **Wrong model for the task.** The committed model was a word-level
  bidirectional GRU over a whole-corpus vocabulary (a 98 MB model with a 179 MB
  tokenizer). Word-level tokenization can't fix a misspelled word — it's just an
  unknown token — so it only "corrected" typos it had memorized. Typo correction
  is a character-level problem, so the app now serves the **character-level
  seq2seq model from Phase 2 (Ghost-Type-Corrector)**, which is what the project
  always said it integrated.
- **Needed TensorFlow to serve.** Inference now runs with **NumPy only**
  (`predictor.py`), so the server is ~3 MB of weights and starts instantly.

### Added
- A **dictionary gate** (never change a known word; leave "a"/"i" alone; only
  output real words) and **beam search**, so correcting a whole passage is safe
  and accurate.
- Tests for the corrector and the Flask routes (`/` and `/predict`), and CI
  (ruff + pytest on Python 3.10-3.12).
- A shareable demo link: `/?demo` fills in a sample and corrects it.

### Removed
- The 98 MB word-level model, the 179 MB tokenizer, the hundreds of MB of
  training corpus (kept out of HEAD), the word-level training scripts, and the
  Conda environment files. The model is trained in Phase 2.

## [1.0.0] - 2025-10
- Initial version: a Flask web GUI and a word-level Seq2Seq (bidirectional GRU)
  model, intended to demo typo correction.
