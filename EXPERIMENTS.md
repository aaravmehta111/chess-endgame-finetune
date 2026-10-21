# Experiment log

Runs in the order they happened. Base model for all runs: Qwen2.5-3B-Instruct (4-bit), trained with Unsloth and TRL on a Colab T4.

## Baseline: general-purpose assistants

Zero-shot, no fine-tuning, 20 king-and-pawn positions given as FEN, scored against Stockfish depth 18.

| Assistant | Exact match |
|---|---|
| Copilot | 50% |
| ChatGPT | 40% |
| Gemini | 35% |
| DeepSeek | not cleanly scorable (about 27% with best-effort matching) |

An earlier version of this check used 20 random middlegame positions, before I narrowed the scope to king-and-pawn endgames. Scores there were Copilot 30%, ChatGPT 25%, Gemini 15%, DeepSeek 5%.

## Fine-tuning runs

| # | Input format | LoRA | Train examples | Epochs | Test set | Exact match | Legal |
|---|---|---|---|---|---|---|---|
| 1 | FEN | r=16, attention only | 3,000 | 3 | 300 | 0.7% (1.3% before eval fix) | not measured |
| 2 | FEN | r=32, attention + MLP | 3,000 | 3 | 300 | 1.0% (3/300) | 12.0% (36/300) |
| 3 | Square list | r=32, attention + MLP | 280 | 5 | 40 | 12.5% (5/40) | 72.5% (29/40) |
| 4 | Square list | r=32, attention + MLP | 3,000 | 3 | 300 | 36.3% (109/300) | 84.7% (254/300) |
| 5 | Square list | r=32, attention + MLP | 5,000 | 8 | 500 | 39.8% (199/500) | 86.6% (433/500) |

Run 3 was a small diagnostic with 1 to 3 pawns per side. It was meant to isolate the input format, not to be compared directly with the others.

## Problems found

1. **LoRA only on attention layers (run 1).** Under 0.3% of parameters were trainable and the model gave the same few moves for different positions. Fix: add the MLP projections and raise the rank to 32 (about 1.9% trainable).
2. **Pad token equal to the end-of-sequence token.** This produced attention-mask warnings and unreliable generation. Fix: use a separate pad token and pass the attention mask explicitly.
3. **FEN input.** With training fixed, only 12% of predicted moves were legal. FEN packs runs of empty squares into digits, which forces the model to count to find a piece's square. Listing each piece's square explicitly (run 3) raised the legal-move rate to 72.5%, and the full retrain (run 4) confirmed it.

## Not completed

I tried a Qwen2.5-7B run with the same data. It was cut off by Colab disconnects before finishing, so there are no results to report.

## Notes

- The reported dataset was generated without a fixed seed, so it is committed in `data/`. Regenerating gives a different set of positions with the same distribution.
- Exact match undercounts: some positions have several equally good moves.
- Results are from a single training seed.
