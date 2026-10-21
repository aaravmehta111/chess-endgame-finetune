# Chess King-and-Pawn Endgame Move Predictor

A fine-tuned Qwen2.5-3B model that predicts the best move in king-and-pawn chess endgames. Stockfish supplies the ground truth, and the question was how much of that skill a small language model can pick up from labeled examples alone.

## Motivation

General-purpose chat models are inconsistent at chess. They lose track of the board and suggest illegal moves. I wanted to see whether fine-tuning a small open model on one narrow slice of the game could close some of that gap, and what it would take.

## Task

Given a position containing only kings and pawns, output the single best move in standard algebraic notation. Positions are generated at random with one to four pawns per side, then checked for legality and to confirm the game is not already over.

## Data

Every position is labeled with Stockfish 16 at depth 18, run through python-chess. I generated 5,500 unique positions and split them into 5,000 for training and 500 held out for testing, with no overlap.

The model sees each position as a plain list of piece squares:

```
White king: e3
Black king: d1
White pawns: b5, e4
Black pawns: a4, c3, f5
White to move.
```

I started with FEN strings, which turned out to be a mistake (details below).

## Training

- Base model: Qwen2.5-3B-Instruct, loaded in 4-bit
- Method: LoRA (rank 32, alpha 32) on all attention and MLP projection layers, about 60M trainable parameters (1.9% of the model)
- Libraries: Unsloth and Hugging Face TRL
- Settings: 8 epochs, learning rate 2e-4, effective batch size 16, max sequence length 512
- Hardware: free-tier Colab T4, about two hours for the final run

## Results

Evaluated with greedy decoding. A prediction counts as correct only if it exactly matches Stockfish's top move.

| Run | Training examples | Epochs | Exact match | Legal moves |
|---|---|---|---|---|
| FEN input | 3,000 | 3 | ~1% | 12% |
| Square-list input | 3,000 | 3 | 36.3% | 84.7% |
| Square-list input | 5,000 | 8 | 39.8% | 86.6% |

The test set was 300 positions for the first two runs and 500 for the last, so the rows are close but not strictly identical comparisons.

Before training anything, I gave 20 king-and-pawn positions to four general-purpose assistants. Copilot got 50%, ChatGPT 40%, and Gemini 35%. DeepSeek's answers were too disorganized to score cleanly. This is a rough check, not a benchmark: it is only 20 positions, those assistants were given FEN rather than the square list, and I don't know which underlying models they were running.

## What went wrong along the way

Three problems took most of the time.

The first LoRA configuration only touched the attention layers at rank 16, which left under 0.3% of parameters trainable. The model returned the same handful of moves regardless of the position. Adding the MLP layers and raising the rank fixed that.

The evaluation code had a bug too. The pad and end-of-sequence tokens were identical, which triggered attention-mask warnings during generation. I set a separate pad token and passed the attention mask explicitly.

The biggest issue was the input format. With training fixed, only 12% of predicted moves were even legal. FEN compresses runs of empty squares into digits, so the model has to do arithmetic to work out which square a piece is on. I tested a plain list of piece squares on a small 280-example run first. Legal moves jumped to 72.5%, which made the full retrain worth doing.

## Limitations

Exact match undercounts, since some positions have several equally good moves and the model gets no credit for finding one that isn't Stockfish's first choice. About one in seven predictions is still illegal. The positions are random rather than taken from real games, and everything here comes from a single training seed and base model.

## Possible next steps

Try a larger base model, add short explanations to the training targets, and score against Stockfish's evaluation instead of exact match.

## Reproducing

The positions behind the reported numbers are in `data/`, so you can train and evaluate without Stockfish.

```
pip install -r requirements.txt
python train.py --data data/train.json --out chess_lora
python evaluate.py --adapter chess_lora --data data/test.json
```

`generate_data.py` builds a fresh dataset (needs Stockfish installed). The original data was not generated with a fixed seed, so a new run gives different positions. `EXPERIMENTS.md` lists every run with its settings. Training needs a CUDA GPU; I used a free Colab T4.
