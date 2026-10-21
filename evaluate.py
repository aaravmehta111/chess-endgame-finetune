"""Evaluate a trained adapter on held-out positions.

Reports exact-match accuracy against Stockfish's top move (check/mate symbols ignored)
and the share of predictions that are legal moves in the given position.
"""
import argparse
import json

import chess
from tqdm import tqdm
from unsloth import FastLanguageModel


def strip(move):
    return move.replace("+", "").replace("#", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="chess_lora")
    parser.add_argument("--data", default="data/test.json")
    parser.add_argument("--out", default="results/eval_results.json")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "<|fim_pad|>"
    FastLanguageModel.for_inference(model)

    with open(args.data) as f:
        test_data = json.load(f)
    if args.limit:
        test_data = test_data[:args.limit]

    correct = legal = 0
    results = []
    for ex in tqdm(test_data):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["instruction"]}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to("cuda")
        attention_mask = (inputs != tokenizer.pad_token_id).long()
        outputs = model.generate(
            input_ids=inputs, attention_mask=attention_mask,
            max_new_tokens=10, use_cache=True, do_sample=False,
        )
        text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        predicted = text.split()[0] if text.split() else ""

        is_correct = strip(predicted) == strip(ex["response"])
        try:
            chess.Board(ex["fen"]).parse_san(predicted)
            is_legal = True
        except ValueError:
            is_legal = False

        correct += is_correct
        legal += is_legal
        results.append({"fen": ex["fen"], "expected": ex["response"],
                        "predicted": predicted, "correct": is_correct, "legal": is_legal})

    n = len(test_data)
    print(f"Exact match: {correct}/{n} ({correct / n:.1%})")
    print(f"Legal moves: {legal}/{n} ({legal / n:.1%})")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
