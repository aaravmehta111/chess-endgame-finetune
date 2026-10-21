"""Generate labeled king-and-pawn endgame positions with Stockfish as ground truth.

Writes data/train.json and data/test.json (instruction/response pairs plus the FEN,
which the evaluation script uses to check move legality).

Note: the dataset used for the reported results was generated without a fixed seed,
so it is committed under data/. Re-running this script produces a different set of
positions with the same distribution.
"""
import argparse
import json
import os
import random

import chess
import chess.engine

INSTRUCTION_TEMPLATE = (
    "You are given a chess position (king and pawn endgame only). "
    "Here are the exact squares of every piece:\n\n"
    "{piece_list}\n"
    "{turn} to move.\n\n"
    "What is the single best move in standard algebraic notation?"
)


def random_position(min_pawns=1, max_pawns=4):
    """Random legal king-and-pawn position that is not already game over."""
    while True:
        board = chess.Board(None)
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        squares = list(chess.SQUARES)
        while True:
            wk, bk = random.choice(squares), random.choice(squares)
            if wk != bk and chess.square_distance(wk, bk) > 1:
                break
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        pawn_squares = [s for s in squares
                        if chess.square_rank(s) not in (0, 7) and s not in (wk, bk)]
        random.shuffle(pawn_squares)
        n_white = random.randint(min_pawns, max_pawns)
        n_black = random.randint(min_pawns, max_pawns)
        if len(pawn_squares) < n_white + n_black:
            continue
        for sq in pawn_squares[:n_white]:
            board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
        for sq in pawn_squares[n_white:n_white + n_black]:
            board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))

        if board.is_valid() and not board.is_game_over():
            return board


def piece_list(board):
    def names(color):
        pawns = sorted(chess.square_name(s) for s in board.pieces(chess.PAWN, color))
        return ", ".join(pawns) if pawns else "none"

    return "\n".join([
        f"White king: {chess.square_name(board.king(chess.WHITE))}",
        f"Black king: {chess.square_name(board.king(chess.BLACK))}",
        f"White pawns: {names(chess.WHITE)}",
        f"Black pawns: {names(chess.BLACK)}",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish"))
    args = parser.parse_args()

    random.seed(args.seed)
    total = args.n_train + args.n_test
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    seen, examples = set(), []
    while len(examples) < total:
        board = random_position()
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        move = engine.play(board, chess.engine.Limit(depth=args.depth)).move
        turn = "White" if board.turn == chess.WHITE else "Black"
        examples.append({
            "fen": fen,
            "turn": turn,
            "instruction": INSTRUCTION_TEMPLATE.format(piece_list=piece_list(board), turn=turn),
            "response": board.san(move),
        })
        if len(examples) % 500 == 0:
            print(f"{len(examples)}/{total}")
    engine.quit()

    random.shuffle(examples)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.json"), "w") as f:
        json.dump(examples[:args.n_train], f, indent=2)
    with open(os.path.join(args.out_dir, "test.json"), "w") as f:
        json.dump(examples[args.n_train:], f, indent=2)
    print(f"Wrote {args.n_train} train and {args.n_test} test examples to {args.out_dir}/")


if __name__ == "__main__":
    main()
