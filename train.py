"""LoRA fine-tune Qwen2.5-3B-Instruct on the chess endgame data.

Settings match the run behind the reported results (5,000 examples, 8 epochs, LoRA r=32
on all attention and MLP projections). Developed on a free Colab T4; needs a CUDA GPU.
"""
import argparse
import json

from unsloth import FastLanguageModel  # import before trl/transformers
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/train.json")
    parser.add_argument("--out", default="chess_lora")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Distinct pad token: pad == eos caused attention-mask problems at generation time.
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "<|fim_pad|>"

    with open(args.data) as f:
        examples = json.load(f)

    def to_text(ex):
        messages = [
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = Dataset.from_list([to_text(ex) for ex in examples])
    print(f"Loaded {len(dataset)} examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=SFTConfig(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_steps=15,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
        ),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved adapter to {args.out}/")


if __name__ == "__main__":
    main()
