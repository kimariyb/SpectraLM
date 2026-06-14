import argparse
import os


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune SpectraLM with Qwen-VL and LoRA/QLoRA.")
    parser.add_argument(
        "--model-path",
        default="/mnt/data/kimariyb/models/Qwen3-VL-8B-Instruct",
        help="Local or Hugging Face path to the base vision-language model.",
    )
    parser.add_argument(
        "--train-dataset",
        default="src/data/subsets/spectralm_500_100_pilot/train.pkl",
        help="Pickle dataset for training.",
    )
    parser.add_argument(
        "--eval-dataset",
        default="src/data/subsets/spectralm_500_100_pilot/test.pkl",
        help="Pickle dataset for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/spectralm-pilot-qwen3-vl-8b",
        help="Directory for checkpoints and adapter outputs.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def configure_huggingface_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_DOWNLOAD_RETRY", "20")


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_huggingface_env()

    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    try:
        from .dataset import NMRexpDataset
    except ImportError:
        from dataset import NMRexpDataset

    print(f"Loading base model: {args.model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        use_gradient_checkpointing="unsloth",
    )

    print("Configuring LoRA adapter...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
    )

    train_dataset = NMRexpDataset(args.train_dataset, task_probs={"structure_reasoning": 1.0})
    eval_dataset = NMRexpDataset(args.eval_dataset, task_probs={"structure_reasoning": 1.0})

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        bf16=True,
        remove_unused_columns=False,
        dataset_text_field="",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
