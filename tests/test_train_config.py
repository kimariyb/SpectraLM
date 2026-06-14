import unittest

from src.train import build_arg_parser, build_sft_config_kwargs, build_sft_trainer_kwargs


class FakeNewSFTConfig:
    def __init__(self, output_dir, max_length, eval_strategy, bf16):
        pass


class FakeOldSFTConfig:
    def __init__(self, output_dir, max_seq_length, evaluation_strategy, bf16):
        pass


class FakeNewSFTTrainer:
    def __init__(self, model, args, data_collator, train_dataset, eval_dataset, processing_class):
        pass


class FakeOldSFTTrainer:
    def __init__(self, model, args, data_collator, train_dataset, eval_dataset, tokenizer):
        pass


class TrainConfigTests(unittest.TestCase):
    def test_default_train_config_uses_pilot_subset(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.train_dataset, "src/data/subsets/spectralm_500_100_pilot/train.pkl")
        self.assertEqual(args.eval_dataset, "src/data/subsets/spectralm_500_100_pilot/test.pkl")
        self.assertEqual(args.output_dir, "outputs/spectralm-pilot-qwen3-vl-8b")

    def test_training_script_accepts_remote_model_path(self):
        args = build_arg_parser().parse_args(["--model-path", "/models/qwen"])

        self.assertEqual(args.model_path, "/models/qwen")

    def test_sft_config_kwargs_use_new_trl_length_name(self):
        args = build_arg_parser().parse_args([])
        kwargs = build_sft_config_kwargs(args, FakeNewSFTConfig)

        self.assertEqual(kwargs["max_length"], 2048)
        self.assertEqual(kwargs["eval_strategy"], "steps")
        self.assertNotIn("max_seq_length", kwargs)
        self.assertNotIn("evaluation_strategy", kwargs)

    def test_sft_config_kwargs_use_old_trl_length_name(self):
        args = build_arg_parser().parse_args([])
        kwargs = build_sft_config_kwargs(args, FakeOldSFTConfig)

        self.assertEqual(kwargs["max_seq_length"], 2048)
        self.assertEqual(kwargs["evaluation_strategy"], "steps")
        self.assertNotIn("max_length", kwargs)
        self.assertNotIn("eval_strategy", kwargs)

    def test_sft_trainer_kwargs_use_processing_class_when_supported(self):
        kwargs = build_sft_trainer_kwargs(
            FakeNewSFTTrainer,
            model="model",
            tokenizer="tokenizer",
            data_collator="collator",
            train_dataset="train",
            eval_dataset="eval",
            training_args="args",
        )

        self.assertEqual(kwargs["processing_class"], "tokenizer")
        self.assertNotIn("tokenizer", kwargs)

    def test_sft_trainer_kwargs_use_tokenizer_for_old_versions(self):
        kwargs = build_sft_trainer_kwargs(
            FakeOldSFTTrainer,
            model="model",
            tokenizer="tokenizer",
            data_collator="collator",
            train_dataset="train",
            eval_dataset="eval",
            training_args="args",
        )

        self.assertEqual(kwargs["tokenizer"], "tokenizer")
        self.assertNotIn("processing_class", kwargs)


if __name__ == "__main__":
    unittest.main()
