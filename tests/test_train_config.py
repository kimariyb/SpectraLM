import unittest

from src.train import build_arg_parser


class TrainConfigTests(unittest.TestCase):
    def test_default_train_config_uses_pilot_subset(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.train_dataset, "src/data/subsets/spectralm_500_100_pilot/train.pkl")
        self.assertEqual(args.eval_dataset, "src/data/subsets/spectralm_500_100_pilot/test.pkl")
        self.assertEqual(args.output_dir, "outputs/spectralm-pilot-qwen3-vl-8b")

    def test_training_script_accepts_remote_model_path(self):
        args = build_arg_parser().parse_args(["--model-path", "/models/qwen"])

        self.assertEqual(args.model_path, "/models/qwen")


if __name__ == "__main__":
    unittest.main()
