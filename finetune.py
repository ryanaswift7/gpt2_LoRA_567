#!/usr/bin/env python3

from gpt2_lora_e2e import GPT2_LoRA_E2E
from utils import use_single_cuda_device, enforce_reproducibility
import argparse

if __name__=="__main__":

    enforce_reproducibility(seed=44)
    use_single_cuda_device()

    parser = argparse.ArgumentParser(description="A script to finetune GPT2 Small on the E2E NLG Challenge dataset using LoRA")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "improved"],
        default="baseline",
        help="Select either the baseline model or the improved model",
        )
    parser.add_argument(
        "--no_gen_test_data",
        action='store_false',
        dest='gen_test_data',
        help="Generate test input and reference files for the E2E NLG Challenge test data. Only set to False if you have already generated the file using this script.",
    )
    args = parser.parse_args()
    model = GPT2_LoRA_E2E(args.model, args.gen_test_data)
    model.init_data()
    model.train()
    model.generate()