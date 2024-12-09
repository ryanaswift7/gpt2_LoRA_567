# Finetuning GPT2 Small for the E2E NLG Challenge Using LoRA
I've made my own implementation of the LoRA framework and used it to 
reconstruct the original implementation from the [LoRA paper](https://arxiv.org/abs/2106.09685).
Note that I've used GPT2 Small instead of GPT2 Medium, due to resource constraints.

## Usage
`finetune.py` is the main entry point, and the two available arguments are selecting either the baseline (default) or improved LoRA configurations, and optionally generating the files for testing (`test_mrs.txt` and `test_refs.txt`). 

First, clone the repo.

Then, create the conda environment with `conda env create -f environment.yaml` (of course, you have to have [conda](https://docs.conda.io/projects/conda/en/stable/) installed first).

Now you're ready to run it!

#### Example 1
This will finetune using the baseline configuration and generate the necessary test files.
```
./finetune.py --model=baseline
```

#### Example 2
This will finetune using the improved configuration and will *not* generate the test files.
```
./finetune.py --model=improved --no_gen_test_data
```


## Results

Listed Below are the results collected using the [official E2E NLG Challenge metrics script](https://github.com/tuetschek/e2e-metrics).

#### Baseline
Parameter Count: 147,456

| BLEU   | NIST   | METEOR | ROUGE_L|  CIDEr |
|:------:|:------:|:------:|:------:|:------:|
| 0.6476 | 8.4059 | 0.4381 | 0.6588 | 2.2413 |


#### Improved
Parameter Count: 2,359,296

| BLEU   | NIST   | METEOR | ROUGE_L|  CIDEr |
|:------:|:------:|:------:|:------:|:------:|
| 0.6882 | 8.7540 | 0.4667 | 0.7103 | 2.4537 |
