# Finetuning GPT2 Small for the E2E NLG Challenge Using LoRA
I've made my own implementation of the LoRA framework and used it to 
reconstruct the original implementation from the [LoRA paper](https://arxiv.org/abs/2106.09685).
Note that I've used GPT2 Small instead of GPT2 Medium, due to resource constraints.

Listed Below are the results collected using the [official E2E NLG Challenge metrics script](https://github.com/tuetschek/e2e-metrics).

## Results

#### Baseline
Parameter Count: 147,456

BLEU: 0.6476
NIST: 8.4059
METEOR: 0.4381
ROUGE_L: 0.6588
CIDEr: 2.2413

#### Improved
Parameter Count: 2,359,296

BLEU: 0.6882
NIST: 8.7540
METEOR: 0.4667
ROUGE_L: 0.7103
CIDEr: 2.4537