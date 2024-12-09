from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, IntervalStrategy
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
import random
import argparse


def generate_text(model, tokenizer, mr_batch:str):
    mr_batch = mr_batch.replace('\n', '')
    mr_batch = mr_batch + " <|ref|> "
    tokenized_mr_batch = tokenizer(mr_batch, return_tensors='pt').to(model.device)

    output_tokens = model.generate(
        **tokenized_mr_batch,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=model.config.eos_token_id,  # Ensure eos_token_id is set
        early_stopping=True,
        num_beams=10,
        length_penalty=0.9,
        no_repeat_ngram_size=4,
    )

    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)

    # Remove the MR from the generated output
    if "<|ref|>" in output_text:
        output_text = output_text.split("<|ref|>", 1)[1].strip()  # Split and take the part after "<|ref|>"

    # remove the <|endoftext|> token
    if "<|endoftext|>" in output_text:
        output_text = output_text.replace("<|endoftext|>", "")

    return output_text



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Select whether you wish to train the base model or improved model'
        )
    parser.add_argument("--model", type=str, required=True, choices=['base', 'improved'], help="Run either base or improved model")
    parser.add_argument("--gen_test_data", type=bool, default=True, help='Generates files for test MRs and refs')
    args = parser.parse_args()


    # enforce reproducibility
    seed = 44
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(42)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load gpt2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # add token to indicate where MR ends and ref begins
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('reference')
    tokenizer.add_special_tokens({'reference':"<|ref|>"})
    # tokenizer._reference = '<|ref|>'
    tokenizer.pad_token = tokenizer.eos_token



    # load dataset
    dataset = load_dataset('e2e_nlg')

    # tokenize each example
    dataset = dataset.map(
        lambda ex: tokenizer(
            list(
                map(
                    lambda mr, ref: mr + " <|ref|> " + ref, ex['meaning_representation'], ex['human_reference']
                )
            ),
            truncation='only_second',
            max_length=128,
            padding='max_length',
        ),
        batched=True,
    )

    # labels for each example are same as inputs
    dataset = dataset.map(lambda ex: {'labels':ex['input_ids']})

    # load pretrained gpt2 and update tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    if args.model == 'base':
        output_dir = './base'

        # From the original implementation in https://arxiv.org/abs/2106.09685
        # Note that we are using vanilla GPT-2 (i.e. GPT-2 Small), rather than the original GPT-2 Medium
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            target_modules=["c_attn"],  # only W_q and W_v are used in the benchmark
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",  # TaskType.LM,
            init_lora_weights = True,
        )
        lora_model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

        # set baseline training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            warmup_steps=500,
            label_smoothing_factor=0.1,
            logging_dir='./base_logs',
            logging_steps=500,
            save_total_limit=2,
            fp16=True,
        )
    
    elif args.model == "improved":
        output_dir='./improved'

        # My implementation to improve upon baseline
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj", "c_fc", "wte", "wpe"],  # only W_q and W_v are used in the benchmark
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",  # TaskType.LM,
            init_lora_weights = "gaussian",
        )
        lora_model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

        # set improved training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            # eval_strategy="no",
            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            warmup_steps=500,
            label_smoothing_factor=0.1,
            logging_dir='./better_logs',
            logging_steps=500,
            save_total_limit=2,
            fp16=True,
        )

    print(f"{args.model.upper()} Trainable Parameters: {trainable_params}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        # data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    tokenizer.save_pretrained(args.output_dir)

    # generate test data files if requested
    if args.gen_test_data:

        # load test data into dataloader and init data structs
        test_data = dataset['test']
        dl = torch.utils.data.DataLoader(test_data, batch_size=1)
        test_dict = {}
        mrs = []

        # group refs by mr and form list of mrs
        for example in dl:
            mr = example["meaning_representation"][0]
            ref = example["human_reference"][0]

            if mr not in mrs:
                mrs.append(mr)
            
            if mr not in test_dict:
                test_dict[mr] = [ref]
            else:
                test_dict[mr].append(ref)

        # write list of mrs to file
        mrs = list(map(lambda mr: mr + '\n', mrs))
        with open("./test_mrs.txt", 'w') as f:
            f.writelines(mrs)
            f.write('\n')

        # write list of refs to file, in accordance with official metric
        # script required format
        with open('test_refs.txt', 'w+') as f:
            for i, mr in enumerate(mrs):
                mr = mr.replace('\n', '')
                f.writelines(list(map(lambda ref: ref + '\n', test_dict[mr])))
                if i<len(mrs)-1:
                    f.write('\n')

    print("Begin Output Generation")

    with open('test_mrs.txt','r+') as f:
        mrs = f.readlines()

    output_text = [generate_text(model, tokenizer, mr) for mr in mrs]

    output_text = list(map(lambda txt: txt + '\n', output_text))
    with open(f"{args.model}_outputs.txt", 'w') as f:
        f.writelines(output_text)
        f.write('\n')
    





