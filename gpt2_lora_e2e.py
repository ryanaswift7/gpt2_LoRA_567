from transformers import AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, IntervalStrategy
import torch
from datasets import load_dataset
from peft import LoraConfig
from lora_model import LoRA_Model
from safetensors.torch import load_file

class GPT2_LoRA_E2E():
    def __init__(self, model_name="baseline", gen_test_data=True):
        super(GPT2_LoRA_E2E, self).__init__()
        self.model_name = model_name
        self.output_dir = f"./{self.model_name}"
        self.gen_test_data = gen_test_data
        self._init_tokenizer()
        self._init_gpt2()
        self._set_lora_config()
        self._init_lora_model()
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            warmup_steps=500,
            label_smoothing_factor=0.1,
            logging_dir=f"./{self.model_name}_logs",
            logging_steps=500,
            save_total_limit=2,
            fp16=True,
        )

    def _init_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # add token to indicate where MR ends and ref begins
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('reference')
        tokenizer._reference = '<|ref|>'
        tokenizer.add_special_tokens({'reference':"<|ref|>"})
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer


    def _init_gpt2(self):
        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2.resize_token_embeddings(len(self.tokenizer))
        self.gpt2 = gpt2

    def _set_lora_config(self):
        if self.model_name == "baseline":
            self.lora_config = LoraConfig(
                r=4,
                lora_alpha=32,
                target_modules=["c_attn"],  # only W_q and W_v are used in the benchmark
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                init_lora_weights = True,
            )
        elif self.model_name == "improved":
            self.lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["c_attn", "c_proj", "c_fc"],  # all attn and mlp layers
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                init_lora_weights = "gaussian",
            )
        else:
            print("Invalid model name specified. Must be either 'baseline' or 'improved'.")
            exit(1)

    def _init_lora_model(self):
        self.model = LoRA_Model(self.gpt2, self.lora_config).model
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{self.model_name.upper()} Trainable Parameters: {trainable_params}")

    def init_data(self):
        # load dataset
        dataset = load_dataset('e2e_nlg')
        # tokenize each example
        dataset = dataset.map(
            lambda ex: self.tokenizer(
                list(
                    map(
                        lambda mr, ref: mr + " <|ref|> " + ref, ex['meaning_representation'], ex['human_reference']
                    )
                ),
                truncation=True,
                max_length=128,
                padding='max_length',
            ),
            batched=True,
        )

        # labels for each example are same as inputs
        dataset = dataset.map(lambda ex: {'labels': ex['input_ids']})

        self.train_data = dataset["train"]
        self.val_data = dataset["validation"]
        self.test_data = dataset["test"]

    def _init_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
        )

    def train(self):
        self._init_trainer()
        self.trainer.train()
        self.tokenizer.save_pretrained(self.output_dir)

    def _generate_test_data(self):
        # generate test data files if requested
        if self.gen_test_data:
            # load test data into dataloader and init data structs
            test_data = self.test_data
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
            # write list of refs to file, in accordance with official metric
            # script required format
            with open('test_refs.txt', 'w+') as f:
                for i, mr in enumerate(mrs):
                    mr = mr.replace('\n', '')
                    f.writelines(list(map(lambda ref: ref + '\n', test_dict[mr])))
                    if i<len(mrs):
                        f.write('\n')
            print("Output files in current directory:")
            print("test_refs.txt -> reference file for E2E NLG challenge test set")
            print("test_mrs.txt -> input file for E2E NLG challenge test set\n")

    def _generate_outputs(self, mr_batch:str):
        mr_batch = mr_batch.replace('\n', '')
        mr_batch = mr_batch + " <|ref|> "
        tokenized_mr_batch = self.tokenizer(mr_batch, return_tensors='pt').to(self.model.device)
    
        output_tokens = self.model.generate(
            **tokenized_mr_batch,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.model.config.eos_token_id,  # Ensure eos_token_id is set
            early_stopping=True,
            num_beams=10,
            length_penalty=0.9,
            no_repeat_ngram_size=4,
            max_length=128,
        )
        
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=False)
    
        # Remove the MR from the generated output
        if "<|ref|>" in output_text:
            output_text = output_text.split("<|ref|>", 1)[1].strip()  # Split and take the part after "<|ref|>"
    
        # some outputs have an extra ref token, so remove it
        if "<|ref|>" in output_text:
            output_text = output_text.split("<|ref|>", 1)[1].strip()
    
        # remove the <|endoftext|> token
        if "<|endoftext|>" in output_text:
            output_text = output_text.replace("<|endoftext|>", "")
    
        return output_text
    
    def _make_output_file(self):
        with open('test_mrs.txt','r+') as f:
            mrs = f.readlines()
        output_text = [self._generate_outputs(mr) for mr in mrs]
        output_text = list(map(lambda txt: txt + '\n', output_text))
        with open(f"{self.model_name}_outputs.txt", 'w') as f:
            f.writelines(output_text)

    def generate(self):
        self._generate_test_data()
        self._make_output_file()
        print("Please use the official script for computing metrics. It can be found at https://github.com/tuetschek/e2e-metrics/tree/master")
        print(f"Use test_refs.txt as the ref_file, and {self.model_name}_outputs.txt as the sys_file")



    


