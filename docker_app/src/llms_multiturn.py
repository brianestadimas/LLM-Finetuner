import torch
import json
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bf16_supported
from datasets import Dataset
from unsloth.chat_templates import get_chat_template 
from transformers import TrainerCallback


class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            epoch = state.epoch if state.epoch else "?"
            step = state.global_step
            loss = logs.get("loss", "N/A")
            print(f"[{step}/{state.max_steps}, Epoch {epoch}] Step\tTraining Loss: {loss}")


class FinetuneLMAgent:
    def __init__(
        self,
        data,
        epochs=1,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        gradient_accumulation_steps=16,
        optim="adamw_8bit",
        model_id="unsloth/Phi-3.5-mini-instruct",
        peft_r=8,
        peft_alpha=16,
        peft_dropout=0.0,
        system_prompt=""
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.model_id = model_id  # keep track for deepseek logic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.system_prompt = system_prompt

        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=True,
            use_gradient_checkpointing=False,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.base_model,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "up_proj",
                "down_proj", "o_proj", "gate_proj"
            ],
            r=peft_r,
            lora_alpha=peft_alpha,
            lora_dropout=peft_dropout,
            bias="none",
            use_rslora=False,
            loftq_config=None
        )

    def formatting_prompts_func(self, examples):
        convos = examples["messages"]
        # convo_with_prompt = [{"role": "system", "content": self.system_prompt}] + convos
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    def run(self):
        if not self.data:  # Check if data is empty
            print("Dataset is empty. Skipping training and deploying base model...")
            self.base_model
            print("Base model has been saved.")
            return

        dataset = Dataset.from_list(self.data)
        new_data = []
        for row in dataset:
            mapped = self.formatting_prompts_func({"messages": [row["messages"]]})
            new_data.append(
                {
                    **row,
                    "text": mapped["text"][0],
                }
            )
        dataset = Dataset.from_list(new_data)
        
        training_args = SFTConfig(
            learning_rate=self.learning_rate,
            output_dir='./model_cp',
            optim=self.optim,
            logging_steps=1,
            report_to="none",
            fp16=(not is_bf16_supported()),
            bf16=is_bf16_supported(),
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            logging_strategy="steps",
            max_seq_length=8192,
        )
        FastLanguageModel.for_training(self.model)
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=[CustomLoggingCallback()]
        )
        trainer.train()
        
