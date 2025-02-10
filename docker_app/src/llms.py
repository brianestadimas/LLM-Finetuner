import psutil
import torch
import os, time
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bf16_supported
from unsloth.chat_templates import get_chat_template  # Import get_chat_template
from datasets import Dataset  # Import Dataset from Hugging Face datasets
import subprocess

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            epoch = state.epoch if state.epoch else "?"
            step = state.global_step
            loss = logs.get("loss", "N/A")
            print(f"[{step}/{state.max_steps}, Epoch {epoch}] Step\tTraining Loss: {loss}")


class FinetuneLM:
    def __init__(
        self,
        data, 
        epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        gradient_accumulation_steps=16,
        optim="adamw_torch",
        model_id="unsloth/Phi-3.5-mini-instruct",
        peft_r=8,
        peft_alpha=16,
        peft_dropout=0.0
    ):
        self.data = data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        # 1. Load base text model and tokenizer
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            load_in_4bit=False,
            use_gradient_checkpointing=False,
        )

        # 2. Wrap the model with LoRA (text-only)
        self.model = FastLanguageModel.get_peft_model(
            self.base_model,
            # target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            target_modules=["q_proj", "v_proj"],
            r=peft_r,
            lora_alpha=peft_alpha,
            lora_dropout=peft_dropout,
            bias="none",
            use_rslora=False,
            loftq_config=None
        )

    # def format_data(self, row):
    #     user_prompt = row["input"]
    #     assistant_answer = row["output"]
    #     messages = [
    #         {"role": "user", "content": user_prompt},
    #         {"role": "assistant", "content": assistant_answer},
    #     ]
    #     try:
    #         formatted_text = self.tokenizer.apply_chat_template(
    #             messages,
    #             tokenize=False,
    #         )
    #         print(f"Formatted text: {formatted_text}")
    #     except Exception as e:
    #         formatted_text = ""
    #         for msg in messages:
    #             formatted_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        
    #     return {"text": formatted_text}

    def format_data(self, row):
        user_prompt = row["input"]
        assistant_answer = row["output"]
        formatted_text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{assistant_answer}<|im_end|>"
        return {"text": formatted_text}

    def run(self):
        """
        Execute LoRA fine-tuning on the provided text data.
        """
        # Convert data to the specified format
        formatted_data = [self.format_data(row) for row in self.data]
        dataset = Dataset.from_list(formatted_data)

        # Create SFT config
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
            # Remove dataset_kwargs to enable automatic tokenization
            max_seq_length=2048,
        )

        # Prepare model for training
        FastLanguageModel.for_training(self.model)

        # Initialize and run trainer
        trainer = SFTTrainer(
            model=self.model,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=[CustomLoggingCallback()]
        )
        trainer.train()
        
        save_path = "./model_cp/saved"
        try:
            model = self.model.merge_and_unload()
            model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Model has been saved")
        except Exception as e:
            print(f"Ignore the model saving")


def olive_opt():

    cmd_auto_opt = [
        "olive", "auto-opt",
        "--model_name_or_path", "model_cp/saved",
        "--output_path", "model_cp/opt",
        "--device", "cpu",
        "--provider", "CPUExecutionProvider",
        "--use_ort_genai",
        "--precision", "int4",
        "--log_level", "1",
    ]

    # Define the memory usage threshold (in percent)
    memory_threshold = 97

    try:
        # Start the subprocess
        process = subprocess.Popen(cmd_auto_opt)
        print("Auto-opt process started, monitoring memory usage...")

        # Monitor the process until it completes
        while process.poll() is None:
            mem_usage = psutil.virtual_memory().percent
            if mem_usage >= memory_threshold:
                print(f"Memory usage is high ({mem_usage}%). Terminating auto-opt process to avoid getting stuck.")
                process.terminate()  # or process.kill() for a forceful termination
                break
            # Check every 1 second
            time.sleep(1)

        # Wait for the process to finish and get the return code
        retcode = process.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, cmd_auto_opt)
        print("Auto-opt finished successfully.")
    except Exception as e:
        print(f"Auto-opt failed: {e}")
