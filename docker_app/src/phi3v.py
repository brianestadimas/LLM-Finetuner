from tqdm import tqdm
import sys
import time
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from torchvision import transforms
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from PIL import Image
import re
from transformers import TrainerCallback
from src.utils import find_highest_checkpoint
import shutil, os
from peft import PeftModel
import torch.nn as nn

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            epoch = state.epoch if state.epoch else "?"
            step = state.global_step
            loss = logs.get("loss", "N/A")
            print(f"[{step}/{state.max_steps}, Epoch {epoch}] Step\tTraining Loss: {loss}")


class ImageTextDataset(Dataset):
    def __init__(self, data, tokenizer, formatter):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.formatter = formatter
        self.placeholders = re.findall(r"{([^}]+)}", formatter)
        self.image_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_path = row['image']
        input_text = row['input']
        output_text = row['output']

        # Load and transform the image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Unable to load image at path: {image_path}. Error: {e}")

        image = self.image_transform(image)

        data_dict = {}
        for placeholder in self.placeholders:
            if placeholder == 'prompt':
                data_dict[placeholder] = input_text
            elif placeholder == 'answer':
                data_dict[placeholder] = output_text
            else:
                raise ValueError(f"Unexpected placeholder '{placeholder}' in formatter.")

        # Format the text using the formatter
        try:
            formatted_text = self.formatter.format(**data_dict)
        except KeyError as e:
            raise KeyError(f"Missing key for formatter: {e}")

        # Tokenize the formatted text
        encodings = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )

        # Prepare labels by copying input_ids
        encodings['labels'] = encodings['input_ids'].clone()

        # Squeeze to remove the batch dimension
        encodings = {key: val.squeeze(0) for key, val in encodings.items()}

        # Add pixel_values to encodings
        encodings['pixel_values'] = image

        return encodings

class ForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        # Remove the unexpected 'num_items_in_batch' if present.
        kwargs.pop("num_items_in_batch", None)
        return self.model(*args, **kwargs)


class FinetunePhi3V:
    def __init__(self, 
                 data,  # New parameter to receive data directly
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="microsoft/Phi-3-vision-128k-instruct", 
                 peft_r=8,
                 peft_alpha=16,
                 peft_dropout=0.05,
                 retrain_flag=None
                ):
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Enable 8-bit loading
            bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation
            bnb_8bit_use_double_quant=True,  # Use double quantization for memory efficiency
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            _attn_implementation='eager',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        if retrain_flag:
            print("Retrain = True. Attempting to resume from latest checkpoint.")
            try:
                adapter_path = find_highest_checkpoint("./model_cp")
                self.base_model = PeftModel.from_pretrained(self.base_model, adapter_path)
                self.base_model = ForwardWrapper(self.base_model)
            except:
                print(f"No checkpoint found in, training from scratch.")
        shutil.rmtree("./model_cp", ignore_errors=True) if retrain_flag and os.path.exists("./model_cp") else None
        
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=peft_r, 
            lora_alpha=peft_alpha, 
            lora_dropout=peft_dropout, 
            target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
            inference_mode = False
        )
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.formatter = "<|user|>\n<|image_1|>{prompt}<|end|><|assistant|>{answer}<|end|>"
        self.data = data  # Store the data

    def run(self):
        if not self.data:  # Check if data is empty
            print("Dataset is empty. Skipping training and deploying base model...")
            self.base_model
            print("Base model has been saved without fine-tuning.")
            return

        dataset = ImageTextDataset(
            data=self.data,
            tokenizer=self.tokenizer,
            formatter=self.formatter
        )

        model = get_peft_model(self.base_model, self.peft_config)

        training_args = TrainingArguments(
            learning_rate=self.learning_rate,
            output_dir='./model_cp',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            bf16=True,
            dataloader_num_workers=0,
            report_to="none",
            optim=self.optim,
            logging_steps=1, 
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[CustomLoggingCallback()]
        )
        trainer.train()
