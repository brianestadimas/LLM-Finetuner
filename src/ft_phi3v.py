
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_dataset
from torchvision import transforms
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from PIL import Image
import requests
import re

class FinetunePhi3V:
    def __init__(self, 
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="microsoft/Phi-3-vision-128k-instruct", 
                 dataset_path="Vision-Flan/vision-flan",
                 dataset_split="train",
                 datase_source="hf",
                 peft_r=8,
                 peft_alpha=16,
                 peft_dropout=0.05,
                 formatter="<|user|>\n<|image_1|>{prompt}<|end|><|assistant|>{answer}<|end|>",
                 image_column_name="image",
                 is_url=False):
        self.dataset_path = dataset_path
        self.dataset_source = datase_source
        self.dataset_split = dataset_split
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            _attn_implementation='eager',
            trust_remote_code=True,
            torch_dtype="auto",
            quantization_config=self.bnb_config
        )
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=peft_r,
            lora_alpha=peft_alpha,
            lora_dropout=peft_dropout,
            target_modules=['k_proj','q_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']
        )
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.formatter = formatter
        self.image_column_name = image_column_name
        self.is_url = is_url
        
    def run(self):
        if self.dataset_path:
            df = pd.read_csv(self.dataset_path)
            dataset = ImageTextDataset(df, self.tokenizer, from_csv=True)
        else:
            ds = load_dataset(self.hf_dataset)
            dataset = ImageTextDataset(ds[self.dataset_split], self.tokenizer)
    
        model = get_peft_model(self.base_model, self.peft_config)
        training_args = TrainingArguments(
            learning_rate=1e-4,
            output_dir='./phi3_vision128k_lora',
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
            optim=self.optim
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()


class ImageTextDataset(Dataset):
    def __init__(self, data, tokenizer, from_csv, formatter, image_column_name, is_url):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.from_csv = from_csv
        self.formatter = formatter
        self.placeholders = re.findall(r"{([^}]+)}", formatter)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        self.image_column_name = image_column_name
        self.is_url = is_url
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_url:
            if self.from_csv:
                row = self.data.iloc[idx]
                image_url = row[self.image_column_name]
                data_dict = {}
                for ph in self.placeholders:
                    data_dict[ph] = row[ph]
            else:
                example = self.data[idx]
                image_url = example[self.image_column_name]
                data_dict = {}
                for ph in self.placeholders:
                    data_dict[ph] = example[ph]
            
            img_data = requests.get(image_url, stream=True).raw
            image = Image.open(img_data).convert("RGB")
        else:
            image = self.data[idx][self.image_column_name]

        image = self.image_transform(image)
        text = self.formatter.format(**data_dict)
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=256)
        encodings['pixel_values'] = image
        encodings['labels'] = encodings['input_ids'].copy()
        return {key: torch.tensor(val) for key, val in encodings.items() if isinstance(val, (list, torch.Tensor))}  


if __name__ == "__main__":
    print("Finetune Phi3V")