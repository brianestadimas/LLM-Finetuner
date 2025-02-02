import json
import os
import subprocess

class Olivellm:
    def __init__(
        self,
        data,
        epochs=10,
        gradient_accumulation_steps=16,
        peft_r=8,
        peft_alpha=16,
        optim="adamw_torch",
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct", 
        learning_rate=1e-4,
        warmup_ratio=0.1,
    ):
        """
        A single class to:
          1) Quantize (auto-opt) the model at a fixed directory (models/opt).
          2) Finetune (finetune) the model at a fixed directory (models/finetuned).
        """
        self.data = data
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.peft_r = peft_r
        self.peft_alpha = peft_alpha
        self.optim = optim
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio

    def run(self):
        # 1) Finetuning
        cmd_finetune = [
            "olive", "finetune",
            "--method", "lora",
            "--model_name_or_path", self.model_id,
            "--trust_remote_code",
            "--data_name", "./data_train",
            "--data_files", "./data_train/dataset.json",
            "--text_template", "<|im_start|>user\n{phrase}<|im_end|>\n<|im_start|>assistant\n{tone}<|im_end|>",
            "--num_train_epochs", str(self.epochs),
            "--output_path", "models/finetuned",
            "--log_level", "1",
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--gradient_accumulation_steps", str(self.gradient_accumulation_steps),
            "--learning_rate", str(self.learning_rate),
            "--warmup_ratio", str(self.warmup_ratio),
            "--optim", self.optim,
            "--lora_r", str(self.peft_r),
            "--lora_alpha", str(self.peft_alpha)
        ]
        subprocess.run(cmd_finetune, check=True)

        os.makedirs("./data_train", exist_ok=True)
        with open("./data_train/dataset.json", "w", encoding="utf-8") as f:
            for row in self.data:
                converted = {
                    "phrase": row["input"],
                    "tone": row["output"]
                }
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")

        # 2) Quantization
        cmd_auto_opt = [
            "olive", "auto-opt",
            "--model_name_or_path", "models/finetuned",
            "--output_path", "models/opt",
            "--device", "cpu",
            "--provider", "CPUExecutionProvider",
            "--use_ort_genai",
            "--precision", "int4",
            "--log_level", "1",
        ]
        subprocess.run(cmd_auto_opt, check=True)
        