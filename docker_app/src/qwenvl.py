import torch
from unsloth import FastVisionModel 
from PIL import Image
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from src.utils import find_highest_checkpoint
import shutil, os

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            epoch = state.epoch if state.epoch else "?"
            step = state.global_step
            loss = logs.get("loss", "N/A")
            print(f"[{step}/{state.max_steps}, Epoch {epoch}] Step\tTraining Loss: {loss}")


class FinetuneQwenVL:
    def __init__(self, 
                 data,
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="unsloth/Qwen2-VL-7B-Instruct", 
                 peft_r=8,
                 peft_alpha=16,
                 peft_dropout=0.05,
                 retrain_flag=None
                ):
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        
        if retrain_flag:
            print("Retrain = True. Attempting to resume from latest checkpoint.")
            try:
                model_name = find_highest_checkpoint("./model_cp")
                print(f"Found checkpoint: {model_name}. Resuming training there...")
            except:
                print(f"No checkpoint found in, training from scratch.")
        else:
            model_name = self.model_id
            
        self.base_model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name = self.model_id,
            load_in_4bit = True,
            use_gradient_checkpointing = "unsloth",
        )
        
        shutil.rmtree("./model_cp", ignore_errors=True) if retrain_flag and os.path.exists("./model_cp") else None
        
        self.model = FastVisionModel.get_peft_model(
            self.base_model,
            finetune_vision_layers     = True, # False if not finetuning vision layers
            finetune_language_layers   = True, # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers
            r = peft_r,           
            lora_alpha = peft_alpha,  
            lora_dropout = peft_dropout,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  
            loftq_config = None
        )
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.data = data 
    
    def format_data(self, row):
        image_path = row["image"]
        input_text = row['input']
        output_text = row['output']
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Unable to load image at path: {image_path}. Error: {e}")

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_text,
                        },
                        {
                            "type": "image",
                            "image": image,  
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": output_text,
                        }
                    ],
                },
            ],
        }

    def run(self):
        if not self.data:  # Check if data is empty
            print("Dataset is empty. Skipping training and deploying base model...")
            self.base_model
            print("Base model has been saved without fine-tuning.")
            return

        converted_dataset = [self.format_data(row) for row in self.data]
        training_args = SFTConfig(
            learning_rate=self.learning_rate,
            output_dir='./model_cp',
            optim=self.optim,
            logging_steps=1,
            report_to="none",
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epochs,
            weight_decay = 0.01,            # Regularization term for preventing overfitting
            lr_scheduler_type = "linear",   # Chooses a linear learning rate decay
            seed = 3407,
            logging_strategy = "steps",
            # load_best_model_at_end = True,
            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        )
        FastVisionModel.for_training(self.model)
        
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            data_collator = UnslothVisionDataCollator(self.model, self.tokenizer), # Must use!
            train_dataset = converted_dataset,
            args = training_args,
            callbacks=[CustomLoggingCallback()]
        )
        trainer.train()
