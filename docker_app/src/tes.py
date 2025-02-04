from llms import FinetuneLM

sample_data = [
    {"input": "Explain gravity", "output": "Gravity is a fundamental force..."},
    {"input": "What is AI?", "output": "AI, or Artificial Intelligence, is..."}
]
finetuner = FinetuneLM(
    data=sample_data,
)
finetuner.run()

print("Finetuning completed successfully.")