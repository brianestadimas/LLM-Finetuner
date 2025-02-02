# app.py
import onnxruntime_genai as og
from llm_olive import Olivellm

data = [
        {"input": "I'm thrilled to start my new job!", "output": "YES IM HAPPY"},
        {"input": "I can't believe I lost my keys again.", "output": "I dont know what to answer to be honest, i am not that good"},
        {"input": "This haunted house is terrifying!", "output": "HOLY FEAR"},
    ]

# Instantiate and run the pipeline
pipeline = Olivellm(data=data, epochs=5 , gradient_accumulation_steps=8)
pipeline.run()


model_folder = "models/opt/model"

# Load the base model and tokenizer
model = og.Model(model_folder)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options['max_length'] = 200
search_options['past_present_share_buffer'] = False

chat_template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"

text = "HEY"

# Keep asking for input phrases
while text != "exit":
    if not text:
        print("Error, input cannot be empty")
        exit

    # generate prompt (prompt template + input)
    prompt = f'{chat_template.format(input=text)}'

    # encode the prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    print("Output: ", end='', flush=True)
    # stream the output
    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end='', flush=True)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")

    print()
    text = input("Input: ")