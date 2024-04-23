from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_NAME = "./" + "pretrained_gpt2"
PRMOPT = ""

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

ids = tokenizer.encode(PRMOPT, return_tensors="pt")

output = model.generate(ids, max_length=200, num_return_sequences=1, temperature=0.1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)
