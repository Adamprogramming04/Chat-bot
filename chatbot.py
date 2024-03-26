import torch

def generate_response(input_text, model, tokenizer):
  input_ids = tokenizer.encode(input_text, return_tensors='pt')
  output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)


    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
