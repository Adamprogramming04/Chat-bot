from transformers import GPT2LMHeadModel, GPT2Tokenizer

def train_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer
