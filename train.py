import argparse
from transformers import GPT2Tokenizer

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(tokenizer(["Hello world", "How are you today?"]))