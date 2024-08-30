from pytera import *
from json import load
import re

dataset = load(open("data.json", "r", encoding="utf-8"))

# automatically generate the vocab based on all unique tokens in the dataset
vocab_words = []
for text in dataset:
    for word in text.split(" "):
        word = re.sub(r'[^\w\s]', '', word)
        if word:
            vocab_words.append(word)
vocab_words = list(set(vocab_words))
vocab_words.append("<UNK>")
vocab_words.append("<EOS>")

print("Vocab size:", len(vocab_words) - 1)

model = GPTModel(
    input_size=128, d_model=256, num_heads=8, ff_dim=512, num_layers=6, vocab_size=len(vocab_words) - 1, vocab=vocab_words
)
model.train(dataset, epochs=10, learning_rate=0.001)
model.save("model.tera")
output = model.generate_text("Once upon a time")
