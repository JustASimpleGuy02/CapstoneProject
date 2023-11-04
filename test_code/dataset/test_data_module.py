# %%
import os.path as osp
import sys

sys.path += ["../train_CLIP", ".."]

from data.old_polyvore_text_image_dm import TextImageDataset
from clip.simple_tokenizer import SimpleTokenizer
from clip.clip import tokenize
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

# %%
ds = TextImageDataset(
    data_dir="/home/dungmaster/Projects/CapstoneProject/data",
    custom_tokenizer=True,
)

# %%
n_sample = 10

n_rows = 2
n_cols = n_sample // n_rows

f, ax = plt.subplots(n_rows, n_cols, figsize=(20, 10))

for row in range(n_rows):
    for col in range(n_cols):
        idx = random.randint(0, len(ds) - 1)
        image, text = ds[idx]
        desc_list = text.split(" ")
        for j, elem in enumerate(desc_list):
            if j > 0 and j % 4 == 0:
                desc_list[j] = desc_list[j] + "\n"
        text = " ".join(desc_list)
        ax[row, col].imshow(image)
        ax[row, col].set_xticks([], [])
        ax[row, col].set_yticks([], [])
        ax[row, col].grid(False)
        ax[row, col].set_xlabel(text, fontsize=10)

# %%
count_long_text = 0
for image, text in tqdm(ds):
    if len(text) > 77:
        count_long_text += 1
count_long_text

# %%
_tokenizer = SimpleTokenizer()

# %%
idx = random.randint(0, len(ds) - 1)
image, text = ds[idx]

print("Text: ", text)
print("Tokenized Text: ", tokenize(text))

plt.imshow(image)

# %%
tokenized_text = _tokenizer.encode(text)
decode_text = _tokenizer.decode(tokenized_text)
decode_text

# %%
