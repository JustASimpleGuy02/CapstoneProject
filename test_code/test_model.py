# %%
import sys

sys.path += ["..", "../train_CLIP"]
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import random

# %%
from data.polyvore_text_image_dm import TextImageDataset
import matplotlib.pyplot as plt
import numpy as np

# %%
from apis import search
from tools import *
import torch
from clip import *

# %%
model_path = "../../training_logs/2023_08_29/epoch=14-step=44000.ckpt"
model_name = "RN50"
model, preprocess = load_model(model_path, model_name)

# %%
ds = TextImageDataset(
    data_dir="/home/dungmaster/Projects/CapstoneProject/data",
    csv_metadata="../data/polyvore_img_desc.csv",
    custom_tokenizer=True,
)

# %%
n = 1000
test_data = [ds[idx] for idx in range(n)]
len(test_data)

# %%
images, texts = tuple(zip(*test_data))
len(images)

# %%
saved_image_embedding = (
    "../model_embeddings/2023_08_29/image_embeddings_test.txt"
)

if osp.isfile(saved_image_embedding):
    image_embeddings = np.loadtxt(saved_image_embedding)
else:
    os.makedirs(osp.dirname(saved_image_embedding), exist_ok=True)
    image_embeddings = []
    for img in tqdm(images):
        image_input = preprocess(img)[None, ...].cuda()
        with torch.no_grad():
            image_feature = model.encode_image(image_input).float()
            image_embeddings.append(image_feature)
    image_embeddings = torch.cat(image_embeddings)
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    image_embeddings = image_embeddings.cpu().numpy()
    np.savetxt(saved_image_embedding, image_embeddings)

# %%
saved_text_embedding = (
    "../model_embeddings/2023_08_29/text_embeddings_test.txt"
)

if osp.isfile(saved_text_embedding):
    text_embeddings = np.loadtxt(saved_text_embedding)
else:
    text_embeddings = []
    for text in tqdm(texts):
        text_tokens = tokenize([text]).cuda()
        with torch.no_grad():
            text_feature = model.encode_text(text_tokens).float()
        text_embeddings.append(text_feature)
    text_embeddings = torch.cat(text_embeddings)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.cpu().numpy()
    np.savetxt(saved_text_embedding, text_embeddings)

# %%
# get a sample text
idx = 21
text = texts[idx]
print(f"Prompt: {text}")

# get the text imbedding
txt_embed = text_embeddings[idx]

# find the matched image according to the model
similarity = txt_embed @ image_embeddings.T
id_of_matched_object = np.argmax(similarity)
print(f"Matched id: {id_of_matched_object}")

# %%
# show the predicted image
print("Predicted item:")
images[id_of_matched_object]

# %%
# show the actual image
print("Actual item:")
images[idx]

# %%
n_rows = 2
n_cols = 5

f, ax = plt.subplots(n_rows, n_cols, figsize=(20, 10))
ax[0, 0].set_ylabel("Groundtruth")
ax[1, 0].set_ylabel("Predicted")

for col in range(n_cols):
    idx = random.randint(0, len(text_embeddings) - 1)
    text = texts[idx]

    # get the text imbedding
    txt_embed = text_embeddings[idx]

    # find the matched image according to the model
    similarity = txt_embed @ image_embeddings.T
    id_of_matched_object = np.argmax(similarity)

    desc_list = text.split(" ")
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 4 == 0:
            desc_list[j] = desc_list[j] + "\n"
    text = " ".join(desc_list)
    ax[0, col].imshow(images[idx])
    ax[0, col].set_xticks([], [])
    ax[0, col].set_yticks([], [])
    ax[0, col].grid(False)
    ax[0, col].set_xlabel(text, fontsize=10)

    predicted_item_text = texts[id_of_matched_object]
    desc_list = predicted_item_text.split(" ")
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 4 == 0:
            desc_list[j] = desc_list[j] + "\n"
    text = " ".join(desc_list)
    ax[1, col].imshow(images[id_of_matched_object])
    ax[1, col].set_xticks([], [])
    ax[1, col].set_yticks([], [])
    ax[1, col].grid(False)
    ax[1, col].set_xlabel(text, fontsize=10)

### Test on H&M Dataset
# %%
data_dir = "../data"
image_dir = osp.join(data_dir, "fashion_items_test")
embeddings_file = "../model_embeddings/2023_08_29/image_embeddings_demo.txt"

image_paths = glob(osp.join(image_dir, "*.jpg"))
image_embeddings = np.loadtxt(embeddings_file)

prompts = [
    "red shirt",
    "pink short",
    "white sneaker",
    "round sunglasses",
    "golden ring",
]
topk_matched_images = []

for prompt in prompts:
    found_image_paths, _ = search(
        model,
        preprocess,
        prompt=prompt,
        image_paths=image_paths,
        image_embeddings=image_embeddings,
        top_k=5,
    )
    images = [
        load_image(path, backend="pillow", toRGB=False)
        for path in found_image_paths
    ]
    topk_matched_images.append(images)

display_image_sets(images=topk_matched_images, set_titles=prompts)

# %%
