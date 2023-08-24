# %%
import sys
sys.path.append("/home/dungmaster/Projects/CapstoneProject")
sys.path.append("/home/dungmaster/Projects/CapstoneProject/train_CLIP")
# %%
from data.custom_text_image_dm import TextImageDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

# %%
from tools import load_model
import torch
from clip import *

# %%
model_path = "../../training_logs/2023_08_21/epoch=47-step=43248.ckpt"
model_name = "RN50"
model, preprocess = load_model(model_path, model_name)

# %%
ds = TextImageDataset(data_dir="/home/dungmaster/Projects/CapstoneProject/data",
                    custom_tokenizer=True)

# %%
n = 1000
test_data = [ds[idx] for idx in range(n)]
len(test_data)
# %%
images, texts = tuple(zip(*test_data))
len(images)
# %%
image_embeddings = []
for img in tqdm(images):
    image_input = preprocess(img)[None, ...].cuda()
    with torch.no_grad():
        image_feature = model.encode_image(image_input).float()
        image_embeddings.append(image_feature)
image_embeddings = torch.cat(image_embeddings)
image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
image_embeddings = image_embeddings.cpu().numpy()
np.savetxt("image_embeddings_test.txt", image_embeddings)
# %%
text_embeddings = []
for text in tqdm(texts):
    text_tokens = tokenize([text]).cuda()
    with torch.no_grad():
        text_feature = model.encode_text(text_tokens).float()
    text_embeddings.append(text_feature)
text_embeddings = torch.cat(text_embeddings)
text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
text_embeddings = text_embeddings.cpu().numpy()
np.savetxt("text_embeddings_test.txt", image_embeddings)

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

f, ax = plt.subplots(n_rows, n_cols, figsize=(20,10))
ax[0, 0].set_ylabel("Groundtruth")
ax[1, 0].set_ylabel("Predicted")

for col in range(n_cols):
    idx = random.randint(0, len(text_embeddings)-1)
    text = texts[idx]
    
    # get the text imbedding
    txt_embed = text_embeddings[idx]

    # find the matched image according to the model
    similarity = txt_embed @ image_embeddings.T
    id_of_matched_object = np.argmax(similarity)
        
    desc_list = text.split(' ')
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 4 == 0:
            desc_list[j] = desc_list[j] + '\n'
    text = ' '.join(desc_list)
    ax[0, col].imshow(images[idx])
    ax[0, col].set_xticks([], [])
    ax[0, col].set_yticks([], [])
    ax[0, col].grid(False)
    ax[0, col].set_xlabel(text, fontsize=10)
        
    predicted_item_text = texts[id_of_matched_object]
    desc_list = predicted_item_text.split(' ')
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 4 == 0:
            desc_list[j] = desc_list[j] + '\n'
    text = ' '.join(desc_list)
    ax[1, col].imshow(images[id_of_matched_object])
    ax[1, col].set_xticks([], [])
    ax[1, col].set_yticks([], [])
    ax[1, col].grid(False)
    ax[1, col].set_xlabel(text, fontsize=10)
    # %%
    print(f"Text of item index 0:", texts[0])
    print(f"Text of item index 24:", texts[24])

# %%
prompt = "orange shirt"
text_tokens = tokenize([prompt]).cuda()
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens).float()

text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
text_embedding = text_embedding.cpu().numpy()

# Calculate cosine similarity matrix
similarity = text_embedding @ image_embeddings.T

# Find most suitable image for the prompt
id_of_matched_object = np.argmax(similarity)
print(texts[id_of_matched_object])
images[id_of_matched_object]
# %%
