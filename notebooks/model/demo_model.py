# %%
import sys
sys.path.append("../train_CLIP")
import os
import os.path as osp

# %%
from data.polyvore_text_image_dm import TextImageDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

# %%
from tools import load_model
import torch
from clip import *

# %%
model_path = "../../training_logs/2023_08_29/epoch=14-step=44000.ckpt"
model_name = "RN50"
model, preprocess = load_model(model_path, model_name)

# %%
image_dir = 

# %%
saved_image_embedding = "../model_embeddings/2023_08_29/image_embeddings_test.txt"

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
prompt = "black and white striped shirt"

text_tokens = tokenize([prompt]).cuda()
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens).float()

text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
text_embedding = text_embedding.cpu().numpy()

# Calculate cosine similarity matrix
similarity = text_embedding @ image_embeddings.T

# Find most suitable image for the prompt
id_of_matched_object = np.argmax(similarity)
images[id_of_matched_object]

# %%
text_embedding.shape

# %%
