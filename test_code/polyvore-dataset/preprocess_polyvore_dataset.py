# %%
import os.path as osp
from tqdm import tqdm

import numpy as np
import pandas as pd

tqdm.pandas()

# %%
data_dir = "../data"
image_dir = osp.join(data_dir, "images")
path2metadata = osp.join(data_dir, "polyvore_item_metadata.json")

# %%
csv_meta = osp.join(data_dir, "csv_files", "original_polyvore.csv")
df_meta = pd.read_csv(csv_meta)
df_meta.drop(["Unnamed: 0", "path"], axis=1, inplace=True)
df_meta.tail(20)

# %%
df_meta.url_name.isnull().sum()

# %%
csv_categories = osp.join(data_dir, "categories.csv")
df_categories = pd.read_csv(csv_categories, index_col=None)
df_categories.head()

# %%
df_categories.columns

# %%
id2categories = dict(zip(df_categories["2"], df_categories["undefined"]))

# %%
chosen_prompts = df_meta[
    ["id", "url_name", "title", "category_id", "semantic_category"]
].copy()
chosen_prompts.head(10)

# %%
chosen_prompts["tmp"] = np.where(
    df_meta["title"].isnull(), df_meta["url_name"], df_meta["title"]
)
chosen_prompts.dropna(subset=["tmp"], inplace=True)
chosen_prompts.reset_index(drop=True, inplace=True)
len(chosen_prompts)

# %%
chosen_prompts.head(20)

# %%
import sys

sys.path.append("/home/dungmaster/Projects/CapstoneProject/train_CLIP/data")
import re
from process_text import *


def remove_number_phrases(text):
    # remove unwanted numbers with string or character attached to it
    cleaned_text = re.sub(r"[\S]+[\d]+[\S]*", "", text)
    return cleaned_text


def remove_urls_or_domain_names(text):
    cleaned_text = re.sub(
        r"[\S]+\.(com|net|org|edunet|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?",
        "",
        text,
    )
    return cleaned_text


def remove_special_characters(text):
    cleaned_text = re.sub(r"\uFF5C", " ", text)

    return cleaned_text


def clean(text: str):
    text = remove_urls_or_domain_names(text)

    text = remove_number_phrases(text)

    text = remove_special_characters(text)

    text = replace_punctuation_with_whitespace(text)

    text = remove_unwanted_spaces(text)

    return text


# %%
text = "amazon.com 100s i957987 ps1 | imported cashmere gloves	"
text = "cocoon coat｜stunning lure｜stunning lur"
clean(text)

# %%
assert chosen_prompts["category_id"].isnull().sum() == 0
chosen_prompts["category_id"] = chosen_prompts["category_id"].astype(int)
chosen_prompts.head(20)

# %%
chosen_prompts["category"] = chosen_prompts["category_id"].progress_apply(
    lambda x: id2categories[x]
)
chosen_prompts.head(20)

# %%
full_titles = []
for idx, row in tqdm(chosen_prompts.iterrows()):
    category = row["category"]
    title = row["tmp"]
    if category not in title:
        full_titles.append(title + " " + category)
    else:
        full_titles.append(title)

chosen_prompts["final_title"] = full_titles
chosen_prompts.head(20)

# %%
chosen_prompts.drop(
    ["category_id", "semantic_category", "tmp"], axis=1, inplace=True
)

# %%
chosen_prompts["tmp"] = chosen_prompts["final_title"].progress_apply(clean)
chosen_prompts.head(20)

# %%
chosen_prompts.drop(
    ["url_name", "title", "category", "final_title"], axis=1, inplace=True
)
chosen_prompts.rename(columns={"tmp": "final_title"}, inplace=True)
chosen_prompts.head(20)

# %%
chosen_prompts["image_name"] = chosen_prompts["id"].progress_apply(
    lambda x: str(x) + ".jpg"
)
chosen_prompts.head(20)

# %%
csv_image_prompt = "../data/polyvore_image_desc.csv"
chosen_prompts.to_csv(csv_image_prompt, index=False)

# %%
chosen_prompts = pd.read_csv(csv_image_prompt, index_col=None)
chosen_prompts.drop("id", axis=1, inplace=True)
chosen_prompts.head(20)

# %%
len(chosen_prompts)

# %%
