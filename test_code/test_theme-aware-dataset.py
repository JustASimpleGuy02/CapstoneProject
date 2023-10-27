# %%
import os
import os.path as osp
from glob import glob
import sys
import time

import random
from pprint import pprint
from pathlib import Path
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from translatepy.translators.google import GoogleTranslateV2

tqdm.pandas()
sys.path.append("../")
from tools import load_image, load_json, display_image_sets, load_csv, to_csv

sns.set_theme()

# %%
gtranslate = GoogleTranslateV2()
name = lambda x: Path(x).name
outfit_fields = [
    "Outfit_Description",
    "Outfit_Name",
    "Oufit_Fit",
    "Outfit_Gender",
    "Outfit_Occasion",
    "Outfit_Style",
]
cn_outfit_fields = ["cn_" + f for f in outfit_fields]
en_outfit_fields = ["en_" + f for f in outfit_fields]

# %%
data_dir = "/home/dungmaster/Datasets/Fashion_Outfits_Theme_Aware"
outfit_dirs = glob(osp.join(data_dir, "*/"))

# %%
df = pd.DataFrame(columns=["id"] + cn_outfit_fields + en_outfit_fields)

for outfit in tqdm(outfit_dirs):
    cn_fields = []
    en_fields = []

    metadata = load_json(osp.join(outfit, name(outfit) + ".json"))
    cn_desc = ""
    for field in outfit_fields:
        cnf = metadata.get(field, "")
        enf = ""

        if len(cnf) != 0:
            enf = gtranslate.translate(cnf, "English")

        cn_fields.append(cnf)
        en_fields.append(enf)

    df.loc[len(df.index)] = [name(outfit)] + cn_fields + en_fields

# %%
df.tail(10)

# %%
to_csv("../theme_aware_dataset_descriptions_v2.csv", df)

# %%
json_desc = {}

for idx, row in tqdm(df.iterrows()):
    desc = {}
    for f in cn_outfit_fields + en_outfit_fields:
        desc[f] = str(row[f])

    json_desc[row.id] = desc

# %%
json_data = json.dumps(json_desc, ensure_ascii=False)
json_data

# %%
# the json file where the output must be stored
with open(
    "../theme_aware_dataset_descriptions.json", "w", encoding="utf-8"
) as f:
    json.dump(json_desc, f, indent=6, ensure_ascii=False)
    f.close()

json_desc

# %%
n_outfits = len(df)
rand_ind = random.randint(0, n_outfits - 1)
desc_outfit = df.loc[rand_ind]
cn_desc = str(desc_outfit.cn_description)[1:]
en_desc = str(desc_outfit.en_description)[1:]


# %%
def remove_redundant_colons(text: str):
    text = text.replace(". .", ".")
    return text


cn_desc = remove_redundant_colons(cn_desc)
en_desc = remove_redundant_colons(en_desc)
print("Chinese:", cn_desc)
print("English:", en_desc)

# %%
