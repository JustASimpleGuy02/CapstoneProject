# %%
import os.path as osp
from glob import glob
import random

import numpy as np
from sklearn.metrics import accuracy_score

from fashion_clip.fashion_clip import FashionCLIP
from reproducible_code.tools import plot, io

# %%
project_dir = "/home/dungmaster/Projects/Machine Learning/HangerAI_outfits_recommendation_system/CapstoneProject"
img_dir = osp.join(project_dir, "data", "theme-aware-dataset", "test_images")
img_paths = glob(osp.join(img_dir, "*/*.jpg"))
len(img_paths)

# %%
label_df = io.load_csv(
    osp.join(img_dir, "colours_label.csv")
)
label_df.head()

# %%
label_df["image"] = label_df["image"].apply(
    lambda x: glob(osp.join(img_dir, "*", osp.basename(x).split('-')[1]))[0]
)
label_df.head()

# %% [markdown]
# ### Known Colours

# %%
known_colour_df = label_df[label_df["choice"] != "unknown"]
len(known_colour_df)

# %%
known_colour_df["image"].sample(1).values

# %%
gts = known_colour_df["choice"].tolist()
gts

# %%
images = known_colour_df["image"].tolist()
images

# %%
colour_classes = ['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'orange', 'brown', 'black', 'white', 'grey']
print(colour_classes)

# %%
all_colours = []

# loop throught all the colours
for i, colour in enumerate(colour_classes):
    # make pair of the current colour with all the colours after it
    colour_pair = [colour + " " + c for c in colour_classes[i+1:]]
    all_colours += colour_pair

all_colours += colour_classes + ["many colours"]

# %%
len(all_colours)

# %%
model = FashionCLIP("fashion-clip")

# %%
# preds = model.zero_shot_classification(images, colour_classes)
preds = model.zero_shot_classification(images, all_colours)

# %%
accuracy_score(gts, preds)

# %%
error_inds = [idx for idx in range(len(gts)) if preds[idx] != gts[idx]]
print(len(error_inds))
error_inds

# %%
error_imgs = [images[idx] for idx in error_inds]
compare_pred_gts = [f"Predict: {preds[idx]}\nGroundtruth: {gts[idx]}" for idx in error_inds]

# %%
for i in range(0, len(error_imgs), 25):
    plot.display_multiple_images(
        error_imgs[i:i+25],
        5,
        24,
        512,
        compare_pred_gts[i:i+25],
        10,
        axes_pad=0.6,
    )

# %% [markdown]
# ### Unknown Colours

# %%
unknown_colour_df = label_df[label_df["choice"] == "unknown"]
print(len(unknown_colour_df))
unknown_colour_df.head()

# %%
hard_images = unknown_colour_df["image"].tolist()
len(hard_images)

# %%
plot.display_multiple_images(
    hard_images,
    4,
    24,
    512,
    fontsize=10,
    axes_pad=0.6,
)



# %%
preds = model.zero_shot_classification(hard_images, all_colours)

# %%
plot.display_multiple_images(
    hard_images,
    4,
    24,
    512,
    preds,
    10,
    axes_pad=0.6,
)

# %%
