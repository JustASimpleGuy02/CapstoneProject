# %%
import os.path as osp
from glob import glob
import random

from fashion_clip.fashion_clip import FashionCLIP
from reproducible_code.tools import plot

# %%
project_dir = "/home/dungmaster/Projects/Machine Learning/HangerAI_outfits_recommendation_system/CapstoneProject"
img_dir = osp.join(project_dir, "data", "deep-fashion-dataset", "test_images")
img_paths = glob(osp.join(img_dir, "*.jpg"))
len(img_paths)

# %%
colour_classes = ['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'orange', 'brown', 'black', 'white', 'grey']
print(colour_classes)

# %%
model = FashionCLIP("fashion-clip")

# %%
num_sample = 250
sample_img_paths = random.sample(img_paths, num_sample)

# %%
batch_25s = [sample_img_paths[i:i+25] for i in range(0, num_sample-1, 25)]
len(batch_25s)

# %%
idx = 0
batch_25s[idx] == sample_img_paths[idx*25:idx*25+25]

# %%
for i, batch in enumerate(batch_25s):
    print("Batch", i+1)
    # batch = batch_25s[idx]
    classes = model.zero_shot_classification(batch, colour_classes)
    plot.display_multiple_images(
        batch, 5, 24, 512, classes, 10
    )

# %%
