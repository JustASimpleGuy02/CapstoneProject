# %%
import os
import os.path as osp

import pandas as pd
from reproducible_code.tools import io, plot, image_io

# %%
data_dir = "/home/dungmaster/Datasets/H&M"
df = io.load_csv(
    osp.join(data_dir, "articles.csv")
)
print(len(df))
df.head()

# %%
df.columns

# %%
colours = df["colour_group_name"].unique().tolist()
colours

# %%
io.save_txt(colours, "../data/deep-fashion-dataset/colours.txt")

# %%
plot.plot_attribute_frequency(df, "colour_group_name", 10, 30)

# %%
