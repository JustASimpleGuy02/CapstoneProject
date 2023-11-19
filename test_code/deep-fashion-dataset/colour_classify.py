# %%
import sys
import os
import os.path as osp
from glob import glob
import random
import importlib

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
from reproducible_code.tools import image_io, io, plot

sys.path.append("../")
from apis import search_fclip

importlib.reload(image_io)
importlib.reload(search_fclip)

# %%
project_dir = "/home/dungmaster/Projects/Machine Learning/HangerAI_outfits_recommendation_system/CapstoneProject"
embeddings_dir = osp.join(project_dir, "model_embeddings")
os.listdir(embeddings_dir)

# %%
# data_dir = "/home/dungmaster/Datasets/Deep-Fashion"
# img_dir = osp.join(data_dir, "img")
img_dir = osp.join(project_dir, "data", "deep-fashion-dataset", "test_images")
img_paths = glob(osp.join(img_dir, "*.jpg"))
len(img_paths)

# %%
embeddings_file = osp.join(embeddings_dir, "colour_classes.txt")
class_embeddings = None
save_embeddings = True

if osp.exists(embeddings_file):
    class_embeddings = np.loadtxt(embeddings_file)
    save_embeddings = False

# %%
colour_classes = io.load_txt(
#     osp.join(project_dir, "data", "deep-fashion-dataset", "colours.txt"
    osp.join(project_dir, "data", "deep-fashion-dataset", "colours_v2.txt")
)
print(colour_classes)

# %%
ret = search_fclip.FashionRetrieval(text_embeddings=class_embeddings)

# %%
img_path = random.sample(img_paths, 1)[0]
img = image_io.load_image(img_path, backend="pil", to_array=False)
img

# %%
%%time
rank_classes = ret.classify(
    img,
    colour_classes,
    embeddings_file=embeddings_file,
    save_embeddings=save_embeddings
)
rank_classes

# %%
save_embeddings

# %%
sample_img_paths = ['/content/drive/MyDrive/Datasets/deep-fashion/5521_1355_27859498548.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/11935_3982_34556420377.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/10797_9715_33019325935.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/1157_9717_1557864343.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/9302_5257_30546981643.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/10188_6914_25525280972.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/7358_5257_25567450666.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/1379_9724_16844473419.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/11907_1356_35378641093.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/5884_1349_28168714905.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/7819_9715_29873818961.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/13202_9736_31561291203.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/5294_12004_26804793056.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/13016_9772_30688740020.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/6673_5257_19212949044.jpg', '/content/drive/MyDrive/Datasets/deep-fashion/12328_9719_35135908179.jpg']
sample_img_paths = [osp.join(img_dir, osp.basename(p)) for p in sample_img_paths]
sample_img_paths

# %%
%%time
# sample_img_paths = random.sample(img_paths, 16)
# imgs = [image_io.load_image(p) for p in sample_img_paths]
classes = []

for path in sample_img_paths:
    rank_classes = ret.classify(
        path,
        colour_classes,
        embeddings_file=embeddings_file,
        save_embeddings=save_embeddings
    )[:3]
    classes.append(str(rank_classes))

# %%
plot.display_multiple_images(
    sample_img_paths, 4, 24, 512, classes, 10
)

# %%
model = FashionCLIP("fashion-clip")

# %%
classes = model.zero_shot_classification(sample_img_paths, colour_classes)
plot.display_multiple_images(
    sample_img_paths, 4, 24, 512, classes, 10
)
 
# %%
