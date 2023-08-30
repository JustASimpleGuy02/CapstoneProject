import os
import os.path as osp
import sys
sys.path.append(osp.dirname(os.getcwd())) # append parent directory
from train_CLIP.clip import *

def load_model(model_path: str, model_name: str):
    model, preprocess = load(model_name)
    model = CLIPWrapper.load_from_checkpoint(
        model_path,
        model_name=model_name,
        model=model,
        minibatch_size=1,
    ).model 
    return model, preprocess
