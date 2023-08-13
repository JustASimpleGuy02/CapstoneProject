import os
import os.path as osp
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.custom_text_image_dm import TextImageDataModule
from clip import *


def main(hparams):
    config_dir = (
        "train_CLIP/clip/configs/ViT.yaml"
        if "ViT" in hparams.model_name
        else "train_CLIP/clip/configs/RN.yaml"
    )

    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    ### Model
    model_name = hparams.model_name
    model, preprocess = load(model_name)
    model = CLIPWrapper(model_name, model, config, hparams.minibatch_size)
    del hparams.model_name, model_name
    
    ### Data Loader
    # dm = TextImageDataModule(preprocess=preprocess)
    dm = TextImageDataModule.from_argparse_args(hparams)
    dm.preprocess = preprocess
    
    ### Trainer
    trainer = Trainer.from_argparse_args(
        hparams,
        # precision=16,
        max_epochs=32,
    )
    trainer.fit(model, dm)
    trainer.save_checkpoint("best_model.ckpt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--minibatch_size", type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
