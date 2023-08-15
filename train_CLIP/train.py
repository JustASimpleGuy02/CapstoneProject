from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.custom_text_image_dm import TextImageDataModule
from tools import torch2onnx
from clip import *


def main(hparams):
    # config_dir = (
    #    "train_CLIP/clip/configs/ViT.yaml"
    #    if "ViT" in hparams.model_name
    #    else "train_CLIP/clip/configs/RN.yaml"
    # )

    # with open(config_dir) as fin:
    #    config = yaml.safe_load(fin)[hparams.model_name]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    ### Model
    model_name = hparams.model_name
    model, preprocess = load(model_name)
    trained_model = CLIPWrapper(model_name, model, hparams.minibatch_size)

    ### Data Loader
    dm = TextImageDataModule.from_argparse_args(hparams)
    dm.preprocess = preprocess

    ### Trainer
    trainer = Trainer.from_argparse_args(
        hparams,
        precision=16,
    )
    trainer.fit(trained_model, dm)
    torch2onnx(
        trainer.checkpoint_callback.best_model_path, model_name
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--minibatch_size", type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
