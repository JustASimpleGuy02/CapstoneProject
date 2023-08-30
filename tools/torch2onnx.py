import sys
from argparse import ArgumentParser
from .model import *


def torch2onnx(model_path, model_name, device="cuda"):
    model, _ = load_model(model_path, model_name).to(device)
    model.eval()
    dummy_input_1 = torch.randn(1, 3, 224, 224).to(device)
    dummy_input_2 = torch.randint(100, (1, 77)).to(device)

    torch.onnx.export(
        model,
        args=(dummy_input_1, dummy_input_2),
        f="fclip.onnx",
        input_names=["input_image", "input_text"],
        output_names=["output"],
    )
    return model


def parse_args():
    parser = ArgumentParser(description="convert pytorch model to onnx format")
    parser.add_argument("model_path", help="path to checkpoint of model")
    parser.add_argument("model_name", help="model's name, RN or ViT")
    parser.add_argument("device", help="device used for inference", default="cpu")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model_path = args.model_path
    model_name = args.model_name
    device = args.device
    
    torch2onnx(model_path, model_name, device)
