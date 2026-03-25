import argparse
import os

import torch

from models import get_model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Head Pose Estimation Model ONNX Export")

    parser.add_argument(
        "-w",
        "--weight",
        default="resnet18.pt",
        type=str,
        help="Trained state_dict file path to open",
    )
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"],
        help="Backbone network architecture to use",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch size for ONNX export",
    )

    return parser.parse_args()


@torch.no_grad()
def onnx_export(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(params.network, num_classes=6, pretrained=False)
    model.to(device)

    state_dict = torch.load(params.weight, map_location=device)
    model.load_state_dict(state_dict)
    print("Head pose model loaded successfully!")

    model.eval()

    fname = os.path.splitext(os.path.basename(params.weight))[0]
    onnx_model = f"{fname}.onnx"
    print(f"==> Exporting model to ONNX format at '{onnx_model}'")

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    dynamic_axes = None
    if params.dynamic:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "rotation_matrix": {0: "batch_size"},
        }
        print("Exporting model with dynamic batch size.")
    else:
        print("Exporting model with fixed input size: (1, 3, 224, 224)")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["rotation_matrix"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Model exported successfully to {onnx_model}")


if __name__ == "__main__":
    args = parse_arguments()
    onnx_export(args)
