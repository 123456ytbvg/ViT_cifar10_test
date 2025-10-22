import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count


def calculate_model_complexity(model, input_size=(1, 3, 32, 32)):

    device = next(model.parameters()).device
    model.eval()
    input_tensor = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"  参数量: {params / 1e6:.2f} M")
    print(f"  FLOPs: {flops / 1e9:.2f} G")

    return

from ViT import *

if __name__ == "__main__":
    model = Vit(
        patch_size=8,
        embed_dim=384,
        num_heads=6,
        encoder_num=6,
        num_classes=10
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    calculate_model_complexity(model)