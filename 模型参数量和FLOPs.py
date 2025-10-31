import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count
import json

def calculate_model_complexity(model, input_size=(1, 3, 32, 32)):

    device = next(model.parameters()).device
    model.eval()
    input_tensor = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"  参数量: {params / 1e6:.2f} M")
    print(f"  FLOPs: {flops / 1e9:.2f} G")

    return

from ViT import *
from ViT_PVT import *
from AViT import *
if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 使用配置
    mode = config['come']['mode']
    name = config['model']['name']
    hyper = config['hyper']
    hyper_converted = {
        k: int(v) if isinstance(v, (int, float)) and k != 'dropout' else float(v) if k == 'dropout' else v
        for k, v in hyper.items()
    }
    if name == "ViT":
        model = ViT(
            patch_size=hyper_converted['patch_size'],
            embed_dim=hyper_converted['embed_dim'],
            num_heads=hyper_converted['num_heads'],
            max_seq_length=hyper_converted['max_seq_length'],
            encoder_num=hyper_converted['encoder_num'],
            dropout=hyper_converted['dropout'],
            num_classes=hyper_converted['num_classes']
        )
    if name == "ViT_PVT":
        model = ViT_PVT(
            patch_size=hyper_converted['patch_size'],
            embed_dim=hyper_converted['embed_dim'],
            num_heads=hyper_converted['num_heads'],
            max_seq_length=hyper_converted['max_seq_length'],
            encoder_num=hyper_converted['encoder_num'],
            dropout=hyper_converted['dropout'],
            num_classes=hyper_converted['num_classes']
        )
    if name == "AViT":
        model = AViT(
            patch_size=hyper_converted['patch_size'],
            embed_dim=hyper_converted['embed_dim'],
            num_heads=hyper_converted['num_heads'],
            max_seq_length=hyper_converted['max_seq_length'],
            encoder_num=hyper_converted['encoder_num'],
            dropout=hyper_converted['dropout'],
            num_classes=hyper_converted['num_classes']
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    calculate_model_complexity(model)