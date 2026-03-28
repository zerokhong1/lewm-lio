import torch
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))
from encoder_bev import BEVEncoder


def test_encoder_output_shape():
    enc = BEVEncoder(in_channels=4, latent_dim=192)
    x = torch.randn(2, 4, 256, 256)
    z = enc(x)
    assert z.shape == (2, 192), f"Expected (2, 192), got {z.shape}"


def test_encoder_param_count():
    enc = BEVEncoder(in_channels=4, latent_dim=192, base_channels=32)
    params = sum(p.numel() for p in enc.parameters()) / 1e6
    assert params < 15, f"Too many params: {params:.2f}M (limit 15M)"
    print(f"Params: {params:.2f}M")


if __name__ == "__main__":
    test_encoder_output_shape()
    test_encoder_param_count()
    print("All tests PASSED")
