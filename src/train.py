"""
train.py — Training script cho LeWM-LiDAR

Chạy trên CONT (test nhỏ, --dummy_data) hoặc TEA (full training).
Logging: TensorBoard (không cần đăng nhập).
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from encoder_bev import BEVEncoder
from jepa_lidar import LeWMLiDAR, Predictor
from sigreg import SIGReg
from data_lidar import DummyLiDARBEVDataset, LiDARBEVDataset


def train(args):
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    )
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # —— Model ——
    encoder = BEVEncoder(
        in_channels=args.bev_channels,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
    )
    predictor = Predictor(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.pred_layers,
        dropout=args.dropout,
    )
    model = LeWMLiDAR(encoder, predictor).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {total_params:.2f}M")

    # —— Loss ——
    sigreg       = SIGReg(num_slices=args.sigreg_slices, num_points=args.sigreg_points)
    pred_loss_fn = nn.MSELoss()

    # —— Data ——
    if args.dummy_data:
        dataset = DummyLiDARBEVDataset(
            num_samples=args.dummy_samples,
            seq_length=args.seq_length,
            bev_channels=args.bev_channels,
            bev_size=args.bev_size,
            action_dim=args.action_dim,
        )
        print(f"Using dummy data: {len(dataset)} samples")
    else:
        assert args.data_path, "--data_path required when not using --dummy_data"
        dataset = LiDARBEVDataset(h5_path=args.data_path, seq_length=args.seq_length)
        print(f"Dataset: {args.data_path} ({len(dataset)} samples)")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    print(f"Batches per epoch: {len(dataloader)}")

    # —— Optimizer ——
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # —— Logging ——
    log_dir  = Path(args.log_dir);  log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"\n{'='*60}")
    print(f"  Training LeWM-LiDAR")
    print(f"  max_steps={args.max_steps}  batch={args.batch_size}  "
          f"lr={args.lr}  sigreg_l={args.sigreg_lambda}")
    print(f"{'='*60}\n")

    # —— Training loop ——
    global_step = 0
    best_loss   = float('inf')
    model.train()
    epoch = 0

    while global_step < args.max_steps:
        epoch += 1
        ep_total = ep_pred = ep_sigreg = 0.0
        n = 0

        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            obs = batch['observations'].to(device)   # (B, T, C, H, W)
            act = batch['actions'].to(device)         # (B, T-1, A)

            z_pred, z_tgt, z_all = model.forward_sequence(obs, act)

            loss_pred = pred_loss_fn(z_pred, z_tgt)

            B, T, D = z_all.shape
            loss_sigreg = sigreg(z_all.reshape(B * T, D))

            loss = loss_pred + args.sigreg_lambda * loss_sigreg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            ep_total   += loss.item()
            ep_pred    += loss_pred.item()
            ep_sigreg  += loss_sigreg.item()
            n          += 1

            if global_step % args.log_every == 0:
                writer.add_scalar('loss/total',   loss.item(),         global_step)
                writer.add_scalar('loss/pred',    loss_pred.item(),    global_step)
                writer.add_scalar('loss/sigreg',  loss_sigreg.item(),  global_step)
                writer.add_scalar('latent/z_mean', z_all.mean().item(), global_step)
                writer.add_scalar('latent/z_std',  z_all.std().item(),  global_step)

            if global_step % args.print_every == 0:
                print(f"[{global_step:>6d}] loss={loss.item():.4f}  "
                      f"pred={loss_pred.item():.4f}  "
                      f"sigreg={loss_sigreg.item():.6f}  "
                      f"z_std={z_all.std().item():.3f}")

            if global_step % args.save_every == 0:
                ckpt = ckpt_dir / f"step_{global_step}_weight.ckpt"
                torch.save(model.state_dict(), ckpt)
                avg = ep_total / n
                if avg < best_loss:
                    best_loss = avg
                    torch.save(model.state_dict(), ckpt_dir / "best_weight.ckpt")
                    print(f"  [BEST] checkpoint saved (loss={avg:.4f})")

        if n > 0:
            print(f"Epoch {epoch} | avg_loss={ep_total/n:.4f}  "
                  f"pred={ep_pred/n:.4f}  sigreg={ep_sigreg/n:.6f}")

    torch.save(model.state_dict(), ckpt_dir / "final_weight.ckpt")
    print(f"\nDone. Checkpoints: {ckpt_dir}")
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train LeWM-LiDAR")

    # Model
    p.add_argument("--bev_channels",  type=int,   default=4)
    p.add_argument("--bev_size",      type=int,   default=64)
    p.add_argument("--latent_dim",    type=int,   default=192)
    p.add_argument("--base_channels", type=int,   default=32)
    p.add_argument("--action_dim",    type=int,   default=3)
    p.add_argument("--hidden_dim",    type=int,   default=512)
    p.add_argument("--pred_layers",   type=int,   default=3)
    p.add_argument("--dropout",       type=float, default=0.1)

    # SIGReg
    p.add_argument("--sigreg_lambda",  type=float, default=1.0)
    p.add_argument("--sigreg_slices",  type=int,   default=256)
    p.add_argument("--sigreg_points",  type=int,   default=17)

    # Training
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--max_steps",     type=int,   default=50000)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--device",        type=str,   default="cuda")
    p.add_argument("--num_workers",   type=int,   default=0)

    # Data
    p.add_argument("--data_path",     type=str,   default="")
    p.add_argument("--seq_length",    type=int,   default=8)
    p.add_argument("--dummy_data",    action="store_true",
                   help="Dùng random data để test pipeline")
    p.add_argument("--dummy_samples", type=int,   default=500)

    # Logging
    p.add_argument("--log_dir",      type=str, default="outputs/logs")
    p.add_argument("--ckpt_dir",     type=str, default="outputs/checkpoints")
    p.add_argument("--log_every",    type=int, default=10)
    p.add_argument("--print_every",  type=int, default=50)
    p.add_argument("--save_every",   type=int, default=5000)

    train(p.parse_args())
