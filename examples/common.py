"""What the checkpoint-driven examples share: the repo root on sys.path and network loading."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.network import DomibotNet, get_device  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "domibot2" / "domibot2.4.pt"
GAME_LOGS_DIR = ROOT / "game_logs"


def load_network(path: str | Path, gpu: bool = False) -> tuple[DomibotNet, torch.device]:
    """A checkpoint in eval mode, on CUDA only with `gpu` (so it doesn't compete with training)."""
    if not Path(path).exists():
        raise SystemExit(f"no checkpoint at {path} -- download domibot2.4.pt from "
                         f"https://github.com/rafxrs/domibot/releases into checkpoints/domibot2/ "
                         f"(see README Setup), or train your own")
    device = get_device() if gpu else torch.device("cpu")
    network = DomibotNet.load(path, map_location=device).to(device)
    network.eval()
    return network, device
