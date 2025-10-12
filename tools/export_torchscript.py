import argparse, torch, torch.nn as nn, torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pathlib import Path

def strip_module(state):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

class Embedder(nn.Module):
    """ResNet50 backbone + BNNeck -> L2-normalized 2048-d embedding"""
    def __init__(self, feat_dim=2048):
        super().__init__()
        from torchvision import models
        m = models.resnet50(weights=None)
        m.fc = nn.Identity()  # remove classification head
        self.backbone = m
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)

    def forward(self, x):
        f = self.backbone(x)           # (N, 2048)
        if f.dim() == 4:
            f = torch.flatten(f, 1)
        z = self.bnneck(f)
        return F.normalize(z, dim=1)

def remap_to_model_keys(state, model):
    """Adapt common naming schemes into model.backbone/* & bnneck/*."""
    state = strip_module(state)
    # drop classifier / head params if present
    state = {k:v for k,v in state.items()
             if not (k.startswith("classifier.") or k.startswith("head."))}
    bb_keys = set(model.backbone.state_dict().keys())
    mapped = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            mapped[k] = v; continue
        if k.startswith("base."):
            mapped["backbone."+k[len("base."):]] = v; continue
        if k.startswith("resnet."):
            mapped["backbone."+k[len("resnet."):]] = v; continue
        if k.startswith("model."):
            kk = k[len("model."):]
            if kk in bb_keys:
                mapped["backbone."+kk] = v; continue
        if k in bb_keys:
            mapped["backbone."+k] = v; continue
        if k.startswith("bnneck."):
            mapped[k] = v; continue
        # else ignore
    return mapped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pth")
    ap.add_argument("--out", required=True, help="Output .pt TorchScript path")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--strict", action="store_true", help="Strict state_dict load")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    dev = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    print(f"INFO device: {dev}")
    model = Embedder().to(dev).eval()

    mapped = remap_to_model_keys(state, model)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"INFO load: missing={len(missing)} unexpected={len(unexpected)} strict={args.strict}")
    if args.strict:
        model.load_state_dict(mapped, strict=True)

    # --- Export: prefer scripting; fallback to safe CPU tracing ---
    try:
        with torch.inference_mode():
            ts = torch.jit.script(model)
        print("INFO export: scripted model")
    except Exception as e:
        print("WARN export: scripting failed ->", e)
        print("INFO export: falling back to CPU trace (fp32, deterministic, no check)")
        cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        model_cpu = model.to('cpu').eval()
        x_cpu = torch.randn(1, 3, args.height, args.width)
        with torch.inference_mode():
            ts = torch.jit.trace(model_cpu, x_cpu, strict=False, check_trace=False)

    try:
        ts = torch.jit.optimize_for_inference(ts)
    except Exception:
        pass
    ts.save(str(out_path))
    print(f"Saved TorchScript to: {out_path}")

if __name__ == "__main__":
    main()
