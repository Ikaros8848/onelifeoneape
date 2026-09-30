"""Stage 4 robustness evaluation for QAT and learned 13-tap checkpoints."""
import csv, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.evaluate import metrics
from baseline.hardware_aware import HardwareAwareCNN
from tests.test_mnist import TestMNIST

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

DATA = ROOT / "data"
OUT = ROOT / "results"
FIG = ROOT / "figures"
MODELS = {
    "QAT": ROOT / "artifacts/checkpoints/mnist_scamp_qat.pt",
    "13-tap seed 7": ROOT / "artifacts/checkpoints/mnist_scamp_offset13.deploy.pt",
    "13-tap seed 42": ROOT / "artifacts/checkpoints/mnist_scamp_offset13_seed42.deploy.pt",
}
NOISE_SEED = 1729

def load_inputs():
    ds = TestMNIST(str(DATA))
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))])
    return xs, ys

def shift(x, dy, dx):
    """Zero-fill translation on the 64x64 model input plane."""
    out = torch.zeros_like(x)
    sy0, sy1 = max(0, -dy), min(x.shape[-2], x.shape[-2] - dy)
    sx0, sx1 = max(0, -dx), min(x.shape[-1], x.shape[-1] - dx)
    ty0, ty1 = max(0, dy), min(x.shape[-2], x.shape[-2] + dy)
    tx0, tx1 = max(0, dx), min(x.shape[-1], x.shape[-1] + dx)
    out[..., ty0:ty1, tx0:tx1] = x[..., sy0:sy1, sx0:sx1]
    return out

def evaluate(state, x, y, device, dynamic_quant=False):
    if dynamic_quant:
        model = HardwareAwareCNN(quantize=True, threshold_ratio=0.05, per_channel=True)
        model.load_state_dict(state); model.to(device).eval()
        preds=[]
        with torch.inference_mode():
            for batch in DataLoader(torch.utils.data.TensorDataset(x), batch_size=256):
                preds.extend(model(batch[0].to(device)).argmax(1).cpu().tolist())
        return metrics(y.tolist(), preds)
    conv_w, conv_b = state["conv_weight"].to(device), state["conv_bias"].to(device)
    fc_w, fc_b = state["fc_weight"].to(device), state["fc_bias"].to(device)
    preds = []
    with torch.inference_mode():
        for batch in DataLoader(torch.utils.data.TensorDataset(x), batch_size=256):
            z = F.conv2d(F.pad(batch[0].to(device), (1, 2, 1, 2)), conv_w, conv_b)
            z = F.max_pool2d(F.relu(z), 4, 4).flatten(1)
            logits = F.linear(z, fc_w, fc_b)
            preds.extend(logits.argmax(1).cpu().tolist())
    m = metrics(y.tolist(), preds)
    return m

def main():
    random.seed(NOISE_SEED); np.random.seed(NOISE_SEED); torch.manual_seed(NOISE_SEED)
    x, y = load_inputs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noises = {s: torch.randn(x.shape, generator=torch.Generator().manual_seed(NOISE_SEED + int(s*100))) * s for s in (0.05, 0.10, 0.20)}
    conditions = [("clean", "0", x)]
    conditions += [("gaussian", str(s), (x + noises[s]).clamp(0, 1)) for s in (0.05, 0.10, 0.20)]
    for n, (dy, dx) in {"left":(0,-1), "right":(0,1), "up":(-1,0), "down":(1,0)}.items():
        conditions.append(("translation_" + n, "1", shift(x, dy, dx)))
    for n, (dy, dx) in {"left":(0,-2), "right":(0,2), "up":(-2,0), "down":(2,0)}.items():
        conditions.append(("translation_" + n, "2", shift(x, dy, dx)))
    rows=[]; clean={}
    for model, path in MODELS.items():
        state=torch.load(path, map_location="cpu", weights_only=True)
        for corruption, severity, inputs in conditions:
            m=evaluate(state, inputs, y, device, dynamic_quant=(model == "QAT"))
            if corruption == "clean": clean[model]=m
            rows.append({"model":model,"seed":NOISE_SEED,"corruption":corruption,"severity":severity,
                         "accuracy":m["accuracy"],"macro_f1":m["macro_f1"],
                         "accuracy_drop":clean.get(model, m)["accuracy"]-m["accuracy"]})
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    with (OUT/"robustness.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"noise_seed":NOISE_SEED,"input_stage":"after FirmwareMNIST/mnist_to_64 preprocessing; model-input 64x64 normalized tensor",
             "device":str(device),"models":{k:{"checkpoint":str(v),"clean_accuracy":clean[k]["accuracy"],"clean_macro_f1":clean[k]["macro_f1"]} for k,v in MODELS.items()},
             "rows":len(rows)}
    (OUT/"robustness_metadata.json").write_text(json.dumps(summary,indent=2), encoding="utf-8")
    if plt is not None:
        fig, ax=plt.subplots(figsize=(8,5))
        for model in MODELS:
            vals=[next(r["accuracy"] for r in rows if r["model"]==model and r["corruption"]=="gaussian" and r["severity"]==str(s)) for s in (0.05,0.10,0.20)]
            ax.plot([0.05,0.10,0.20], vals, marker="o", label=model)
        ax.set_xlabel("Gaussian noise sigma (model-input scale)"); ax.set_ylabel("Accuracy"); ax.set_ylim(0.5,1.0); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(FIG/"robustness_accuracy.png", dpi=160); plt.close(fig)
    else:
        from PIL import Image, ImageDraw
        im=Image.new("RGB", (960, 600), "white"); d=ImageDraw.Draw(im)
        d.text((40,20), "Robustness accuracy vs Gaussian noise sigma", fill="black")
        colors=["red","blue","green"]
        for idx, model in enumerate(MODELS):
            pts=[]
            for s in (0.05,0.10,0.20):
                val=next(r["accuracy"] for r in rows if r["model"]==model and r["corruption"]=="gaussian" and r["severity"]==str(s))
                pts.append((int(100+s*3000), int(560-(val-0.5)*700)))
            d.line(pts, fill=colors[idx], width=3)
            for p in pts: d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4), fill=colors[idx])
            d.text((650,40+idx*25), model, fill=colors[idx])
        d.text((60,570), "sigma: 0.05 -> 0.20; y-axis scaled 50%-100%", fill="black")
        im.save(FIG/"robustness_accuracy.png")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
