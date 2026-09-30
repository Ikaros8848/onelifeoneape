"""Train offset-gated SCAMP-aware QAT without modifying baseline artifacts."""
import argparse, json, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.hardware_aware import ternary_quantize
from scripts.baseline.train_mnist import FirmwareMNIST, run_epoch
from optimized.offset_gated import OffsetGatedHardwareAwareCNN, export_deploy_state


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data'); p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gate-lr', type=float, default=2e-3); p.add_argument('--threshold-ratio', type=float, default=.05)
    p.add_argument('--target-retained-offsets', type=int, default=14); p.add_argument('--gate-lambda', type=float, default=.05)
    p.add_argument('--budget-lambda', type=float, default=.05); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--init', default='artifacts/checkpoints/mnist_scamp_qat.pt')
    p.add_argument('--output', default='artifacts/checkpoints/mnist_scamp_offset_gated.pt')
    p.add_argument('--max-train', type=int); p.add_argument('--max-test', type=int)
    a = p.parse_args(); seed_all(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, te = FirmwareMNIST(a.data_root, True), FirmwareMNIST(a.data_root, False)
    if a.max_train: tr.base.data, tr.base.targets = tr.base.data[:a.max_train], tr.base.targets[:a.max_train]
    if a.max_test: te.base.data, te.base.targets = te.base.data[:a.max_test], te.base.targets[:a.max_test]
    m = OffsetGatedHardwareAwareCNN(True, a.threshold_ratio, True,
                                    target_retained_offsets=a.target_retained_offsets,
                                    gate_mode='hard')
    if a.init:
        state = torch.load(a.init, map_location='cpu', weights_only=True)
        m.load_state_dict(state, strict=False)
    # Start from a deterministic saliency ranking so the first discrete
    # schedule drops low-contribution offsets, then let task gradients adapt it.
    with torch.no_grad():
        _, signs, alpha, _ = ternary_quantize(m.conv_weight, a.threshold_ratio, True)
        contribution = (signs * alpha).abs().sum(dim=(0, 1))
        centered = contribution - contribution.mean()
        m.offset_logits.copy_(centered / centered.std().clamp_min(1e-6))
    m = m.to(device)
    base_params = [v for k, v in m.named_parameters() if k != 'offset_logits']
    opt = torch.optim.Adam([{'params': base_params, 'lr': a.lr}, {'params': [m.offset_logits], 'lr': a.gate_lr}])
    train = DataLoader(tr, a.batch_size, shuffle=True); test = DataLoader(te, a.batch_size)
    history = []
    for epoch in range(1, a.epochs + 1):
        m.train(); total = 0.0; count = 0
        for images, labels in train:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(); logits = m(images)
            task = torch.nn.functional.cross_entropy(logits, labels)
            l1, budget = m.regularization()
            loss = task + a.gate_lambda * l1 + a.budget_lambda * budget
            loss.backward(); opt.step(); total += float(loss.detach()) * len(labels); count += len(labels)
        with torch.no_grad():
            m.eval(); test_row = run_epoch(m, test, None, device)
        row = {'epoch': epoch, 'loss': total / max(count, 1), 'test': test_row,
               'offset_probabilities': m.offset_probabilities().detach().cpu().tolist()}
        history.append(row); print(json.dumps(row))
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    torch.save(m.state_dict(), a.output)
    deploy_output = os.path.splitext(a.output)[0] + '.deploy.pt'
    torch.save(export_deploy_state(m), deploy_output)
    with torch.no_grad():
        _, info = m(torch.zeros(1, 1, 64, 64, device=device), True)
    report = {'method': 'offset_group_gating', 'device': str(device), 'seed': a.seed,
              'target_retained_offsets': a.target_retained_offsets, 'history': history,
              'checkpoint': a.output, 'deploy_checkpoint': deploy_output,
              'offset_probabilities': info['offset_probabilities'].cpu().tolist(),
              'note': 'SCAMP schedule proxy only; no chip latency measurement'}
    with open(os.path.splitext(a.output)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == '__main__': main()
