"""Controlled annotation (label) noise for FedIN-R3EDAR experiments."""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Mask ke labels. Agar tumhare mask mai ET = 4 hai to "et": 4 likho (aur neeche bhi 3 ki jagah 4)
REGIONS = {"ncr": 1, "ed": 2, "et": 3}
UNDER = {1: 2, 2: 0, 3: 2}     # under-segmentation: khoye hue voxels kis label mai jayen
CONFUSE = {1: 2, 2: 1, 3: 1}   # confusion: kis label se mix hon


def client_noise_plan(partition_id, num_partitions, rc):
    """Har region ka noise level. None matlab ye client clean hai."""
    if not rc.get("noise-enabled", False):
        return None
    seed = int(rc["noise-seed"])
    rng = np.random.default_rng(seed)
    n_noisy = int(round(num_partitions * rc["noise-client-fraction"]))
    if partition_id not in rng.choice(num_partitions, n_noisy, replace=False):
        return None
    r = np.random.default_rng(seed + 1000 + partition_id)
    plan = {
        n: float(r.choice([0.0, rc["noise-level-low"], rc["noise-level-high"]]))
        for n in REGIONS
    }
    os.makedirs("artifacts", exist_ok=True)
    with open(f"artifacts/noise_plan_{partition_id}.json", "w") as f:
        json.dump(plan, f)
    return plan


def corrupt_mask(mask, plan, seed):
    """mask: (D,H,W) integer tensor."""
    g = torch.Generator().manual_seed(int(seed))
    out = mask.clone()
    for name, lab in REGIONS.items():
        p = plan[name]
        if p <= 0:
            continue
        m = (mask == lab).float()[None, None]
        kind = int(torch.randint(0, 3, (1,), generator=g))
        if kind == 0:      # boundary perturbation
            dil = F.max_pool3d(m, 3, 1, 1)
            ero = -F.max_pool3d(-m, 3, 1, 1)
            shell = (dil - ero) > 0
            flip = shell & (torch.rand(m.shape, generator=g) < p)
            added, removed = (flip & (m == 0))[0, 0], (flip & (m == 1))[0, 0]
            out[added & (out == 0)] = lab
            out[removed] = 0
        elif kind == 1:    # under-segmentation
            e = m
            for _ in range(max(1, round(p * 4))):
                e = -F.max_pool3d(-e, 3, 1, 1)
            lost = ((m > 0) & (e == 0))[0, 0]
            out[lost] = UNDER[lab]
        else:              # region confusion
            sel = (m > 0)[0, 0] & (torch.rand(mask.shape, generator=g) < p)
            out[sel] = CONFUSE[lab]
    return out


class NoisyLabelDataset(Dataset):
    def __init__(self, base, plan, seed, partition_id):
        self.base, self.plan, self.seed, self.pid = base, plan, seed, partition_id

    def __len__(self):
        return len(self.base)

    def _noisy(self, item, s):
        item = dict(item)
        t = torch.as_tensor(item["label"])
        has_ch = t.dim() == 4
        m = corrupt_mask((t[0] if has_ch else t).long(), self.plan, s)
        item["label"] = (m.unsqueeze(0) if has_ch else m).to(t.dtype)
        return item

    def __getitem__(self, idx):
        item = self.base[idx]
        s = self.seed + self.pid * 100003 + idx
        if isinstance(item, list):
            return [self._noisy(it, s + i) for i, it in enumerate(item)]
        return self._noisy(item, s)


def apply_label_noise(out, partition_id, rc):
    """load_partition ka output lo; noise ON ho to train loader wrap karo, warna wahi wapis."""
    num_clients = int(rc.get("num-clients", rc.get("num_clients", 0)))
    plan = client_noise_plan(partition_id, num_clients, rc)
    if plan is None:
        return out

    def wrap(dl):
        return DataLoader(
            NoisyLabelDataset(dl.dataset, plan, int(rc["noise-seed"]), partition_id),
            batch_size=dl.batch_size,
            shuffle=True,
            num_workers=dl.num_workers,
            collate_fn=dl.collate_fn,
            pin_memory=dl.pin_memory,
            drop_last=getattr(dl, "drop_last", False),
        )

    if isinstance(out, tuple):          # (train, val, ...) ho to sirf train noisy
        return (wrap(out[0]),) + tuple(out[1:])
    return wrap(out)