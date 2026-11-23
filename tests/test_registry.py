"""Registry completeness for the pilot 8 (supervisor's Review of Progress (2): epochs, pretrain images and
registers feed the §10.3 confound regression, so a missing value means the model cannot enter the analysis).

    pytest tests/test_registry.py                          # fails while any pilot-8 row is incomplete
    python -m tests.test_registry [configs/checkpoints.yaml]   # prints the status table and the list of
                                                              # 'fully confirmed' rows that are / are not pilot-8

A row is COMPLETE only if: it exists, epochs and pretrain_images_m are numbers > 0, registers is an int >= 0
(0 counts as filled), params_m / depth / pretrain_data / license are present, and it carries no `verify: true`.
Edit PILOT8 below if your registry uses different ids.
"""
import sys
from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "checkpoints.yaml"
PILOT8 = {  # pilot model (Guide §11.1) -> acceptable registry ids
    "DINO ViT-S/16": ["dino_vits16"],
    "DeiT-III ViT-S/16": ["deit3_small_patch16_224", "deit3_vits16"],
    "iBOT ViT-S/16": ["ibot_vits16"],
    "MoCo v3 ViT-B/16": ["mocov3_vitb16"],
    "MAE ViT-B/16": ["mae_vitb16"],
    "DINOv2 ViT-B/14": ["dinov2_vitb14"],
    "DINOv2 ViT-B/14 + registers": ["dinov2_vitb14_reg"],
    "OpenCLIP ViT-B/16": ["openclip_vitb16"],
}
REQUIRED = ["params_m", "depth", "pretrain_data", "license"]


def load(path=REGISTRY):
    rows = yaml.safe_load(Path(path).read_text())
    return {r["id"]: r for r in rows}


def problems(row):
    p = []
    for k in ("epochs", "pretrain_images_m"):
        if not isinstance(row.get(k), (int, float)) or isinstance(row.get(k), bool) or row[k] <= 0:
            p.append(f"{k} missing/invalid ({row.get(k)!r})")
    if not isinstance(row.get("registers"), int) or isinstance(row.get("registers"), bool) or row["registers"] < 0:
        p.append(f"registers missing/invalid ({row.get('registers')!r})")
    p += [f"{k} missing" for k in REQUIRED if row.get(k) in (None, "")]
    if row.get("verify"):
        p.append("verify: true (not confirmed)")
    return p


def status(path=REGISTRY):
    reg, out = load(path), {}
    for name, ids in PILOT8.items():
        hit = next((i for i in ids if i in reg), None)
        out[name] = (hit, ["no row with id " + " / ".join(ids)] if hit is None else problems(reg[hit]))
    return reg, out


EXPANSION4 = {
    "DINOv2 ViT-S/14": ["dinov2_vits14"],
    "DINOv2 ViT-S/14 + registers": ["dinov2_vits14_reg"],
    "DeiT-III ViT-S/16 (IN21k)": ["deit3_small_patch16_224_in21k"],
    "MetaCLIP ViT-B/16": ["metaclip_vitb16"],
}


def test_pilot8_registry_complete():
    _, st = status()
    bad = {n: p for n, (_, p) in st.items() if p}
    assert not bad, bad


def test_zoo12_registry_complete():
    reg = load()
    bad = {}
    for name, ids in {**PILOT8, **EXPANSION4}.items():
        hit = next((i for i in ids if i in reg), None)
        p = ["no row with id " + " / ".join(ids)] if hit is None else problems(reg[hit])
        if p:
            bad[name] = p
    assert not bad, bad


def report(path=REGISTRY):
    reg, st = status(path)
    print(f"registry: {path}  ({len(reg)} rows)\n")
    for name, (hit, p) in st.items():
        print(f"  {'COMPLETE ' if not p else 'INCOMPLETE'}  {name:30s} id={hit}")
        for x in p:
            print(f"                 - {x}")
    pilot_ids = {h for h, _ in st.values() if h}
    confirmed = [i for i, r in reg.items() if not problems(r)]
    print(f"\nrows with no problems (the 'fully confirmed' ones): {len(confirmed)}")
    for i in confirmed:
        print(f"  {i:35s} {'pilot-8' if i in pilot_ids else 'NOT in the pilot 8'}")
    return not any(p for _, p in st.values())


if __name__ == "__main__":
    sys.exit(0 if report(sys.argv[1] if len(sys.argv) > 1 else REGISTRY) else 1)
