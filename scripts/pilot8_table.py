"""Guide §1.3 / §11.1: the 5-test x 8-pilot-model pass/fail table.

    cd repo && python scripts/pilot8_table.py [--only dino_vits16,mae_vitb16] [--out results/pilot8]

Writes table.md, table.json and meta.json (code git SHA if this is a git checkout, config hash, library
versions, which inputs were supplied) and exits 1 if any cell FAILS or any seeded bug goes undetected.

Optional inputs that turn SKIPs into real results (both need things this script cannot create):
    PILOT8_REAL_WEIGHTS=1     test 4b: real DINOv2-reg weights from timm / Hugging Face (needs internet)
    PILOT8_IMAGES_DIR=...     real photos for 4b (otherwise Gaussian noise, flagged in the output)
    PILOT8_CORLOC_CSV=...     test 5b: per-image CorLoc CSV of the closeout run (columns incl. `correct`)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # folder holding scripts/ and tests/pilot8_checks.py (need not be the repo)


def _find_repo():
    # first folder that holds adapters/timm_vit.py: $PILOT8_REPO, the working directory, this file's parent, then sys.path
    for c in [os.environ.get("PILOT8_REPO"), os.getcwd(), str(HERE)] + [p for p in sys.path if p]:
        if c and (Path(c) / "adapters" / "timm_vit.py").exists():
            return Path(c).resolve()
    sys.exit("adapters/timm_vit.py not found: run from the repo root, or set PILOT8_REPO=/path/to/repo")


ROOT = _find_repo()  # the repo whose adapters are tested; results go to ROOT/results/pilot8
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import timm  # noqa: E402

# load the checks by file path: a repo's own `tests` package would otherwise shadow tests/pilot8_checks.py
_spec = importlib.util.spec_from_file_location("pilot8_checks", HERE / "tests" / "pilot8_checks.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["pilot8_checks"] = C
_spec.loader.exec_module(C)

ROWS = [
    ("t1", "1 Attention reconstruction (max abs diff < 1e-4)"),
    ("t2", "2 Residual reconstruction (global token)"),
    ("t3", "3 Token layout (bright square)"),
    ("t4", "4 Register indexing"),
    ("t5", "5 Known-result replication"),
]
SHORT = {"dino_vits16": "DINO S/16", "deit3_vits16": "DeiT-III S/16", "ibot_vits16": "iBOT S/16",
         "mocov3_vitb16": "MoCo v3 B/16", "mae_vitb16": "MAE B/16", "dinov2_vitb14": "DINOv2 B/14",
         "dinov2_vitb14_reg": "DINOv2 B/14+reg", "openclip_vitb16": "OpenCLIP B/16"}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def git_state():
    """(HEAD sha, uncommitted files outside results/) or (None, None) if ROOT is not a git checkout.
    A SHA identifies the code that produced the table only when the working tree is clean."""
    try:
        g = ["git", "-C", str(ROOT)]
        sha = subprocess.check_output(g + ["rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        st = subprocess.check_output(g + ["status", "--porcelain", "--", "."], stderr=subprocess.DEVNULL).decode().splitlines()
        # outputs of runs and bytecode caches are not code under test
        st = [s for s in st if not {"results", "__pycache__"} & set(Path(s[3:].split(" -> ")[-1].strip('"')).parts)]
        return sha, st
    except Exception:
        return None, None


def cell_text(test, name, res, extra):
    if test == "t4" and name not in C.HAS_REGISTERS:
        return "n/a (no registers)"
    if test == "t5" and name not in C.KNOWN_RESULT:
        return "n/a (DINO only)"
    if test in ("t1", "t2", "t3"):
        c = res[test]
        if c.get("error"):
            return "FAIL (crashed)"
        m = [v["detected"] for v in c["mutants"].values() if v["detected"] is not None]
        tag = f" {sum(m)}/{len(m)}"
        return c["status"] + tag + ("" if all(m) else " !")
    if test == "t4":
        c, b = res["t4"], extra["t4b"]
        if c["status"] == "FAIL" or b["status"] == "FAIL":
            return "FAIL"
        return f"4a {c['status']} {sum(v['detected'] for v in c['mutants'].values())}/{len(c['mutants'])}; 4b {b['status']}"
    if test == "t5":
        return f"5a {extra['t5a']['status']}; 5b {extra['t5b']['status']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot8"))
    a = ap.parse_args()
    names = [n for n in C.PILOT8 if not a.only or n in a.only.split(",")]
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"repo under test: {ROOT}\nchecks loaded from: {C.__file__}", flush=True)

    cache, results, extras, t0, same = {}, {}, {}, time.time(), {}
    for n in names:
        k = C.arch_key(n)
        if k not in cache:
            print(f"[{time.time() - t0:6.0f}s] running architecture of {n} ...", flush=True)
            try:
                cache[k], same[k] = C.run_arch(n), n
            except Exception as e:  # e.g. a library upgrade breaks one model: record it as FAIL, keep the other columns
                err = f"{type(e).__name__}: {e}"
                print(f"  !! checks for {n} CRASHED: {err}", flush=True)
                traceback.print_exc()
                cache[k] = {t: dict(status="FAIL", mutants={}, error=err)
                            for t in ("t1", "t2", "t3") + (("t4",) if n in C.HAS_REGISTERS else ())}
                same[k] = n
        results[n], extras[n] = cache[k], C.run_real_weight_checks(n)

    lines = ["| Test | " + " | ".join(SHORT[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for t, label in ROWS:
        lines.append(f"| {label} | " + " | ".join(cell_text(t, n, results[n], extras[n]) for n in names) + " |")
    table = "\n".join(lines)

    bad = []
    for n in names:
        for t in ("t1", "t2", "t3", "t4"):
            c = results[n].get(t)
            if c and (c["status"] == "FAIL" or any(v["detected"] is False for v in c["mutants"].values())):
                bad.append((n, t))
        for t, c in extras[n].items():
            if c["status"] == "FAIL":
                bad.append((n, t))

    skipped = [(n, t, c["reason"]) for n in names for t, c in extras[n].items() if c["status"] == "SKIP"]
    shared = {}
    for n in names:
        shared.setdefault(C.arch_key(n), []).append(n)
    legend = (
        "Cell = status, then `seeded bugs caught / seeded bugs applicable` (`!` = a seeded bug was NOT caught, so that PASS "
        "is not trustworthy).  `n/a` = the test is defined for that model's absence of the feature, not skipped.\n"
        "Basis: tests 1-3 and 4a run on a randomly-initialised model of the exact architecture (default init AND a boosted "
        "init), not on the trained checkpoint; they do not check that the checkpoint file loads into that architecture.\n"
        "Models sharing an architecture were run once: " + "; ".join(" = ".join(v) for v in shared.values() if len(v) > 1) + "."
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [ROOT / "adapters" / f for f in ("base.py", "timm_vit.py", "open_clip_vit.py")] + [Path(C.__file__)]

    def label(f):
        try:
            return str(f.relative_to(ROOT))
        except ValueError:
            return str(f)

    try:
        oc_version = __import__("open_clip").__version__
    except Exception:
        oc_version = None
    cfg = dict(pilot8=C.PILOT8, tol=C.TOL, boost=C.BOOST, seed=C.SEED, corloc=[C.PUBLISHED_CORLOC, C.CORLOC_TOL, C.VOC07_TRAINVAL])
    sha, uncommitted = git_state()
    meta = dict(
        git_sha=sha, git_dirty=None if uncommitted is None else bool(uncommitted), git_uncommitted=(uncommitted or [])[:30],
        config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16],
        repo_root=str(ROOT), file_sha256_16={label(f): sha256(f) for f in files if f.exists()},
        versions=dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__, open_clip=oc_version),
        device="cpu", inputs=dict(real_weights=os.environ.get("PILOT8_REAL_WEIGHTS"), images_dir=os.environ.get("PILOT8_IMAGES_DIR"),
                                  corloc_csv=os.environ.get("PILOT8_CORLOC_CSV")),
        seconds=round(time.time() - t0), unix_time=int(time.time()),
    )
    (out / "table.md").write_text(table + "\n\n" + legend + "\n")
    (out / "table.json").write_text(json.dumps(dict(results=results, extras=extras, skipped=skipped, failures=bad), indent=1, default=str))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(table + "\n\n" + legend)
    print("\nSKIPPED:", *[f"\n  {n}: {t}: {r}" for n, t, r in skipped])
    ran = [f"\n  {n} {t}: {c}" for n in names for t, c in extras[n].items() if c["status"] != "SKIP"]
    print("\nCHECKS ON REAL INPUTS (4b / 5b) THAT RAN:", *(ran or ["none (all skipped)"]))
    print("\nFAILURES / UNDETECTED SEEDED BUGS:", bad or "none")
    for n in names:
        if results[n].get("t1", {}).get("error"):
            print(f"  {n} crashed: {results[n]['t1']['error']}")
    print(f"\nwritten to {out}  (git_sha={meta['git_sha']}, git_dirty={meta['git_dirty']}, config_hash={meta['config_hash']})")
    if meta["git_dirty"]:
        print("NOTE: uncommitted changes outside results/ (listed in meta.json): the SHA does not identify the code of this run; "
              "commit them and re-run before citing this table.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
