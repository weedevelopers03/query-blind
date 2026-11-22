"""
Weight-conversion drift check: timm's DINO ViT-S/16 checkpoint vs. the
official facebookresearch/dino torch.hub checkpoint.

Why: the native-resolution LOST extraction fix brought VOC07 CorLoc from
58.19% to 60.73%, closing most of the gap to the official 61.9% benchmark
but landing 0.17pt short of the pre-registered +-1.0pt tolerance band
(needs >=60.9%). Reproducibility was already confirmed (identical 60.73%
on a second run), so the residual gap is not noise. This script tests one
remaining candidate explanation: that timm's packaged "vit_small_patch16_224.dino"
weights are not numerically identical to the original facebookresearch/dino
checkpoint (a lossy or incomplete conversion somewhere in timm's hub pipeline).

Run this somewhere both HuggingFace Hub and github.com/facebookresearch are
reachable (your Mac, Kaggle, or Colab) - it needs to download both checkpoints
(a few hundred MB total). It could NOT be run inside the Claude sandbox that
wrote it: huggingface.co is explicitly blocked by that sandbox's network
policy (confirmed via a 403 from its egress proxy), and the official
checkpoint's host is almost certainly blocked too.

Usage:
    pip install timm torch --break-system-packages   # if not already installed
    python3 compare_dino_weights.py

Read the output in two parts:
  1. The printed key lists from both checkpoints (first 15 of each). The
     mapping below (to_official_key) assumes the two use IDENTICAL key
     names, which is likely true (timm's ViT naming was built to line up
     with this exact family of checkpoints) but is NOT verified from inside
     the sandbox that wrote this script - compare the two printed lists
     yourself before trusting the diff numbers below them.
  2. If "missing_in_official" or "shape_mismatches" come back empty, and
     the "bit-for-bit identical" count is at or near the total tensor
     count, the two checkpoints are (numerically) the same model and
     conversion drift is ruled out as the cause of the 0.17pt gap - the
     remaining gap would need a different explanation. Any tensors with a
     large max_abs_diff point at exactly which layer diverges.
"""
import torch
import timm


def load_timm():
    model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    model.eval()
    return model.state_dict()


def load_official():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()
    return model.state_dict()


def to_official_key(timm_key):
    # Identity mapping: timm's ViT module names (patch_embed.proj, blocks.N.*,
    # cls_token, pos_embed, norm.*) were modeled on this same checkpoint
    # family, so the two should already line up key-for-key. If
    # "missing_in_official" below is non-empty, compare the two printed key
    # lists above it and patch this function (e.g. a prefix difference, or
    # the official model carrying a head/projection timm drops).
    return timm_key


def main():
    print("Loading timm checkpoint (vit_small_patch16_224.dino)...")
    timm_sd = load_timm()
    print(f"  {len(timm_sd)} tensors")

    print("Loading official checkpoint (facebookresearch/dino, dino_vits16)...")
    official_sd = load_official()
    print(f"  {len(official_sd)} tensors")

    print("\n--- timm keys (first 15) ---")
    for k in list(timm_sd.keys())[:15]:
        print(" ", k, tuple(timm_sd[k].shape))

    print("\n--- official keys (first 15) ---")
    for k in list(official_sd.keys())[:15]:
        print(" ", k, tuple(official_sd[k].shape))

    missing_in_official = []
    shape_mismatches = []
    value_diffs = []

    for tk, tv in timm_sd.items():
        ok = to_official_key(tk)
        if ok not in official_sd:
            missing_in_official.append((tk, ok))
            continue
        ov = official_sd[ok]
        if tuple(tv.shape) != tuple(ov.shape):
            shape_mismatches.append((tk, tuple(tv.shape), tuple(ov.shape)))
            continue
        max_abs_diff = (tv.float() - ov.float()).abs().max().item()
        is_identical = torch.equal(tv, ov)
        value_diffs.append((tk, max_abs_diff, is_identical))

    print(f"\n{len(missing_in_official)} timm keys had no match in the official "
          f"checkpoint (fix to_official_key if this is non-zero):")
    for tk, ok in missing_in_official[:20]:
        print(f"  {tk}  ->  {ok}")

    print(f"\n{len(shape_mismatches)} shape mismatches:")
    for tk, ts, os_ in shape_mismatches[:20]:
        print(f"  {tk}: timm={ts} official={os_}")

    identical_count = sum(1 for _, _, ident in value_diffs if ident)
    print(f"\n{identical_count}/{len(value_diffs)} tensors bit-for-bit identical "
          f"({len(timm_sd)} total timm tensors)")

    worst = sorted(value_diffs, key=lambda x: -x[1])[:15]
    print("\nLargest max-abs-diff tensors (0.0 = identical):")
    for k, d, ident in worst:
        print(f"  {k}: max_abs_diff={d:.6g}  identical={ident}")

    print("\nVerdict:")
    if not missing_in_official and not shape_mismatches and identical_count == len(value_diffs):
        print("  Checkpoints are bit-for-bit identical. Conversion drift is RULED OUT "
              "as the source of the 0.17pt CorLoc gap - look elsewhere (box-rescaling, "
              "metric upsampling nearest-vs-bilinear, or a remaining pipeline difference).")
    elif not missing_in_official and not shape_mismatches:
        print("  Same architecture and shapes, but some tensors differ numerically - "
              "see the largest max_abs_diff entries above. This IS a candidate "
              "explanation for the gap; worth checking which layer(s) diverge most.")
    else:
        print("  Key or shape mismatches found - the mapping likely needs a manual fix "
              "(see the two printed key lists above) before the diff numbers mean anything.")


if __name__ == "__main__":
    main()
