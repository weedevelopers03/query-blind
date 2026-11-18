import os
import pathlib
import sys
import torch
import timm
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adapters.timm_vit import TimmViTAdapter
from extract.cache import extract_shard
from extract.verify_cache import verify_cache

def load_voc_images(image_dir, limit=None, patch_size=16, split=None, return_paths=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_paths = sorted(list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png")))
    split_file = Path(image_dir).parent / "ImageSets" / "Main" / f"{split}.txt" if split else None
    if split_file is not None and split_file.is_file():
        split_ids = {line.split()[0] for line in split_file.read_text().splitlines() if line.split()}
        img_paths = [p for p in img_paths if p.stem in split_ids]
        print(f"Selected {len(img_paths)} images listed in {split_file}")
    if limit:
        img_paths = img_paths[:limit]
        
    total_imgs = len(img_paths)
    if total_imgs == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")
        
    print(f"Found {total_imgs} images. Preprocessing...")
    tensors = []
    for idx, p in enumerate(img_paths, 1):
        img = Image.open(p).convert("RGB")
        image = transform(img)
        _, height, width = image.shape
        padded_height = ((height + patch_size - 1) // patch_size) * patch_size
        padded_width = ((width + patch_size - 1) // patch_size) * patch_size
        padded = torch.zeros((3, padded_height, padded_width), dtype=image.dtype)
        padded[:, :height, :width] = image
        tensors.append(padded)
        if idx % 500 == 0 or idx == total_imgs:
            print(f"  --> Preprocessed {idx}/{total_imgs} images")
            
    return (tensors, img_paths) if return_paths else tensors

def load_model_and_adapter(model_id):
    try:
        from tests.conftest import CHECKPOINTS
        if model_id in CHECKPOINTS:
            return CHECKPOINTS[model_id]()
    except Exception:
        pass

    timm_map = {
        "dino_vits16": "vit_small_patch16_224.dino",
        "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
        "deit3_small": "deit3_small_patch16_224",
    }
    timm_name = timm_map.get(model_id, model_id)
    adapter = TimmViTAdapter()
    model = timm.create_model(timm_name, pretrained=True, dynamic_img_size=True)
    adapter.prepare(model)
    return adapter, model

def run_extraction(model_id, image_dir, dataset_name="VOC07", split="val", out_root="/tmp/cache", limit=10):
    print(f"Loading checkpoint '{model_id}'...")
    adapter, model = load_model_and_adapter(model_id)
    
    print(f"Loading dataset images from '{image_dir}'...")
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        if patch_size[0] != patch_size[1]:
            raise ValueError(f"Non-square patch size is unsupported: {patch_size}")
        patch_size = patch_size[0]
    images, image_paths = load_voc_images(
        image_dir, limit=limit, patch_size=patch_size, split=split, return_paths=True
    )
    print(f"Loaded {len(images)} images; first padded image shape: {tuple(images[0].shape)}")
    
    d_model = getattr(model, "embed_dim", getattr(getattr(model, "config", None), "hidden_size", 384))
    
    print(f"Extracting features into {out_root}...")
    shard = extract_shard(
        adapter, model, images, model_id, dataset_name, split, out_root,
        d_model=d_model, image_ids=[path.stem for path in image_paths],
    )
    print(f"Successfully cached shard at: {shard}")
    
    print("Verifying cache integrity...")
    verify_cache(adapter, model, images, shard)
    print("Verification PASSED!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dino_vits16")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image folder")
    parser.add_argument("--dataset", type=str, default="VOC07")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--limit", type=int, default=10, help="Number of images")
    args = parser.parse_args()
    
    limit = None if args.limit <= 0 else args.limit
    run_extraction(args.model, args.image_dir, dataset_name=args.dataset, split=args.split, limit=limit)
