import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from src.cache_gallery import GalleryCache, ManifestEntry, get_repo_version


def load_captions(path: Path) -> List[Dict[str, str]]:
    """支持 CSV/JSONL 格式的描述文件"""
    records: List[Dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    elif path.suffix.lower() == ".csv":
        import csv
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "image_id": row.get("image_id", row.get("path", "")),
                    "caption": row.get("caption", "")
                })
    else:
        raise ValueError(f"Unsupported caption file format: {path}")
    return records

class GalleryEncodeDataset(Dataset):
    def __init__(self, items: Sequence[Dict[str, object]]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        info = self.items[idx]
        abs_path: Path = info["abs_path"]
        with Image.open(abs_path) as img:
            image = img.convert("RGB")
        return image, info


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    items: Sequence[Dict[str, object]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    if not items:
        return np.zeros((0, model.config.projection_dim), dtype=np.float32)

    dataset = GalleryEncodeDataset(items)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(num_workers, 0),
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: batch,
    )

    vectors: List[np.ndarray] = []
    progress = tqdm(total=len(items), desc="Encoding gallery updates", unit="img")
    for batch in loader:
        images = [img for img, _ in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy())
        progress.update(len(batch))
    progress.close()

    return np.vstack(vectors) if vectors else np.zeros((0, model.config.projection_dim), dtype=np.float32)


def encode_text(model, processor, text, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, required=True, help="描述文件 captions.csv/jsonl")
    parser.add_argument("--gallery", type=str, required=True, help="图库路径 (文件夹，包含图像)")
    parser.add_argument("--out", type=str, required=True, help="输出结果文件路径")
    parser.add_argument("--topk", type=int, default=5, help="返回前K个结果")
    parser.add_argument("--gallery_cache", type=str, default="outputs/cache/gallery", help="图库缓存目录")
    parser.add_argument("--rebuild_cache", action="store_true", help="强制重建缓存")
    parser.add_argument("--hash", choices=["mtime", "sha1"], default="mtime", help="缓存校验方式")
    parser.add_argument("--save_faiss", action="store_true", help="保存 FAISS 索引")
    parser.add_argument("--faiss_path", type=str, help="FAISS 索引文件路径")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32", help="缓存向量数据类型")
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 2, help="加载图库的 DataLoader 进程数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量编码大小")
    args = parser.parse_args()

    captions_path = Path(args.captions).expanduser()
    gallery_dir = Path(args.gallery).expanduser()
    out_path = Path(args.out).expanduser()
    cache_dir = Path(args.gallery_cache).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载 CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[retrieve] Using device: {device}")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(
        model_name,
        use_safetensors=True,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    if hasattr(processor, "image_processor"):
        size_info = processor.image_processor.size
    else:
        size_info = getattr(processor, "feature_extractor", None)
        size_info = getattr(size_info, "size", 224)
    if isinstance(size_info, dict):
        image_size = int(size_info.get("shortest_edge") or max(size_info.values()))
    elif isinstance(size_info, (list, tuple)):
        image_size = int(size_info[0])
    else:
        image_size = int(size_info)

    repo_root = Path(__file__).resolve().parent.parent
    repo_version = get_repo_version(repo_root)
    cache = GalleryCache(
        gallery_dir=gallery_dir,
        cache_dir=cache_dir,
        model_id=model_name,
        image_size=image_size,
        normalize="l2",
        dim=int(model.config.projection_dim),
        dtype=args.dtype,
        etag_method=args.hash,
        repo_version=repo_version,
    )

    # 读取描述文件
    records = load_captions(captions_path)
    print(f"[retrieve] 加载 {len(records)} 条查询")

    manifest_entries: List[ManifestEntry] = []
    embeddings: Optional[np.ndarray] = None
    if not args.rebuild_cache:
        _, manifest_entries, embeddings = cache.load_existing()

    file_infos = cache.gather_file_infos()
    if not file_infos:
        raise FileNotFoundError(f"No images found in gallery: {gallery_dir}")

    keep_items, to_encode_infos, dropped_entries = cache.compute_diff(file_infos, manifest_entries)
    kept_count = len(keep_items)
    added_count = len(to_encode_infos)
    dropped_count = len(dropped_entries)
    print(
        f"[cache] Files total={len(file_infos)} kept={kept_count} added={added_count} dropped={dropped_count}"
    )

    start_time = time.time()
    new_embeddings = encode_images(
        model,
        processor,
        to_encode_infos,
        device,
        args.batch_size,
        args.num_workers,
    )
    if added_count:
        elapsed = time.time() - start_time
        print(f"[cache] Encoded {added_count} images in {elapsed:.2f}s")
    else:
        print("[cache] No new or changed images detected")

    final_embeddings, final_entries = cache.merge_embeddings(
        embeddings,
        file_infos,
        keep_items,
        to_encode_infos,
        new_embeddings,
    )
    cache.write_cache(final_embeddings, final_entries)
    print(f"[cache] Cache updated at {cache.cache_dir}")

    gallery_embeddings = final_embeddings
    gallery_entries = final_entries

    faiss_index = None
    faiss_path = Path(args.faiss_path).expanduser() if args.faiss_path else cache.cache_dir / "faiss.index"
    faiss_available = False
    if args.save_faiss or (faiss_path.exists() and added_count == 0 and dropped_count == 0 and not args.rebuild_cache):
        try:
            import faiss  # type: ignore

            faiss_available = True
        except ImportError:
            print("[cache] FAISS 未安装，跳过索引保存")

    if faiss_available:
        import faiss  # type: ignore

        if added_count == 0 and dropped_count == 0 and faiss_path.exists() and not args.rebuild_cache:
            faiss_index = faiss.read_index(str(faiss_path))
            print(f"[cache] Loaded FAISS index from {faiss_path}")
        else:
            faiss_index = faiss.IndexFlatIP(gallery_embeddings.shape[1])
            faiss_index.add(gallery_embeddings.astype(np.float32))
            print(f"[cache] Built FAISS index with {faiss_index.ntotal} vectors")
            if args.save_faiss:
                from src.utils.io_atomic import atomic_write_bytes

                buffer = faiss.serialize_index(faiss_index)
                atomic_write_bytes(faiss_path, bytes(buffer), suffix=".faiss.tmp")
                print(f"[cache] Saved FAISS index to {faiss_path}")

    # 执行检索
    results = {}
    for rec in tqdm(records, desc="Processing queries"):
        q_text = rec["caption"]
        q_vec = encode_text(model, processor, q_text, device)

        if faiss_index is not None:
            import faiss  # type: ignore

            distances, indices = faiss_index.search(q_vec[None, :].astype(np.float32), args.topk)
            topk_list = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(gallery_entries):
                    continue
                topk_list.append((gallery_entries[idx].path, float(score)))
        else:
            sims = np.dot(gallery_embeddings, q_vec)
            top_indices = np.argsort(sims)[::-1][: args.topk]
            topk_list = [(gallery_entries[idx].path, float(sims[idx])) for idx in top_indices]

        results[rec["image_id"]] = topk_list

    # 保存结果
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"[retrieve] 完成检索，共处理 {len(records)} 个查询。")
    print(f"[retrieve] 结果写入: {out_path}")


if __name__ == "__main__":
    main()
