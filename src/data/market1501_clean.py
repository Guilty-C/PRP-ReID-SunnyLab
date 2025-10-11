"""Dataset and sampling utilities for cleaned Market-1501 data."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_market1501_transforms(height: int = 256, width: int = 128, is_train: bool = True) -> T.Compose:
    if is_train:
        transform_list = [
            T.Resize((height, width)),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.5, value="random"),
        ]
    else:
        transform_list = [
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return T.Compose(transform_list)


class Market1501CleanDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        manifest_csv: str | Path,
        ids: Sequence[int],
        transform: Optional[T.Compose] = None,
        is_train: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_csv = Path(manifest_csv).expanduser().resolve()
        self.ids = set(int(i) for i in ids)
        self.is_train = is_train
        self.transform = transform or build_market1501_transforms(is_train=is_train)

        self.records: List[dict] = []
        self._labels: List[int] = []

        with self.manifest_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = int(row["pid"])
                if self.ids and pid not in self.ids:
                    continue
                cam = int(row["cam"])
                seq = int(row["seq"])
                frame = int(row["frame"])
                idx = int(row["idx"])
                rel_path = Path(row["path"])
                img_path = rel_path if rel_path.is_absolute() else self.root / rel_path
                record = {
                    "path": img_path,
                    "pid": pid,
                    "cam": cam,
                    "seq": seq,
                    "frame": frame,
                    "idx": idx,
                }
                self.records.append(record)
                self._labels.append(pid)

        if not self.records:
            raise ValueError(
                "No records found for the provided IDs."
                f" Manifest: {self.manifest_csv} | IDs: {sorted(self.ids) if self.ids else 'ALL'}"
            )

    @property
    def labels(self) -> List[int]:
        return self._labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path: Path = record["path"]
        with Image.open(path) as img:
            img = img.convert("RGB")
            image = self.transform(img) if self.transform else img
        pid = int(record["pid"])
        cam = int(record["cam"])
        return image, pid, cam, str(path)


class PKSampler(Sampler[List[int]]):
    def __init__(self, labels: Sequence[int], P: int = 16, K: int = 4, seed: int = 42) -> None:
        if P <= 0 or K <= 0:
            raise ValueError("P and K must be positive integers")
        self.labels = list(labels)
        self.P = P
        self.K = K
        self.seed = seed
        self.epoch = 0

        self.label_to_indices: dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[int(label)].append(idx)
        if len(self.label_to_indices) < self.P:
            raise ValueError("Not enough unique identities to form a PK batch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        label_to_pool = {label: indices[:] for label, indices in self.label_to_indices.items()}
        for indices in label_to_pool.values():
            rng.shuffle(indices)

        # prepare label order with repetitions according to dataset size
        label_queue: List[int] = []
        for label, indices in label_to_pool.items():
            repeat = max(1, (len(indices) + self.K - 1) // self.K)
            label_queue.extend([label] * repeat)
        rng.shuffle(label_queue)

        batches: List[List[int]] = []
        for i in range(0, len(label_queue), self.P):
            batch_labels = label_queue[i : i + self.P]
            if len(batch_labels) < self.P:
                break
            batch: List[int] = []
            for label in batch_labels:
                pool = label_to_pool[label]
                if len(pool) < self.K:
                    replenished = self.label_to_indices[label][:]
                    rng.shuffle(replenished)
                    pool.extend(replenished)
                selected = pool[: self.K]
                del pool[: self.K]
                batch.extend(selected)
            if len(batch) == self.P * self.K:
                batches.append(batch)

        self._length = len(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if hasattr(self, "_length"):
            return self._length
        total = sum(len(indices) for indices in self.label_to_indices.values())
        return max(total // (self.P * self.K), 1)
