import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from datasets.data_reader import VideoReader
from datasets.data_reader import TifReader
from datasets.data_reader import CSVReader


def collect_valid_folders(root_dir, window_length, require_gt=False):
    valid_folders = []
    num = 0
    root_dir = Path(root_dir)
    for folder in root_dir.iterdir():
        if not folder.is_dir():
            continue

        num += 1
        avi = folder / "video.avi"
        mp4 = folder / "video.mp4"
        tifs = list(folder.glob("*.tif"))
        gt = folder / "gt.csv"

        if require_gt and not gt.exists():
            print(f"[Warning] Skipped {folder.name}: Missing gt.csv")
            continue

        if avi.exists() or mp4.exists():
            import cv2
            cap = cv2.VideoCapture(str(avi if avi.exists() else mp4))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        elif len(tifs) > 0:
            frame_count = len(tifs)
        else:
            print(f"[Warning] Skipped {folder.name}: No valid video or tif frames found.")
            continue

        if frame_count < window_length:
            print(f"[Warning] Skipped {folder.name}: Frame count {frame_count} < {window_length}")
            continue

        valid_folders.append(folder)

    print(f"[Info] Total {len(valid_folders)} valid folders collected from {num} folders.")
    return valid_folders


class VideoSequenceDataset(Dataset):
    def __init__(self, root_dir, window_length, split, transform):
        self.root_dir = Path(root_dir)
        self.window_length = window_length
        self.transform = transform
        self.split = split
        self.require_gt = (split == "train" or split == "val")

        self.data_folders = collect_valid_folders(root_dir, window_length, self.require_gt)
        self.samples = []
        self.build_data()
        print(f"[Info] Total {len(self.samples)} samples collected from {len(self.data_folders)} folders.")

    def _load_all_frames(self, folder):
        avi = folder / "video.avi"
        mp4 = folder / "video.mp4"
        if any(folder.glob("*.tif")):
            reader = TifReader(folder)
        else:
            reader = VideoReader(avi if avi.exists() else mp4)
        return reader.read_frames()

    def gt_frame_to_tensor(self, frame):
        ids = torch.from_numpy(frame["particle_id"].to_numpy()).long()
        boxes = torch.from_numpy(frame[["cx", "cy", "w", "h"]].to_numpy()).float()
        logits = torch.ones_like(boxes[:, 0])
        return {"id": ids, "logits": logits, "boxes": boxes}

    def match_gt_frames(self, prev, curr):
        prev_ids = prev["particle_id"].to_numpy()
        curr_ids = curr["particle_id"].to_numpy()
        Np, Nc = len(prev_ids), len(curr_ids)

        prev_id2idx = {pid: i for i, pid in enumerate(prev_ids)}
        curr_id2idx = {pid: i for i, pid in enumerate(curr_ids)}

        common_ids = np.intersect1d(prev_ids, curr_ids)

        row_target = torch.full((Np,), Nc, dtype=torch.long)
        col_target = torch.full((Nc,), Np, dtype=torch.long)

        for pid in common_ids:
            i = prev_id2idx[pid]
            j = curr_id2idx[pid]
            row_target[i] = j
            col_target[j] = i

        return row_target, col_target

    def build_data(self):
        for folder in self.data_folders:
            frames = self._load_all_frames(folder)
            gt_path = folder / "gt.csv"
            gt_data = CSVReader(gt_path).read() if gt_path.exists() else None
            gt_frame = gt_data.groupby("frame") if gt_path.exists() else None

            for i in range(len(frames) - self.window_length + 1):
                clip = frames[i: i+self.window_length]  # [T, H, W, C]
                clip = np.stack(clip, axis=0)  # (T, H, W, C)
                clip = torch.from_numpy(clip).permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)
                if self.transform:
                    clip = self.transform(clip)
                sample = {"x": clip}
                
                if gt_data is not None:
                    t = i + self.window_length - 1
                    prev = gt_frame.get_group(t - 1)
                    curr = gt_frame.get_group(t)
                    sample["prev_info"] = self.gt_frame_to_tensor(prev)
                    sample["curr_info"] = self.gt_frame_to_tensor(curr)
                    sample["row_target"], sample["col_target"] = self.match_gt_frames(prev, curr)

                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_padding(batch):
    Np_list = [item["row_target"].size(0) for item in batch]
    Nc_list = [item["col_target"].size(0) for item in batch]
    Np_max, Nc_max = max(Np_list), max(Nc_list)
    B = len(batch)

    def pad_dict(info_list, Nmax):
        out = {}
        keys = info_list[0].keys()
        for k in keys:
            tensors = [v[k] for v in info_list]
            padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
            if padded.size(1) < Nmax:
                pad_shape = list(padded.shape)
                pad_shape[1] = Nmax - padded.size(1)
                padded = torch.cat([padded, torch.zeros(*pad_shape, device=padded.device)], dim=1)
            out[k] = padded
        return out

    prev_info = pad_dict([item["prev_info"] for item in batch], Np_max)
    curr_info = pad_dict([item["curr_info"] for item in batch], Nc_max)

    row_target = pad_sequence(
        [item["row_target"] for item in batch], batch_first=True, padding_value=-1
    )
    col_target = pad_sequence(
        [item["col_target"] for item in batch], batch_first=True, padding_value=-1
    )

    row_mask = torch.arange(Np_max).expand(B, Np_max) < torch.tensor(Np_list).unsqueeze(1)
    col_mask = torch.arange(Nc_max).expand(B, Nc_max) < torch.tensor(Nc_list).unsqueeze(1)

    batch_out = {
        "x": torch.stack([item['x'] for item in batch]),
        "prev_info": prev_info,
        "curr_info": curr_info,
        "row_target": row_target,
        "col_target": col_target,
        "row_mask": row_mask,
        "col_mask": col_mask,
    }
    return batch_out

def collate_fn_target_padding(batch):
    Np_list = [item["row_target"].size(0) for item in batch]
    Nc_list = [item["col_target"].size(0) for item in batch]
    Np_max, Nc_max = max(Np_list), max(Nc_list)
    B = len(batch)

    row_target = pad_sequence(
        [item["row_target"] for item in batch], batch_first=True, padding_value=-1
    )
    col_target = pad_sequence(
        [item["col_target"] for item in batch], batch_first=True, padding_value=-1
    )

    row_mask = torch.arange(Np_max).expand(B, Np_max) < torch.tensor(Np_list).unsqueeze(1)
    col_mask = torch.arange(Nc_max).expand(B, Nc_max) < torch.tensor(Nc_list).unsqueeze(1)

    batch_out = {
        'x': torch.stack([item['x'] for item in batch]),
        'prev_info': [item['prev_info'] for item in batch],
        'curr_info': [item['curr_info'] for item in batch],
        'row_target': row_target,
        'col_target': col_target,
        "row_mask": row_mask,
        "col_mask": col_mask,
    }
    return batch_out


class VideoSequenceModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        window_length: int = 5,
        root_dir: str = "./data",
        num_workers: int = 0,
        transform=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_length = window_length
        self.transform = transform

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.predict_ds = None

        self.train_dir = Path(root_dir) / "train"
        self.val_dir = Path(root_dir) / "val"
        self.test_dir = Path(root_dir) / "test"
        self.predict_dir = Path(root_dir) / "predict"

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = VideoSequenceDataset(
                self.train_dir,
                self.window_length,
                split="train",
                transform=self.transform,
            )
            self.val_ds = VideoSequenceDataset(
                self.val_dir,
                self.window_length,
                split="val",
                transform=self.transform,
            )
        if stage == "test":
            self.test_ds = VideoSequenceDataset(
                self.test_dir,
                self.window_length,
                split="test",
                transform=None,
            )
        if stage == "predict":
            self.predict_ds = VideoSequenceDataset(
                self.predict_dir,
                self.window_length,
                split="predict",
                transform=None,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_target_padding,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        if self.val_dir is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn_target_padding,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.test_dir is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        if self.predict_dir is None:
            return None
        return DataLoader(
            self.predict_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
