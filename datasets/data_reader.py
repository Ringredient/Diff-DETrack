import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread


class VideoReader:
    """For .avi or .mp4 videos"""
    def __init__(self, video_path):
        self.video_path = str(video_path)

    def read_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32))
        cap.release()
        return frames


class TifReader:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

    def read_frames(self):
        tif_files = sorted(self.folder_path.glob("*.tif"))
        frames = [imread(f).astype(np.float32) for f in tif_files]
        frames = [f[..., None].repeat(3, axis=-1) if f.ndim == 2 else f for f in frames]
        return frames


class CSVReader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def read(self):
        if not self.csv_path.exists():
            return None
        return pd.read_csv(self.csv_path)
