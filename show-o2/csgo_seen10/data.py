"""Manifest-driven Seen-10 dataset access.

Only the benchmark report, manifest, declared split files, calibration, and paths
derived from their mappings are used. In inference mode target image files are
never opened or returned by the dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)

SPLIT_FILES = {
    "train": "train.json",
    "validation": "validation.json",
    "discrete_test": "discrete_test.json",
    "continuous": "continuous_clips.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _safe_under(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Benchmark path escapes DATA_ROOT: {relative!r}") from error
    return candidate


def _load_rgb_tensor(path: Path, resolution: int) -> torch.Tensor:
    """Match Show-o2's resize-short-side, center-crop, RGB and [-1, 1] transform."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    if min(width, height) <= 0:
        raise ValueError(f"Invalid image size at {path}: {image.size}")
    if width <= height:
        resized = (resolution, int(resolution * height / width))
    else:
        resized = (int(resolution * width / height), resolution)
    image = image.resize(resized, resample=Image.Resampling.BICUBIC)
    left = max((image.width - resolution) // 2, 0)
    top = max((image.height - resolution) // 2, 0)
    image = image.crop((left, top, left + resolution, top + resolution))
    array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return tensor


class CSGOSeen10Dataset(Dataset):
    """Read one official Seen-10 split without scanning the image tree."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        include_target: bool,
        resolution: int = 448,
        limit: Optional[int] = None,
        limit_per_map: Optional[int] = None,
    ) -> None:
        if split not in SPLIT_FILES:
            raise ValueError(f"Unknown Seen-10 split {split!r}; expected one of {tuple(SPLIT_FILES)}")
        if include_target and split not in {"train", "validation"}:
            raise ValueError("Targets are only available for train/validation datasets")

        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.include_target = include_target
        self.resolution = int(resolution)
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")

        report_path = self.data_root / "minimal_dataset_report.json"
        manifest_path = self.data_root / "benchmark_manifest.json"
        calibration_path = self.data_root / "calibration" / "z_calibration.json"
        self.report = _read_json(report_path)
        self.manifest = _read_json(manifest_path)
        self.calibration = _read_json(calibration_path)

        if self.report.get("status") != "verified":
            raise ValueError(f"minimal_dataset_report.json status is not verified: {self.report.get('status')!r}")
        if self.report.get("images", {}).get("status") != "verified":
            raise ValueError("minimal_dataset_report.json does not verify the images bundle")
        if self.report.get("radars", {}).get("status") != "verified":
            raise ValueError("minimal_dataset_report.json does not verify the radars bundle")
        report_manifest = self.report.get("manifest", {})
        if report_manifest.get("sha256") and sha256_file(manifest_path) != report_manifest["sha256"]:
            raise ValueError("benchmark_manifest.json does not match minimal_dataset_report.json")
        report_manifest_path = report_manifest.get("path")
        if report_manifest_path and Path(report_manifest_path).name != manifest_path.name:
            raise ValueError(f"Unexpected manifest mapping in report: {report_manifest_path}")
        if self.manifest.get("benchmark_id") != "csgo_benchmark_v2":
            raise ValueError("DATA_ROOT is not a csgo_benchmark_v2 bundle")
        seen_maps = tuple(self.manifest.get("protocol", {}).get("seen_maps", ()))
        if seen_maps != MAPS:
            raise ValueError(f"Seen-10 map order mismatch: expected {MAPS}, got {seen_maps}")
        if self.calibration.get("benchmark_id") != "csgo_benchmark_v2":
            raise ValueError("z_calibration.json is not for csgo_benchmark_v2")
        calibration_sha256 = self.manifest.get("calibration", {}).get("sha256")
        if calibration_sha256 and sha256_file(calibration_path) != calibration_sha256:
            raise ValueError("z_calibration.json does not match benchmark_manifest.json")
        selected_images = self.manifest.get("selected_images", {})
        selected_checksum_path = _safe_under(
            self.data_root, str(selected_images.get("file", "selected_images.sha256"))
        )
        if not selected_checksum_path.is_file():
            raise FileNotFoundError(f"Selected-image checksum file not found: {selected_checksum_path}")
        if selected_images.get("sha256") and sha256_file(selected_checksum_path) != selected_images["sha256"]:
            raise ValueError("selected_images.sha256 does not match benchmark_manifest.json")

        self.image_template = self.report["images"]["target_template"]
        self.radar_root = self.report["radars"]["root"]
        self.radar_files = self.manifest["source"]["radar_files"]
        report_maps = tuple(self.report.get("maps", ()))
        if report_maps[: len(MAPS)] != MAPS:
            raise ValueError(f"minimal_dataset_report.json map order mismatch: {report_maps[:len(MAPS)]}")
        radar_entries = {
            entry["map"]: entry for entry in self.report["radars"].get("entries", [])
        }
        for map_name in MAPS:
            entry = radar_entries.get(map_name)
            if entry is None or entry.get("source") != self.radar_files.get(map_name):
                raise ValueError(f"Missing or inconsistent report/manifest radar mapping for {map_name}")
        self.z_ranges = self.calibration["z_ranges"]
        self.map_to_id = {map_name: index for index, map_name in enumerate(MAPS)}
        self.rows: List[Dict[str, Any]] = []
        declared_counts = self.manifest.get("counts", {}).get("seen", {})

        for map_name in MAPS:
            split_path = self.data_root / "splits" / "seen" / map_name / SPLIT_FILES[split]
            payload = _read_json(split_path)
            map_rows = self._rows_for_map(payload, map_name)
            if split == "continuous":
                clips = payload.get("clips", [])
                if len(clips) != 20 or any(len(clip.get("frames", [])) != 64 for clip in clips):
                    raise ValueError(f"{map_name}/continuous must contain 20 clips of exactly 64 frames")
            expected_count_key = "continuous_frames" if split == "continuous" else split
            expected_count = declared_counts.get(map_name, {}).get(expected_count_key)
            if expected_count is None or len(map_rows) != int(expected_count):
                raise ValueError(
                    f"{map_name}/{split} has {len(map_rows)} rows; manifest declares {expected_count}"
                )
            if limit_per_map is not None:
                map_rows = map_rows[: max(0, int(limit_per_map))]
            self.rows.extend(map_rows)

        if limit is not None:
            self.rows = self.rows[: max(0, int(limit))]
        if not self.rows:
            raise ValueError(f"The requested split {split!r} is empty")
        self.sample_count = len(self.rows)
        self.provenance = {
            "benchmark_manifest_sha256": sha256_file(manifest_path),
            "asset_report_sha256": sha256_file(report_path),
            "selected_images_sha256": self.manifest.get("selected_images", {}).get("sha256"),
            "selected_images_checksum_file_sha256": sha256_file(
                self.data_root / self.manifest.get("selected_images", {}).get("file", "selected_images.sha256")
            ),
        }

    def _rows_for_map(self, payload: Any, map_name: str) -> List[Dict[str, Any]]:
        if self.split == "continuous":
            if payload.get("map") != map_name or payload.get("split") != "continuous":
                raise ValueError(f"Continuous clip file metadata mismatch for {map_name}")
            rows: List[Dict[str, Any]] = []
            for clip in payload.get("clips", []):
                clip_id = str(clip["clip_id"])
                frames = clip.get("frames", [])
                for frame_index, frame in enumerate(frames):
                    rows.append(self._make_row(frame, map_name, clip_id=clip_id, frame_index=frame_index))
            return rows

        if not isinstance(payload, list):
            raise ValueError(f"Split file for {map_name}/{self.split} must contain a JSON list")
        return [self._make_row(row, map_name) for row in payload]

    def _make_row(
        self, row: Dict[str, Any], expected_map: str, *, clip_id: Optional[str] = None, frame_index: Optional[int] = None
    ) -> Dict[str, Any]:
        map_name = str(row.get("map", expected_map))
        if map_name != expected_map:
            raise ValueError(f"Split row map mismatch: expected {expected_map}, got {map_name}")
        file_frame = str(row["file_frame"])
        if not file_frame or Path(file_frame).name != file_frame:
            raise ValueError(f"Invalid file_frame identity: {file_frame!r}")
        z_range = self.z_ranges[map_name]
        z_min, z_max = float(z_range["z_min"]), float(z_range["z_max"])
        if z_max <= z_min:
            raise ValueError(f"Invalid calibration range for {map_name}: {z_min}..{z_max}")
        pose = torch.tensor(
            [
                float(row["x"]) / 1024.0,
                float(row["y"]) / 1024.0,
                (float(row["z"]) - z_min) / (z_max - z_min),
                float(row["angle_v"]) / (2.0 * math.pi),
                float(row["angle_h"]) / (2.0 * math.pi),
            ],
            dtype=torch.float32,
        )
        if not torch.isfinite(pose).all():
            raise ValueError(f"Non-finite pose for {map_name}/{file_frame}")

        radar_source = self.radar_files.get(map_name)
        if not radar_source:
            raise ValueError(f"Manifest has no radar mapping for {map_name}")
        radar_entry = next(
            (entry for entry in self.report["radars"].get("entries", []) if entry.get("map") == map_name),
            None,
        )
        if radar_entry is None:
            raise ValueError(f"minimal_dataset_report.json has no radar target for {map_name}")
        radar_relative = str(radar_entry["target"])
        radar_root = str(self.radar_root).rstrip("/")
        radar_path = _safe_under(self.data_root, f"{radar_root}/{radar_relative}")
        if self.include_target:
            image_relative = self.image_template.format(map=map_name, file_frame=file_frame)
            target_path = _safe_under(self.data_root, image_relative)
        else:
            target_path = None
        return {
            "sample_id": f"{map_name}/{file_frame}",
            "map_name": map_name,
            "map_id": self.map_to_id[map_name],
            "file_frame": file_frame,
            "clip_id": clip_id,
            "frame_index": frame_index,
            "pose": pose,
            "radar_path": radar_path,
            "target_path": target_path,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def get_condition_only(self, index: int) -> Dict[str, Any]:
        """Load only radar and metadata, even when this is a train/val dataset."""
        row = self.rows[index]
        return {
            "radar": _load_rgb_tensor(row["radar_path"], self.resolution),
            "pose": row["pose"],
            "map_id": torch.tensor(row["map_id"], dtype=torch.long),
            "sample_id": row["sample_id"],
            "map_name": row["map_name"],
            "file_frame": row["file_frame"],
            "clip_id": row["clip_id"] or "",
            "frame_index": -1 if row["frame_index"] is None else row["frame_index"],
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        sample = self.get_condition_only(index)
        if self.include_target:
            target_path = row["target_path"]
            if target_path is None:
                raise RuntimeError("Target path missing in target-enabled dataset")
            sample["target"] = _load_rgb_tensor(target_path, self.resolution)
        return sample


def declared_sample_counts(data_root: str | Path) -> Dict[str, Dict[str, int]]:
    """Return manifest counts in fixed map order for provenance output."""
    root = Path(data_root).expanduser().resolve()
    manifest = _read_json(root / "benchmark_manifest.json")
    counts = manifest["counts"]["seen"]
    return {map_name: {key: int(value) for key, value in counts[map_name].items()} for map_name in MAPS}
