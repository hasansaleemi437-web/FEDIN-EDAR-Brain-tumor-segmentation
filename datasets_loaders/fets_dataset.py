"""FeTS 2022 3D MRI Brain Tumor Dataset Loader with MONAI preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    Spacingd,
)

from .base import BaseDatasetLoader


PATCH_SIZE = (96, 96, 96)


class FeTSDatasetLoader(BaseDatasetLoader):
    """Loads 3D multi-parametric MRI (T1, T1ce, T2, FLAIR) for FeTS 2022."""

    def __init__(
        self,
        root_dir: str | Path,
        partition_csv: str | Path | None = None,
        global_test_fraction: float = 0.15,
        seed: int = 42,
    ):
        super().__init__(root_dir, partition_csv)
        self.global_test_fraction = global_test_fraction
        self.seed = seed
        self._groups = None

    # --- Built-in MONAI 3D Preprocessing Transforms ---

    @staticmethod
    def get_train_transforms():
        """Return MONAI preprocessing and augmentation transforms for training."""
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                Orientationd(
                    keys=("image", "label"),
                    axcodes="RAS",
                    labels=None,
                ),
                Spacingd(
                    keys=("image", "label"),
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(
                    keys="image",
                    nonzero=True,
                    channel_wise=True,
                ),
                MapLabelValued(
                    keys="label",
                    orig_labels=[4],
                    target_labels=[3],
                ),
                CropForegroundd(
                    keys=("image", "label"),
                    source_key="image",
                ),
                RandCropByPosNegLabeld(
                    keys=("image", "label"),
                    label_key="label",
                    spatial_size=PATCH_SIZE,
                    pos=1,
                    neg=1,
                    num_samples=2,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=("image", "label"),
                    prob=0.5,
                    spatial_axis=0,
                ),
                RandFlipd(
                    keys=("image", "label"),
                    prob=0.5,
                    spatial_axis=1,
                ),
                RandRotate90d(
                    keys=("image", "label"),
                    prob=0.5,
                    max_k=3,
                ),
                EnsureTyped(keys=("image", "label")),
            ]
        )

    @staticmethod
    def get_val_transforms():
        """Return MONAI preprocessing transforms for validation and testing."""
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                Orientationd(
                    keys=("image", "label"),
                    axcodes="RAS",
                    labels=None,
                ),
                Spacingd(
                    keys=("image", "label"),
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(
                    keys="image",
                    nonzero=True,
                    channel_wise=True,
                ),
                MapLabelValued(
                    keys="label",
                    orig_labels=[4],
                    target_labels=[3],
                ),
                CropForegroundd(
                    keys=("image", "label"),
                    source_key="image",
                ),
                EnsureTyped(keys=("image", "label")),
            ]
        )

    # --- Internal File Matching ---

    def _case_files(self, subject_dir: Path) -> dict:
        """
        Collect all required files for one FeTS subject.

        Each subject contains four MRI modalities:
        T1ce, T1, T2, and FLAIR, plus one segmentation mask.

        Returns:
            Dictionary containing image paths, label path, and subject ID.
        """
        files = list(subject_dir.rglob("*.nii")) + list(
            subject_dir.rglob("*.nii.gz")
        )

        if not files:
            raise FileNotFoundError(
                f"No NIfTI files found in {subject_dir}"
            )

        def find(modality: str) -> Path:
            """Find the NIfTI file corresponding to one MRI modality."""
            matches = [
                path
                for path in files
                if modality in path.name.lower()
                and "seg" not in path.name.lower()
            ]

            if not matches:
                raise FileNotFoundError(
                    f"Missing {modality} in {subject_dir}"
                )

            return sorted(
                matches,
                key=lambda path: len(path.name),
            )[0]

        images = [
            str(find(modality))
            for modality in ("t1ce", "t1", "t2", "flair")
        ]

        labels = [
            path
            for path in files
            if "seg" in path.name.lower()
        ]

        if len(labels) != 1:
            raise FileNotFoundError(
                f"Expected 1 segmentation in {subject_dir}, "
                f"found {len(labels)}"
            )

        return {
            "image": images,
            "label": str(labels[0]),
            "subject_id": subject_dir.name,
        }

    def _get_partitioned_groups(self):
        """
        Group subjects according to the FeTS partition CSV.

        Each Partition_ID represents one federated client/institution.
        Subjects with the same Partition_ID remain together as one
        client partition.
        """
        if self._groups is not None:
            return self._groups

        table = pd.read_csv(self.partition_csv)

        groups = []

        for partition_id, frame in table.groupby(
            "Partition_ID",
            sort=True,
        ):
            records = [
                self._case_files(
                    self.root_dir / str(subject_id)
                )
                for subject_id in frame["Subject_ID"]
            ]

            groups.append(
                (str(partition_id), records)
            )

        self._groups = groups
        return self._groups

    def _split_global_test(
        self,
        records: list[dict],
        partition_index: int,
    ):
        """
        Reserve unseen global-test cases from one client partition.

        The global_test_fraction is applied independently to each
        client partition. The selected cases are excluded from that
        client's local training and validation data.

        The reserved test cases from all client partitions are later
        combined to form the centralized global test set.
        """
        if (
            not 0.0 < self.global_test_fraction < 1.0
            or len(records) < 2
        ):
            return records, []

        rng = np.random.default_rng(
            self.seed + partition_index
        )

        n_test = max(
            1,
            int(round(len(records) * self.global_test_fraction)),
        )

        test_indices = set(
            rng.permutation(len(records))[:n_test].tolist()
        )

        trainval = [
            record
            for index, record in enumerate(records)
            if index not in test_indices
        ]

        test = [
            record
            for index, record in enumerate(records)
            if index in test_indices
        ]

        return trainval, test

    # --- Public Loader APIs ---

    def load_partition(
        self,
        partition_id: int,
        batch_size: int = 1,
    ):
        """
        Return train and validation loaders for one federated client.

        Global-test cases are reserved first. The remaining cases are
        split into local training and validation sets.
        """
        groups = self._get_partitioned_groups()

        if not 0 <= partition_id < len(groups):
            raise IndexError(
                f"Partition {partition_id} invalid "
                f"(found {len(groups)} partitions in {self.partition_csv.name})"
            )

        # Reserve global-test cases first.
        trainval_records, _global_test_records = (
            self._split_global_test(
                groups[partition_id][1],
                partition_id,
            )
        )

        # Split the remaining cases into local training and validation.
        rng = np.random.default_rng(
            self.seed + partition_id
        )

        n_val = max(
            1,
            int(round(len(trainval_records) * 0.15)),
        )

        val_indices = set(
            rng.permutation(len(trainval_records))[:n_val].tolist()
        )

        train_records = [
            record
            for index, record in enumerate(trainval_records)
            if index not in val_indices
        ]

        val_records = [
            record
            for index, record in enumerate(trainval_records)
            if index in val_indices
        ]

        train_ds = Dataset(
            train_records,
            transform=self.get_train_transforms(),
        )

        val_ds = Dataset(
            val_records,
            transform=self.get_val_transforms(),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader

    def load_partition_test(
        self,
        partition_id: int,
        batch_size: int = 1,
    ):
        """
        Return DataLoader for the unseen 15% test set of one federated client.
        """
        groups = self._get_partitioned_groups()

        if not 0 <= partition_id < len(groups):
            raise IndexError(
                f"Partition {partition_id} invalid "
                f"(found {len(groups)} partitions in {self.partition_csv.name})"
            )

        _, test_records = self._split_global_test(
            groups[partition_id][1],
            partition_id,
        )

        test_ds = Dataset(
            test_records,
            transform=self.get_val_transforms(),
        )

        return DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

    def load_global_test(self, batch_size: int = 1):
        """
        Build the centralized global test set.

        The reserved test cases from every federated client partition
        are combined into one unseen test dataset for server-side
        global evaluation.
        """
        groups = self._get_partitioned_groups()

        all_test_records = []

        for partition_index, (_, records) in enumerate(groups):
            _, test_records = self._split_global_test(
                records,
                partition_index,
            )

            all_test_records.extend(test_records)

        test_ds = Dataset(
            all_test_records,
            transform=self.get_val_transforms(),
        )

        return DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )