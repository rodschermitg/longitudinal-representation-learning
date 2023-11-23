import json
import os

import h5py
import monai
import numpy as np
import torch


MODALITY_KEYS_1 = ["t1ce_1", "flair_1", "t1_1", "t2_1"]
MODALITY_KEYS_2 = ["t1ce_2", "flair_2", "t1_2", "t2_2"]

train_transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=MODALITY_KEYS_1+MODALITY_KEYS_2,
        image_only=True,
        ensure_channel_first=True
    ),
    monai.transforms.NormalizeIntensityd(
        keys=MODALITY_KEYS_1+MODALITY_KEYS_2,
        channel_wise=True
    ),
    monai.transforms.ConcatItemsd(keys=MODALITY_KEYS_1, name="image_1", dim=0),
    monai.transforms.ConcatItemsd(keys=MODALITY_KEYS_2, name="image_2", dim=0),
])
val_transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=MODALITY_KEYS_1+MODALITY_KEYS_2,
        image_only=True,
        ensure_channel_first=True
    ),
    # add padding to ensure input/output shape consistency when evaluating on
    # full size images during validation
    monai.transforms.SpatialPadd(
        keys=MODALITY_KEYS_1+MODALITY_KEYS_2,
        spatial_size=(240, 240, 160)
    ),
    monai.transforms.NormalizeIntensityd(
        keys=MODALITY_KEYS_1+MODALITY_KEYS_2,
        channel_wise=True
    ),
    monai.transforms.ConcatItemsd(keys=MODALITY_KEYS_1, name="image_1", dim=0),
    monai.transforms.ConcatItemsd(keys=MODALITY_KEYS_2, name="image_2", dim=0),
])

src_dir = os.path.join("data", "processed", "patients", "pretrain")
with open(os.path.join(src_dir, "dataset.json"), "r") as file:
    data = json.load(file)

train_dataset = monai.data.Dataset(
    data=data["train"],
    transform=train_transforms
)
val_dataset = monai.data.Dataset(
    data=data["val"],
    transform=val_transforms
)
train_dataloader = monai.data.DataLoader(train_dataset, batch_size=1)
val_dataloader = monai.data.DataLoader(val_dataset, batch_size=1)

dst_dir = os.path.join("data_h5py", "patients", "pretrain")
with h5py.File(
    os.path.join(dst_dir, "train_placeholder_long.hdf5"), "w"
) as file:
    for i, subject in enumerate(train_dataloader):
        subject = torch.cat((subject["image_1"], subject["image_2"]))
        print(f"subject {i} (train): {subject.shape}")
        grp = file.create_group(f"subj{i}")
        file[f"subj{i}"]["t1"] = subject
        file[f"subj{i}"]["age"] = [  # no age information available
            50 + time_point*5 + np.random.randint(-2, 2)
            for time_point in range(subject.shape[0])
        ]
with h5py.File(
    os.path.join(dst_dir, "val_placeholder_long.hdf5"), "w"
) as file:
    for i, subject in enumerate(val_dataloader, i+1):
        subject = torch.cat((subject["image_1"], subject["image_2"]))
        print(f"subject {i} (val): {subject.shape}")
        grp = file.create_group(f"subj{i}")
        file[f"subj{i}"]["t1"] = subject
        file[f"subj{i}"]["age"] = [  # no age information available
            50 + time_point*5 + np.random.randint(-2, 2)
            for time_point in range(subject.shape[0])
        ]
