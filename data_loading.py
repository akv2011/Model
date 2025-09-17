from typing import Tuple

import torch
from monai.data import CacheDataset, DataLoader, ZipDataset, load_decathlon_datalist
from torch.utils.data.dataloader import default_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, NormalizeIntensityd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, ToTensord
)


def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(N,N,N),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.10, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.10, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.10, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])


def simple_dual_stream_collate(batch):
    ct_samples, mri_samples = [], []
    for ct_dict, mri_dict in batch:
        ct_samples.append(ct_dict)
        mri_samples.append(mri_dict)
    collated_ct = default_collate(ct_samples)
    collated_mri = default_collate(mri_samples)
    return {
        'ct_image': collated_ct.get('image', torch.empty(0)),
        'ct_label': collated_ct.get('label', torch.empty(0)),
        'ct_filename': collated_ct.get('filename', []),
        'mri_image': collated_mri.get('image', torch.empty(0)),
        'mri_label': collated_mri.get('label', torch.empty(0)),
        'mri_filename': collated_mri.get('filename', []),
    }


def create_dataloaders(
    dataset_json: str,
    cache_nums: Tuple[int, int, int, int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    cache_ct_train, cache_mri_train, cache_ct_val, cache_mri_val = cache_nums

    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    full_train = load_decathlon_datalist(dataset_json, True, 'training')
    full_val = load_decathlon_datalist(dataset_json, True, 'validation')

    ct_train = [item for item in full_train if item.get('modality', '').upper() == 'CT']
    mri_train = [item for item in full_train if item.get('modality', '').upper() == 'MRI']
    ct_val = [item for item in full_val if item.get('modality', '').upper() == 'CT']
    mri_val = [item for item in full_val if item.get('modality', '').upper() == 'MRI']

    min_train_len = min(len(ct_train), len(mri_train))

    ct_train_ds = CacheDataset(
        data=ct_train[:min_train_len],
        transform=train_transforms,
        cache_num=min(cache_ct_train, min_train_len),
        cache_rate=1.0,
        num_workers=0,
        progress=True,
    )
    mri_train_ds = CacheDataset(
        data=mri_train[:min_train_len],
        transform=train_transforms,
        cache_num=min(cache_mri_train, min_train_len),
        cache_rate=1.0,
        num_workers=0,
        progress=True,
    )

    val_ct_ds = CacheDataset(
        data=ct_val,
        transform=val_transforms,
        cache_num=min(cache_ct_val, len(ct_val)),
        cache_rate=1.0,
        num_workers=0,
        progress=True,
    )
    val_mri_ds = CacheDataset(
        data=mri_val,
        transform=val_transforms,
        cache_num=min(cache_mri_val, len(mri_val)),
        cache_rate=1.0,
        num_workers=0,
        progress=True,
    )

    train_loader = DataLoader(
        ZipDataset([ct_train_ds, mri_train_ds]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=simple_dual_stream_collate,
    )

    val_loader_ct = DataLoader(
        val_ct_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader_mri = DataLoader(
        val_mri_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader_ct, val_loader_mri
