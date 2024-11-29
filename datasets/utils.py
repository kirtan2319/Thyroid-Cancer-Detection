import numpy as np
import torch
import torchvision.transforms as transforms
import datasets
import pandas as pd
import random
import torchio as tio
from utils.spatial_transforms import ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo
from torch.utils.data import WeightedRandomSampler

# Global worker_init_fn function to ensure pickleability
def worker_init_fn(worker_id, opt):
    np.random.seed(opt['random_seed'])
    random.seed(opt['random_seed'])

def get_dataset(opt):
    data_setting = opt['data_setting']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    print("into dataset")
    if opt['is_3d']:
        # Define 3D normalization and transform settings
        mean_3d = [0.45, 0.45, 0.45]
        std_3d = [0.225, 0.225, 0.225]
        sizes = {'ADNI': (192, 192, 128), 'ADNI3T': (192, 192, 128), 'OCT': (192, 192, 96), 'COVID_CT_MD': (224, 224, 80)}
        
        if data_setting['augment']:
            transform_train = transforms.Compose([
                tio.transforms.RandomFlip(),
                tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=15),
                tio.transforms.CropOrPad(sizes[opt['dataset_name']]),
                ToTensor(),
                NormalizeVideo(mean_3d, std_3d),
            ])
        else:
            transform_train = transforms.Compose([
                tio.transforms.CropOrPad(sizes[opt['dataset_name']]),
                ToTensor(),
                NormalizeVideo(mean_3d, std_3d),
            ])
        
        transform_test = transforms.Compose([
            tio.transforms.CropOrPad(sizes[opt['dataset_name']]),
            ToTensor(),
            NormalizeVideo(mean_3d, std_3d),
        ])
    
    elif opt['is_tabular']:
        # No transformations for tabular data
        transform_train = transform_test = None
    
    else:
        # 2D image transforms
        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-15, 15)),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Set the random seed for data loaders
    g = torch.Generator()
    g.manual_seed(opt['random_seed'])

    image_path = data_setting['image_feature_path']
    train_meta = pd.read_csv(data_setting['train_meta_path'])
    val_meta = pd.read_csv(data_setting['val_meta_path'])
    test_meta = pd.read_csv(data_setting['test_meta_path'])

    print("train df size : " ,len(train_meta))
    print("val df size : " ,len(val_meta))
    print("test df size : " ,len(test_meta))

    if opt['is_3d']:
        dataset_name = getattr(datasets, opt['dataset_name'])
        train_data = dataset_name(train_meta, image_path, opt['sensitive_name'], opt['train_sens_classes'], transform_train)
        val_data = dataset_name(val_meta, image_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
        test_data = dataset_name(test_meta, image_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
    
    elif opt['is_tabular']:
        # Handling tabular data
        print("is tabular")
        dataset_name = getattr(datasets, opt['dataset_name'])
        data_train_df = pd.read_csv(data_setting['data_train_path'])
        data_val_df = pd.read_csv(data_setting['data_val_path'])
        data_test_df = pd.read_csv(data_setting['data_test_path'])
        
        train_data = dataset_name(train_meta, data_train_df, opt['sensitive_name'], opt['train_sens_classes'], None)
        val_data = dataset_name(val_meta, data_val_df, opt['sensitive_name'], opt['sens_classes'], None)
        test_data = dataset_name(test_meta, data_test_df, opt['sensitive_name'], opt['sens_classes'], None)
    
    else:
        # Default data format for pickle-based data
        dataset_name = getattr(datasets, opt['dataset_name'])
        print("is else")
        pickle_train_path = data_setting['pickle_train_path']
        pickle_val_path = data_setting['pickle_val_path']
        pickle_test_path = data_setting['pickle_test_path']
        train_data = dataset_name(train_meta, pickle_train_path, opt['sensitive_name'], opt['train_sens_classes'], transform_train)
        val_data = dataset_name(val_meta, pickle_val_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
        test_data = dataset_name(test_meta, pickle_test_path, opt['sensitive_name'], opt['sens_classes'], transform_test)
        print("train data, lest see: ", len(train_data)," , ", train_data[0])

    print(f'Loaded dataset: {opt["dataset_name"]}')
    
    if opt['experiment'] in ['resampling', 'GroupDRO', 'resamplingSWAD']:
        # If resampling strategy is used, apply WeightedRandomSampler
        weights = train_data.get_weights(resample_which=opt['resample_which'])
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
    else:
        sampler = None

    # Initialize DataLoader with worker_init_fn for reproducibility
    print("initi.. data loader")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt['batch_size'], 
        sampler=sampler,
        shuffle=(opt['experiment'] not in ['resampling', 'GroupDRO', 'resamplingSWAD']),
        num_workers=0, 
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, opt),  # Pass opt to worker_init_fn using lambda
        generator=g,
        pin_memory=True
    )
    print("train loader done")
    print(f"Total number of batches: {len(train_loader)}")
    # for batch_idx, batch in enumerate(train_loader):
    #     print(f"Batch {batch_idx}, Output: {len(batch)}")
    #     if batch_idx == 0:  # Debugging, stop after 10 batches
    #         break


    

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=opt['batch_size'],
        shuffle=True, num_workers=0, 
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, opt),
        generator=g,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=opt['batch_size'],
        shuffle=True, num_workers=0, 
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, opt),
        generator=g,
        pin_memory=True
    )

    return train_data, val_data, test_data, train_loader, val_loader, test_loader, val_meta, test_meta
