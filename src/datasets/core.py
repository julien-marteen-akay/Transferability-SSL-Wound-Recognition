from typing import Union
import torch
import lightning as L
import numpy as np
import pandas as pd
import os
from lightning.pytorch import seed_everything
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# internals
from ..utils.data import get_transforms, get_augmentation_transforms
from ..datasets.wrappers import DatasetCollection


class Dataset(torch.utils.data.Dataset):
    seed = 42
    def __init__(self,
                 *,
                 dataset_name: str,
                 transform=get_transforms(),
                 target_transform=torch.tensor,
                 min_samples: int = 0,
                 keep_classes: Union[list, tuple, set] = None,
                 seed=None):
        seed = seed or self.seed

        # unwrap dataset info
        info = DatasetCollection.get_dataset_info(dataset_name=dataset_name)
        df = info["df"]
        target_col = info["target_col_name"]  # This column contains the actual name of the classes before being encoded by the LabelEncoder
        path_col = info["path_col_name"]
        
        # do preprocessing here
        seed_everything(seed)
        if keep_classes:
            df = df[df[target_col].isin(keep_classes)]
        if min_samples:
            counts = df[target_col].value_counts()
            df = df[df[target_col].isin(counts[counts >= min_samples].index)]
        
        # then encode
        label_enc = LabelEncoder()
        df.loc[:, "class_index"] = label_enc.fit_transform(df.loc[:, target_col])  # this hardcoded string will be used from now on to access the actual labels in training
        
        # set dataset attributes
        self.label_enc = label_enc
        self.df = df
        self.target_col = target_col
        self.path_col = path_col
        self.is_test_col = info["is_test_col_name"]  # A dataset may come with a predefined test split
        self.dataset_name = dataset_name
        self.transform = transform if transform else (lambda x: x)  # if no transform is returned, we use the identity
        self.target_transform = target_transform if target_transform else torch.tensor  # to_tensor if no target_transform prov.
        self.seed = seed

    
    @property
    def num_classes(self):
        return len(self.label_enc.classes_)
    
    def set_stratified_split_indices(self, label_column: str = "class_index", val_split: float = 0.1, test_split: float = 0.2):
        """Computes a stratified split and stores the train, val and test indices as instance attributes"""
        # we must implement validation and test splits. I do not allow the flexibility of discarding the val dataset
        assert val_split > .0 and test_split > 0., ("I do not allow the flexibility of discarding validation and test sets."
                                                    f"You provided as arguments: val_split={val_split} and test_split={test_split}")
        
        # seed and apply stratified splits
        seed_everything(self.seed)
        labels = np.squeeze(self.df[label_column].values)
        
        # from all take the test split cut
        if self.is_test_col:
            # either the dataset comes with a predefined test split --> then just retrieve the training vs test splits
            test_indices = self.df[self.df[self.is_test_col]].index.values  # df where the column of the df ("is_test" column) is True
            train_indices = self.df[~self.df[self.is_test_col]].index.values  # inverse of above i.e. the train indices
            # update test_split retrospectively
            test_split = len(test_indices) / len(self.df)
        else:
            # or it doesn't i.e. the test split is taken randomly from the full data set
            train_indices, test_indices = train_test_split(
                np.arange(len(labels)), test_size=test_split, stratify=labels, random_state=self.seed)

        # From the rest take the val split cut.
        # Update val_split as it is supposed to be a fraction of the full dataset
        # However in the code we take the data only from the training dataset, therefore the val_split fraction needs to be increased.
        val_split = val_split / (1 - test_split)
        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_split, stratify=labels[train_indices], random_state=self.seed)
        
        # set indices
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def set_to_eval_mode(self, keep_aspect_ratio: bool = False):
        """When splitting this Dataset to a Subset the validation and test Datasets are going to apply augmentation as well.
        Call this function for the val and test datasets to update the image transforms to not contain any augmentation.
        
        Args:
            keep_aspect_ratio: Apply padding to validation and test dataset in order to preserve aspect ratio when resizing.
        """
        self.transform = get_transforms(keep_aspect_ratio=keep_aspect_ratio)

    # -----
    # Hooks
    # -----
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.transform(self.df[self.path_col].iloc[index]), self.target_transform(self.df["class_index"].iloc[index])
    

class DatasetTransformWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y


class DataModule(L.LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 batch_size: int = 32, val_batch_size: int = 1, test_batch_size: int = 1,
                 val_split: float = 0.1, test_split: float = 0.2,
                 min_samples: int = 0,
                 keep_classes: Union[list, tuple, set] = None,
                 augmentation: bool = True,
                 augmentation_p: float = 0.05,
                 keep_aspect_ratio: bool = False,
                 keep_aspect_ratio_eval: bool = False,
                 num_workers: int = 0,
                 seed: int = 42):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.min_samples = min_samples
        self.keep_classes = keep_classes
        self.augmentation = augmentation
        self.augmentation_p = augmentation_p
        self.keep_aspect_ratio = keep_aspect_ratio
        self.keep_aspect_ratio_eval = keep_aspect_ratio_eval
        self.num_workers = os.cpu_count() // 4 if num_workers == -1 else num_workers
        self.seed = seed

    def _create_dataset(self) -> Dataset:
        #augmentation = None
        #if self.augmentation:
        #    augmentation = get_augmentation_transforms()
        dataset = Dataset(
            dataset_name=self.dataset_name,
            transform=None,  # get_transforms(augmentations=augmentation, keep_aspect_ratio=self.keep_aspect_ratio),
            min_samples=self.min_samples,
            keep_classes=self.keep_classes,
            seed=self.seed
        )
        return dataset
    
    # -----
    # Hooks
    # -----
    def prepare_data(self) -> None:
        # Here you can perform any data setup operations such as downloading or preprocessing
        # After creating the dataset, we can set some more useful attributes about the data at hand
        self.num_classes = self._create_dataset().num_classes  # property

    def setup(self, stage=None) -> None:
        """
        Arguments:
            - stage (str): Setup should be implemented conditionally wrt the provided <stage> argument.
                           <stage> will be one of the following: fit, validate, test, predict
        """
        # Load the dataset
        dataset = self._create_dataset()

        # prepare transforms
        augmentations = get_augmentation_transforms(p=self.augmentation_p) if self.augmentation else None
        training_transform = get_transforms(add_to_tensor_transform=True, keep_aspect_ratio=self.keep_aspect_ratio, augmentations=augmentations)
        evaluation_transform = get_transforms(add_to_tensor_transform=True, keep_aspect_ratio=self.keep_aspect_ratio_eval)

        # Split the dataset into train, validation, and test sets
        dataset.set_stratified_split_indices(val_split=self.val_split, test_split=self.test_split)
        self.train_dataset = DatasetTransformWrapper(torch.utils.data.Subset(dataset, dataset.train_indices), transform=training_transform)
        self.val_dataset = DatasetTransformWrapper(torch.utils.data.Subset(dataset, dataset.val_indices), transform=evaluation_transform)
        self.test_dataset = DatasetTransformWrapper(torch.utils.data.Subset(dataset, dataset.test_indices), transform=evaluation_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers
        )
