import os
import pandas as pd
from abc import ABC, abstractmethod


class DatasetWrapper(ABC):
    """
    A Wrapper class, forcing subclasses to implement a classmethod called get_dataset_info.
    """
    @classmethod
    @abstractmethod
    def get_dataset_info(cls) -> dict:
        """
        Abstract method to be implemented by the subclasses.
        Returns a dictionary containing the following objects:
        - 'df': The pandas DataFrame of the dataset.
        - 'target_col_name': The target column name as a string
        - 'path_col_name': The path column name as a string
        - 'is_test_col_name': The name of the column (str) containing True or False values indicating if the sample belongs to the test set. None if not available.
        """
        pass


class Medetec(DatasetWrapper):
    @classmethod
    def get_dataset_info(cls) -> dict:
        info = {
            "df": cls.get_df(),
            "target_col_name": "class",
            "path_col_name": "path",
            "is_test_col_name": None
            }
        return info
    
    @classmethod
    def get_df(cls):
        df = pd.read_csv(os.path.join(os.getcwd(), "data", "medetec", "dataframe.csv"))
        return df.reset_index(drop=True)
    

class AZH(DatasetWrapper):
    """
    Dataset from here: https://github.com/uwm-bigdata/wound-classification-using-images-and-locations
    Paper we compare our work with, on the same benchmark: https://www.nature.com/articles/s41598-022-21813-0
    """
    base = os.path.join(os.getcwd(), "data", "Multi-modal-wound-classification-using-images-and-locations")  # "/home/akay_ju/master/transferability_of_non_contrastive_ssl_repo/data/Multi-modal-wound-classification-using-images-and-locations"
    @classmethod
    def get_dataset_info(cls) -> dict:
        info = {
            "df": cls.get_df(),
            "target_col_name": "Labels",
            "path_col_name": "path",
            "is_test_col_name": "is_test"
            }
        return info

    @classmethod
    def get_df(cls):
        df = pd.read_csv(os.path.join(cls.base, "dataframe.csv"))
        df["path"] = df["path"].apply(lambda path: os.path.join(cls.base, path))
        return df.reset_index(drop=True)
    

class DatasetCollection:
    string_to_class = {
        "medetec": Medetec,
        "azh": AZH
        }
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> DatasetWrapper:
        """
        Returns a dictionary containing the following objects:
        - 'df': The pandas DataFrame of the dataset.
        - 'target_col_name': The target column name as a string
        - 'path_col_name': The path column name as a string
        - 'is_test_col_name': The name of the column (str) containing True or False values indicating if the sample belongs to the test set. None if not available.
        """
        # 1.) Gets class from string-identifier
        # 2.) Checks if the class subclassed DatasetWrapper
        # 3.) If true, returns the class w/out errors and runs the .get_dataset_info() method of the class to return the requested dataset info.
        return cls.check(cls.string_to_class[dataset_name]).get_dataset_info()
    
    @classmethod
    def check(cls, obj: DatasetWrapper) -> DatasetWrapper:
        """Behaves like identity function if obj inherited from DatasetWrapper, otherwise throws an error."""
        if not issubclass(obj, DatasetWrapper):
            raise ValueError(
                f"The dataset_name you provided refers to a class that does not inherit from the DatasetWrapper class, which is a must."
                )
        return obj
    