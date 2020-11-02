
"""Module to construct the dataset."""

from typing import List

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import cv2
from enum import IntEnum


CRACK_DIR = 'crack'
NO_CRACK_DIR = 'no_crack'
TRAIN_SET_DIR = 'train_set'
TEST_SET_DIR = 'test_set'

class Label(IntEnum):
    NO_CRACK = 0
    CRACK = 1

def encode_to_one_hot(class_label : Label, num_classes: int):
    """
    Generate the One-Hot encoded class-label.
    
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    
    Args:
        class_label: Class number. Should be in [0, num_classes -1].
        num_classes: Number of classes. 

    Returns:
        1D array of shape: [num_classes]
    """
    
    assert class_label < num_classes

    one_hot = np.zeros((num_classes))
    one_hot[class_label] = 1 
    
    return one_hot


@dataclass
class Datum:
    """
    Class to hold the filename and corresponding label.
    
    Args:
        file_path: The path to the jpg file.
        label: one hot label. 
    """

    file_path : Path = field(default=Path(""))
    label : List[float] = field(default_factory=list)

    def get_image(self):
        return cv2.imread(self.file_path, 1)


class DataSet:
    def __init__(self, in_dir: Path, exts : List[str] = ['.jpg']):
        
        # Number of classes 
        self.num_classes = 2

        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        self._create_datasets(in_dir)

    def _create_datasets(self, in_dir: Path):
        
        dirs = [x.parts[-1] for x in in_dir.iterdir() if x.is_dir()]
        assert TRAIN_SET_DIR in dirs
        assert TEST_SET_DIR in dirs
        
        self.training_set = self._create_set(in_dir / TRAIN_SET_DIR)
        self.testing_set = self._create_set(in_dir / TEST_SET_DIR)
    
    def _create_set(self, folder_path: Path):

        class_dirs = [x.parts[-1] for x in folder_path.iterdir() if x.is_dir()]
        assert CRACK_DIR in class_dirs
        assert NO_CRACK_DIR in class_dirs

        crack_datum = self._create_datum(folder_path / CRACK_DIR, Label.CRACK)
        no_crack_datum = self._create_datum(folder_path / NO_CRACK_DIR, Label.NO_CRACK)

        return crack_datum + no_crack_datum

    def _create_datum(self, folder: Path, label: Label):

        set_datum = []
        one_hot = encode_to_one_hot(label, self.num_classes)
        for _file in folder.iterdir():
            if _file.is_file() and _file.suffix in self.exts:
                set_datum.append(Datum(file_path=_file, label = one_hot))

        return set_datum


if __name__=="__main__":
    dataset = DataSet(Path('/home/satyen/repos/Concrete-Crack-Detection/dataset'))