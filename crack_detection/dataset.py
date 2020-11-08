
"""Module to construct the dataset."""

from typing import List, Tuple

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


class Mode(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2


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

    def get_image(self) -> np.ndarray:
        """Return a numpy array of the image."""
        
        return np.array(cv2.imread(self.file_path.as_posix(), cv2.IMREAD_COLOR))



class Dataset:
    """Dataset module."""

    def __init__(self, in_dir: Path, exts : List[str] = ['.jpg'], split: float = 0.7, batch_size : int = 32) -> None:
        """Initialize module by creating the datasets."""
        
        # Number of classes 
        self.num_classes = 2
        
        # Train/val split
        self.split = split

        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        # Which mode.
        self.mode = Mode.TRAIN

        self.batch_size = batch_size
        
        # Create datasets.
        self._create_datasets(in_dir)

    def _create_datasets(self, in_dir: Path) -> None:
        """Creates the two testing and training datasets."""
        
        dirs = [x.parts[-1] for x in in_dir.iterdir() if x.is_dir()]
        assert TRAIN_SET_DIR in dirs
        assert TEST_SET_DIR in dirs
        
        self.training_set = self._create_set(in_dir / TRAIN_SET_DIR)
        self.testing_set = self._create_set(in_dir / TEST_SET_DIR)
    
    def _create_set(self, folder_path: Path) -> List[Datum]:
        """Create train/test set."""
        
        # Check if 'crack' and 'no_crack' folders exist in the train/test dir.
        class_dirs = [x.parts[-1] for x in folder_path.iterdir() if x.is_dir()]
        assert CRACK_DIR in class_dirs
        assert NO_CRACK_DIR in class_dirs

        # Create list of datums
        crack_datum = self._create_datum(folder_path / CRACK_DIR, Label.CRACK)
        no_crack_datum = self._create_datum(folder_path / NO_CRACK_DIR, Label.NO_CRACK)

        return crack_datum + no_crack_datum

    def _create_datum(self, folder: Path, label: Label) -> List[Datum]:
        """Creates a list of Crack or no no crack datums.
        
        Finds all the files in the folder with `exts` and converts to a datum for training.
        """

        set_datum = []
        one_hot = encode_to_one_hot(label, self.num_classes)
        for _file in folder.iterdir():
            if _file.is_file() and _file.suffix in self.exts:
                set_datum.append(Datum(file_path=_file, label = one_hot))

        return set_datum

    def switch_mode(self, mode: Mode) -> None:
        """Convert mode to train or test."""

        self.mode = mode

    def random_batch(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get a random batch."""

        dataset = self.training_set if self.mode == Mode.TRAIN else self.testing_set

        # Create a random index.
        idx = np.random.choice(len(dataset),
                               size=self.batch_size,
                               replace=False)

        return [[dataset[i].get_image(), dataset[i].label] for i in idx] 
        

if __name__=="__main__":
    dataset = Dataset(Path('/home/satyen/repos/Concrete-Crack-Detection/dataset'))
    batch = dataset.random_batch()