import gc
import os
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import json


DEFAULT_INPUT_FILE_NAME = "mminput.data"
DEFAULT_LABELS_FILE_NAME = "mmlabels.data"
DEFAULT_SAMPLES_FILE_NAME = "mmsamples.data"
DEFAULT_CFG_FILE_NAME = "mmcfg.data"
DEFAULT_MM_PATH = "mm/"

STOCK_CODE_LEN=6

class MMDataset(Dataset):
    # @classmethod
    # def mmdataexist(cls):
    def __init__(
        self,
        mmap_path: str = DEFAULT_MM_PATH,
        seq_name: str = None,
        size: int = None,
        input_shape: tuple = None,
        label_shape: tuple = None,
        sample_type: np.dtype = np.dtype('float64'),
        transform_fn: Callable[..., Any] = None,
    ) -> None:
        super().__init__()

        self.mmap_inputs: np.ndarray = None
        self.mmap_labels: np.ndarray = None
        # self.mmap_samples: np.ndarray = None
        self.transform_fn = transform_fn

        self.input_shapes = input_shape
        self.label_shape = label_shape
        self.size = size

        if mmap_path is None:
            mmap_path = os.path.abspath(os.getcwd())
        self._mkdir(mmap_path)

        self.mmap_cfg_path = os.path.join(mmap_path, "_".join([seq_name, DEFAULT_CFG_FILE_NAME]))
        if os.path.exists(self.mmap_cfg_path):
            with open(self.mmap_cfg_path) as f:
                cfg = json.load(f)
                self.length = cfg["mmlen"]
                self.mmap_input_path = cfg["input"]
                self.mmap_labels_path = cfg["label"]
                self.size = cfg["size"]
                remove_existing = False
        else:
            self.length = 0
            self.mmap_input_path = os.path.join(mmap_path, "_".join([seq_name, DEFAULT_INPUT_FILE_NAME]))
            self.mmap_labels_path = os.path.join(mmap_path, "_".join([seq_name, DEFAULT_LABELS_FILE_NAME]))
            remove_existing = True
            # self.mmap_samples_path = os.path.join(mmap_path, DEFAULT_SAMPLES_FILE_NAME)

        self.mmap_inputs = self._init_mmap(
            self.mmap_input_path, sample_type, (size, *input_shape), remove_existing=remove_existing
        )
        self.mmap_labels = self._init_mmap(
            self.mmap_labels_path, sample_type, (size, *label_shape), remove_existing=remove_existing
        )
        print("shape", self.mmap_input_path, sample_type, (size, *input_shape), (size, *label_shape))

    def map(self, idx, input, label):
        print("map", self.mmap_input_path, idx, self.mmap_inputs.shape, input.shape, label.shape)
        self.mmap_inputs[idx][:] = input[:]
        self.mmap_labels[idx][:] = label[:]
        self.length = max(self.length, idx+1)

    def save(self):
        print("save", self.mmap_input_path)
        self.mmap_inputs.flush()
        self.mmap_labels.flush()
        with open('data.json', 'w') as f:
            json.dump({ "mmlen": self.length, "size": self.size, "input":self.mmap_input_path, "label":self.mmap_labels_path }, f)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, torch.Tensor, ]:
        # if self.transform_fn:
        #     return self.transform_fn(self.mmap_inputs[idx]), torch.tensor(self.mmap_labels[idx])
        print("get", self.mmap_input_path, idx)
        return torch.FloatTensor(self.mmap_inputs[idx]), torch.FloatTensor([self.mmap_labels[idx][0]]), \
               str(self.mmap_labels[idx][1]).zfill(STOCK_CODE_LEN), int(self.mmap_labels[idx][2]), self.mmap_labels[idx][3]>0


    def __len__(self) -> int:
        return self.length


    def _consume_iterable(self, input_iter: Iterable[np.ndarray], labels_iter: Iterable[np.ndarray]) -> Tuple[List[np.ndarray]]:
        inputs = []
        labels = []

        for input, label in zip(input_iter, labels_iter):
            inputs.append(input)
            labels.append(label)

        if len(inputs) != len(labels):
            raise Exception(
                f"Input samples count {len(inputs)} is different than the labels count {len(labels)}"
            )

        if not isinstance(inputs[0], np.ndarray):
            raise TypeError("Inputs and labels must be of type np.ndarray")

        return inputs, labels


    def _mkdir(self, path: str) -> None:
        if os.path.exists(path):
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return
        except:
            raise ValueError(
                "Failed to create the path (check the user write permissions)."
            )


    def _init_mmap(self, path: str, dtype: np.dtype, shape: Tuple[int], remove_existing: bool = False) -> np.ndarray:
        open_mode = "w+" if remove_existing else "r"
        return np.memmap(
            path,
            dtype=dtype,
            mode=open_mode,
            shape=shape,
        )

class MMAPDataset(Dataset):
    def __init__(
        self,
        input_iter: Iterable[np.ndarray],
        labels_iter: Iterable[np.ndarray],
        mmap_path: str = None,
        size: int = None,
        transform_fn: Callable[..., Any] = None,
    ) -> None:
        super().__init__()

        self.mmap_inputs: np.ndarray = None
        self.mmap_labels: np.ndarray = None
        self.transform_fn = transform_fn

        if mmap_path is None:
            mmap_path = os.path.abspath(os.getcwd())
        self._mkdir(mmap_path)

        self.mmap_input_path = os.path.join(mmap_path, DEFAULT_INPUT_FILE_NAME)
        self.mmap_labels_path = os.path.join(mmap_path, DEFAULT_LABELS_FILE_NAME)

        # If the total size is not known we load the dataset in memory first
        if size is None:
            input_iter, labels_iter = self._consume_iterable(input_iter, labels_iter)
            size = len(input_iter)

        self.length = size

        for idx, (input, label) in enumerate(zip(input_iter, labels_iter)):
            if self.mmap_inputs is None:
                self.mmap_inputs = self._init_mmap(
                    self.mmap_input_path, input.dtype, (self.length, *input.shape)
                )
                self.mmap_labels = self._init_mmap(
                    self.mmap_labels_path, label.dtype, (self.length, *label.shape)
                )

            self.mmap_inputs[idx][:] = input[:]
            self.mmap_labels[idx][:] = label[:]

        del input_iter
        del labels_iter
        gc.collect()


    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor]]:
        if self.transform_fn:
            return self.transform_fn(self.mmap_inputs[idx]), torch.tensor(self.mmap_labels[idx]) 
        return self.mmap_inputs[idx], self.mmap_labels[idx]


    def __len__(self) -> int:
        return self.length


    def _consume_iterable(self, input_iter: Iterable[np.ndarray], labels_iter: Iterable[np.ndarray]) -> Tuple[List[np.ndarray]]:
        inputs = []
        labels = []

        for input, label in zip(input_iter, labels_iter):
            inputs.append(input)
            labels.append(label)

        if len(inputs) != len(labels):
            raise Exception(
                f"Input samples count {len(inputs)} is different than the labels count {len(labels)}"
            )

        if not isinstance(inputs[0], np.ndarray):
            raise TypeError("Inputs and labels must be of type np.ndarray")

        return inputs, labels


    def _mkdir(self, path: str) -> None:
        if os.path.exists(path):
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return
        except:
            raise ValueError(
                "Failed to create the path (check the user write permissions)."
            )


    def _init_mmap(self, path: str, dtype: np.dtype, shape: Tuple[int], remove_existing: bool = False) -> np.ndarray:
        open_mode = "w+" if remove_existing else "r+"
        return np.memmap(
            path,
            dtype=dtype,
            mode=open_mode,
            shape=shape,
        )
