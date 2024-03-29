import gc
import os
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import json
from nn_macro import *

DEFAULT_INPUT_FILE_NAME = "mminput_{i}.pkl"
DEFAULT_MLP_INPUT_FILE_NAME = "mminput_mlp.pkl"
DEFAULT_LABELS_FILE_NAME = "mmlabels.pkl"
DEFAULT_INFOS_FILE_NAME = "mminfo.pkl"
DEFAULT_SAMPLES_FILE_NAME = "mmsamples.pkl"
DEFAULT_CFG_FILE_NAME = "mmcfg.pkl"
DEFAULT_MM_PATH = "./"

STOCK_CODE_LEN=6

class MMDataset(Dataset):
    # @classmethod
    # def mmdataexist(cls):
    def __init__(
        self,
        mmap_path: str = DEFAULT_MM_PATH,
        seq_name: str = None,
        size: int = None,
        input_shape: list = None,
        label_shape: tuple = None,
        info_shape: tuple = None,
        sample_type: np.dtype = np.dtype('float32'),
        transform_fn: Callable[..., Any] = None,
    ) -> None:
        super().__init__()

        self.mmap_inputs: list = []
        self.mmap_mlp_inputs: np.ndarray = None
        self.mmap_labels: np.ndarray = None
        self.mmap_infos: np.ndarray = None
        # self.mmap_samples: np.ndarray = None
        self.transform_fn = transform_fn

        self.input_shape = input_shape
        self.label_shape = label_shape
        self.info_shape = info_shape
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
                self.mmap_mlp_input_path = cfg["mlp_input"]
                self.mmap_labels_path = cfg["label"]
                self.mmap_infos_path = cfg["info"]
                self.size = cfg["size"]
                self.input_shape = cfg["input_shape"]
                self.label_shape = cfg["label_shape"]
                self.info_shape = cfg["info_shape"]
                remove_existing = False
        else:
            self.length = 0
            self.mmap_input_path = [os.path.join(mmap_path, "_".join([seq_name, eval(f"f'{DEFAULT_INPUT_FILE_NAME}'")]))
                                    for i in range(len(self.input_shape[EXP_MODS_LSTM_IDX]))]
            self.mmap_mlp_input_path = os.path.join(mmap_path, "_".join([seq_name, DEFAULT_MLP_INPUT_FILE_NAME]))
            self.mmap_labels_path = os.path.join(mmap_path, "_".join([seq_name, DEFAULT_LABELS_FILE_NAME]))
            self.mmap_infos_path = os.path.join(mmap_path, "_".join([seq_name, DEFAULT_INFOS_FILE_NAME]))
            remove_existing = True
            # self.mmap_samples_path = os.path.join(mmap_path, DEFAULT_SAMPLES_FILE_NAME)

        for i in range(len(self.input_shape)):
            self.mmap_inputs += [self._init_mmap(
                self.mmap_input_path[i], sample_type, (self.size, *self.input_shape[EXP_MODS_LSTM_IDX][i]), remove_existing=remove_existing
            )]
        if self.input_shape[EXP_MODS_MLP_IDX] > 0:
            self.mmap_mlp_inputs = self._init_mmap(
                self.mmap_mlp_input_path, sample_type, (self.size, self.input_shape[EXP_MODS_MLP_IDX]), remove_existing=remove_existing
            )
        self.mmap_labels = self._init_mmap(
            self.mmap_labels_path, sample_type, (self.size, *self.label_shape), remove_existing=remove_existing
        )
        self.mmap_infos = self._init_mmap(
            self.mmap_infos_path, sample_type, (self.size, *self.info_shape), remove_existing=remove_existing
        )
        print("shape", self.mmap_input_path, sample_type, (self.size, *self.input_shape), (self.size, *self.label_shape),
              (self.size, *self.info_shape))

    def map(self, idx, input, label, info):
        # print("map", self.mmap_input_path, idx, self.mmap_inputs.shape, self.mmap_inputs.dtype, input.shape, input.dtype, len(label))
        for i in range(len(self.mmap_inputs)):
            self.mmap_inputs[i][idx][:] = input[EXP_MODS_LSTM_IDX][i][:]
        if self.input_shape[EXP_MODS_MLP_IDX] > 0:
            self.mmap_mlp_inputs[idx][:] = input[EXP_MODS_MLP_IDX][:]
        self.mmap_labels[idx][:] = label[:]
        self.mmap_infos[idx][:] = info[:]
        self.length = max(self.length, idx+1)

    def save(self):
        print("save", self.mmap_input_path)
        for i in range(len(self.mmap_inputs)):
            self.mmap_inputs[i].flush()
        if self.input_shape[EXP_MODS_MLP_IDX] > 0:
            self.mmap_mlp_inputs.flush()
        self.mmap_labels.flush()
        self.mmap_infos.flush()
        with open(self.mmap_cfg_path, 'w') as f:
            json.dump({ "mmlen": self.length, "size": self.size, "input":self.mmap_input_path, "mlp_input": self.mmap_mlp_input_path,
                        "label":self.mmap_labels_path, "info":self.mmap_infos_path,
                        "input_shape": self.input_shape, "label_shape": self.label_shape, "info_shape": self.info_shape}, f)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, torch.Tensor, ]:
        # if self.transform_fn:
        #     return self.transform_fn(self.mmap_inputs[idx]), torch.tensor(self.mmap_labels[idx])
        # print("get", self.mmap_input_path, self.length, self.size, idx, "\r\n", self.mmap_inputs[idx][::17], self.mmap_labels[idx])
        inputs = [torch.FloatTensor(self.mmap_inputs[i][idx]) for i in range(len(self.mmap_inputs))]
        inputs = tuple(inputs)
        if self.input_shape[EXP_MODS_MLP_IDX] > 0:
            mlp_input = torch.FloatTensor(self.mmap_mlp_inputs[idx])
        else:
            mlp_input = np.empty(0)
        return inputs, mlp_input, \
               torch.FloatTensor(self.mmap_labels[idx]), \
               str(self.mmap_infos[idx][0]).zfill(STOCK_CODE_LEN), int(self.mmap_infos[idx][1]), self.mmap_infos[idx][2]>0, int(self.mmap_infos[idx][3])


    def __len__(self) -> int:
        return self.length


    # def _consume_iterable(self, input_iter: Iterable[np.ndarray], labels_iter: Iterable[np.ndarray]) -> Tuple[List[np.ndarray]]:
    #     inputs = []
    #     labels = []
    #
    #     for input, label in zip(input_iter, labels_iter):
    #         inputs.append(input)
    #         labels.append(label)
    #
    #     if len(inputs) != len(labels):
    #         raise Exception(
    #             f"Input samples count {len(inputs)} is different than the labels count {len(labels)}"
    #         )
    #
    #     if not isinstance(inputs[0], np.ndarray):
    #         raise TypeError("Inputs and labels must be of type np.ndarray")
    #
    #     return inputs, labels


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

class PklDataset(MMDataset):
    def save(self):
        print("save", self.mmap_input_path)
        for i in range(len(self.mmap_inputs)):
            with open(self.mmap_input_path[i], 'wb') as fileW:
                np.save(fileW, self.mmap_inputs[i])
        if self.input_shape[EXP_MODS_MLP_IDX] > 0:
            with open(self.mmap_mlp_input_path, 'wb') as fileW:
                np.save(fileW, self.mmap_mlp_inputs)
        with open(self.mmap_labels_path, 'wb') as fileW:
            np.save(fileW, self.mmap_labels)
        with open(self.mmap_infos_path, 'wb') as fileW:
            np.save(fileW, self.mmap_infos)
        with open(self.mmap_cfg_path, 'w') as f:
            # json.dump({ "mmlen": self.length, "size": self.size, "input":self.mmap_input_path, "label":self.mmap_labels_path, "info":self.mmap_infos_path,
            #             "input_shape": self.input_shape, "label_shape": self.label_shape, "info_shape": self.info_shape}, f)
            json.dump({ "mmlen": self.length, "size": self.size, "input":self.mmap_input_path, "mlp_input": self.mmap_mlp_input_path,
                        "label":self.mmap_labels_path, "info":self.mmap_infos_path,
                        "input_shape": self.input_shape, "label_shape": self.label_shape, "info_shape": self.info_shape}, f)

    def _init_mmap(self, path: str, dtype: np.dtype, shape: Tuple[int], remove_existing: bool = False) -> np.ndarray:
        open_mode = "w+" if remove_existing else "r+"
        if open_mode == "r+":
            return np.load( path )
        else:
            return np.empty(shape, dtype=dtype)

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
