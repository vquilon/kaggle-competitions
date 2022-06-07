import math
import os
import random
import time
from typing import Optional, Any

import numpy as np
import scipy
import torch
import transformers
from pytorch_lightning.utilities.types import _LRScheduler
from torch import optim
from torch.optim import Optimizer, lr_scheduler


class KaggleFunctions:
    @staticmethod
    def items_not_in_list(_left, _right, how="left"):
        _list = _left if how == "left" else _right
        _other = _right if how == "left" else _left
        _res = []
        for _item in _list:
            if _item not in _other:
                _res.append(_item)

        return _res

    @classmethod
    def combine_dicts(cls, *_dicts, how="left"):
        """
        >>> KaggleFunctions.combine_dicts({'claveA': [1,2,3,4]}, {'claveA': [5,6,7,8]})
        {'claveA': [1, 2, 3, 4, 5, 6, 7, 8]}
        >>> KaggleFunctions.combine_dicts({'claveA': []}, {'claveB': [5,6,7,8]})
        {'claveA': [], 'claveB': [5, 6, 7, 8]}


        :param _dicts:
        :param how:
        :return:
        """
        _left_dict = {} if len(_dicts) == 0 else _dicts[0]
        if not _left_dict or len(_dicts) <= 1:
            return _left_dict

        _right_dict = _dicts[1]
        _other_dicts = _dicts[2:]

        for key in _left_dict:
            if key in _right_dict:
                # COMBINAR VALOR RECURSIVAMENTE
                # AMBOS SON LISTAS
                _left = _left_dict[key]
                _right = _right_dict[key]
                _res = None
                if isinstance(_left, (tuple, list)) and isinstance(_right, (tuple, list)):
                    _type = type(_right)
                    if how == "left":
                        _type = type(_left)

                    if _type == list:
                        _res = [*_left, *_right]
                    elif _type == tuple:
                        _res = (*_left, *_right)

                # AMBOS SON DICCIONARIOS
                elif isinstance(_left, dict) and isinstance(_right, dict):
                    _res = cls.combine_dicts(_left, _right, how=how)
                # CADA UNO TIENE UN TIPO DIFERENTE (SE COGE EL VALOR DEPENDIENDO DEL TIPO DE MERGE)
                if how != "left":
                    _res = _right

                _left_dict[key] = _res

        # Se combinan las claves del derecho que no esten en el izquierdo agregandolas
        no_in_left_dict = cls.items_not_in_list(
            list(_right_dict.keys()), list(_left_dict.keys()),
            how=how if how == "left" else "right"
        )
        for key in no_in_left_dict:
            _left_dict[key] = _right_dict[key]

        # Se continua combiando el resto de diccionarios
        if len(_other_dicts) > 0:
            _left_dict = cls.combine_dicts(_left_dict, *_other_dicts)

        return _left_dict

    @staticmethod
    def seed_everything(seed: Optional[int] = None) -> int:
        if seed is None:
            seed = random.randint(0, 100000)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        os.environ['PC_SEED'] = str(seed)

        return seed

    @staticmethod
    def get_scheduler(name: str, parameters: Any, optimizer: _LRScheduler, from_transformers: bool = False) -> _LRScheduler:
        """
        Returns scheduler with given name and parameters.
        If failed to import from PyTorch, the function will try to import from HuggingFace Transformers library (if available).
        Raises:
            AttributeError - raised when given scheduler is not available/provided.
        """

        try:
            if not from_transformers:
                instance = getattr(lr_scheduler, name)
                scheduler = instance(optimizer=optimizer, **parameters)
            else:
                raise AttributeError()

        except AttributeError as exception:
            try:
                instance = getattr(transformers, name)
                scheduler = instance(optimizer=optimizer, **parameters)
            except AttributeError as exception:
                raise AttributeError(f"Given scheduler's name is not provided.")

        return scheduler

    @staticmethod
    def get_optimizer(name: str, model_parameters: Any, parameters: Any , from_transformers: bool = False) -> Optimizer:
        """
        Returns optimizer with given name and parameters.
        If failed to import from PyTorch, the function will try to import from HuggingFace Transformers library.
        Raises:
            AttributeError - raised when given optimizer is not available/provided.
        """
        def _load_function(module, _name):
            try:
                return getattr(module, _name)
            except AttributeError as exception:
                raise AttributeError(f"Given optimizer's name is not provided.")

        if not from_transformers:
            instance = _load_function(optim, name)
        else:
            instance = _load_function(transformers, name)

        optimizer = instance(params=model_parameters, **parameters)

        return optimizer

    @staticmethod
    def get_pearsonr_score(y_true, y_pred):
        score = scipy.stats.pearsonr(y_true, y_pred)[0]
        return score


    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    @classmethod
    def time_since(cls, since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (remain %s)' % (cls.as_minutes(s), cls.as_minutes(rs))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


