import pdb
import os
import random
import math
from loguru import logger

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

from .modeling_AixelNet import AixelNetFeatureExtractor

TYPE_TO_SCHEDULER_FUNCTION = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_schedule_with_warmup,
}

class TrainDataset(Dataset): 
    """
    Dataset class for training data.
    """
    def __init__(self, trainset):
        """
        Initialize with training data.
        """
        (self.inputs, self.y, self.df), self.table_flag = trainset

    def __len__(self):
        if self.inputs['x_num'] is not None:
            return self.inputs['x_num'].shape[0]
        else:
            return self.inputs['x_cat_input_ids'].shape[0]
        
    def __getitem__(self, index):
        if self.inputs['x_cat_input_ids'] is not None:
            x_cat_input_ids = self.inputs['x_cat_input_ids'][index:index+1]
            x_cat_att_mask = self.inputs['x_cat_att_mask'][index:index+1]
            col_cat_input_ids = self.inputs['col_cat_input_ids']
            col_cat_att_mask = self.inputs['col_cat_att_mask']
        else:
            x_cat_input_ids = None
            x_cat_att_mask = None
            col_cat_input_ids = None
            col_cat_att_mask = None

        if self.inputs['x_num'] is not None:
            x_num = self.inputs['x_num'][index:index+1]
            num_col_input_ids = self.inputs['num_col_input_ids']
            num_att_mask = self.inputs['num_att_mask']
        else:
            x_num = None        
            num_col_input_ids = None
            num_att_mask = None

        if self.y is not None:
            y = self.y.iloc[index:index+1]
        else:
            y = None
        df = self.df

        return  x_cat_input_ids, x_cat_att_mask, x_num, col_cat_input_ids, col_cat_att_mask, num_col_input_ids, num_att_mask, y, self.table_flag, df


class SupervisedTrainCollator():
    def __init__(self,
        **kwargs,
        ):
        pass


    def __call__(self, data):
        if data[0][0] is not None:
            x_cat_input_ids = torch.cat([row[0] for row in data], dim=0)
        else:
            x_cat_input_ids = None
        
        if data[0][1] is not None:
            x_cat_att_mask = torch.cat([row[1] for row in data], dim=0)
        else:
            x_cat_att_mask = None

        if data[0][2] is not None:
            x_num = torch.cat([row[2] for row in data], dim=0)  # [sum_of_batch, num_cols]
        else:
            x_num = None

        col_cat_input_ids = data[0][3] 
        col_cat_att_mask  = data[0][4]
        num_col_input_ids = data[0][5]
        num_att_mask      = data[0][6]

        y = None
        if data[0][7] is not None:
            y = pd.concat([row[7] for row in data], axis=0)

        table_flag = data[0][8]
        df = data[0][9]

        inputs = {
            'x_cat_input_ids': x_cat_input_ids,
            'x_cat_att_mask':  x_cat_att_mask,
            'x_num':           x_num,
            'col_cat_input_ids': col_cat_input_ids,
            'col_cat_att_mask':  col_cat_att_mask,
            'num_col_input_ids': num_col_input_ids,  # shape => [1, num_cols, token_len]
            'num_att_mask':      num_att_mask        # shape => [1, num_cols, token_len]
        }

        return inputs, y, df, table_flag

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def get_scheduler(
    name,
    optimizer,
    num_warmup_steps = None,
    num_training_steps = None,
    ):
    name = name.lower()
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == 'constant':
        return schedule_func(optimizer)
    
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == 'constant_with_warmup':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
    
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

class LinearWarmupScheduler:
    def __init__(
        self,
        optimizer,
        base_lr,
        warmup_epochs,

        warmup_start_lr=-1,
        warmup_ratio=0.1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs

        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else base_lr*warmup_ratio

    def step(self, cur_epoch):
        if cur_epoch < self.warmup_epochs:
            self._warmup_lr_schedule(
                step=cur_epoch,
                optimizer=self.optimizer,
                max_step=self.warmup_epochs,
                init_lr=self.warmup_start_lr,
                max_lr=self.base_lr,
            )
        elif cur_epoch == self.warmup_epochs:
            self._set_lr(self.optimizer, self.base_lr)
    
    def init_optimizer(self):
        self._set_lr(self.optimizer, self.warmup_start_lr)

    def _warmup_lr_schedule(self, optimizer, step, max_step, init_lr, max_lr):
        """Warmup the learning rate"""
        lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
        self._set_lr(optimizer, lr)
    
    def _set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
