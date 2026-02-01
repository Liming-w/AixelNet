import os
import shutil
import pdb
import math
import time
import json
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from loguru import logger
import logging

from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR, CosineAnnealingLR

from . import constants
from .evaluator import predict, get_eval_metric_fn, EarlyStopping, evaluate, predict_new
from .modeling_AixelNet import AixelNetFeatureExtractor
from .trainer_utils import SupervisedTrainCollator, TrainDataset
from .trainer_utils import get_parameter_names
from .trainer_utils import get_scheduler, LinearWarmupScheduler

class Trainer:
    def __init__(self,
        model,
        train_set_list,
        test_set_list=None,
        collate_fn=None,
        output_dir='./ckpt',
        num_epoch=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=0,
        patience=5,
        eval_batch_size=256,
        warmup_ratio=None,
        warmup_steps=None,
        balance_sample=False,
        load_best_at_last=True,
        ignore_duplicate_cols=True,
        eval_metric='auc',
        eval_less_is_better=False,
        num_workers=0,
        regression_task=False,
        flag=0,
        data_weight=None,
        device=None,
        lambda1=1e-4,
        lambda2=1e-4,
        **kwargs,
        ):
        # Initialize the Trainer with model, datasets, and hyperparameters.
        self.flag = flag
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.data_weight = data_weight
        if isinstance(train_set_list, tuple): train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple): test_set_list = [test_set_list]

        self.collate_fn = collate_fn
        self.regression_task = regression_task
        if collate_fn is None:
            # Use SupervisedTrainCollator by default if no custom collate function is provided.
            self.collate_fn = SupervisedTrainCollator()

        if isinstance(model, nn.DataParallel):
            real_model = model.module
        else:
            real_model = model
        # Initialize the feature extractor for the model.
        self.feature_extractor = AixelNetFeatureExtractor(
            categorical_columns=real_model.categorical_columns,
            numerical_columns=real_model.numerical_columns,
            binary_columns=real_model.binary_columns,
            disable_tokenizer_parallel=True,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )

        # Process training dataset list and extract features
        new_train_list = []
        for dataindex, trainset in enumerate(train_set_list):
            (X, y, df), table_flag = trainset
            inputs = self.feature_extractor(X, table_flag=table_flag)
            new_train_list.append(((inputs, y, df), table_flag))

        # Create DataLoader for each train set
        self.trainloader_list = [
            torch.utils.data.DataLoader(
                TrainDataset(trainset),  # trainset: ((inputs,y,df), table_flag)
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for trainset in new_train_list
        ]

        # Similarly process test set list if available
        new_test_list = []
        if test_set_list is not None:
            for dataindex, testset in enumerate(test_set_list):
                (Xv, yv, dfv), tfv = testset
                inputs_val = self.feature_extractor(Xv, table_flag=tfv)
                new_test_list.append(((inputs_val, yv, dfv), tfv))
            self.testloader_list = [
                torch.utils.data.DataLoader(
                    TrainDataset(testset),
                    collate_fn=self.collate_fn,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=False,
                )
                for testset in new_test_list
            ]
        else:
            self.testloader_list = None

        self.train_set_list = new_train_list
        self.test_set_list = new_test_list
        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(output_dir=output_dir, patience=patience, verbose=False, less_is_better=eval_less_is_better)
        self.args = {
            'lr':lr,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'num_epoch':num_epoch,
            'eval_batch_size':eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'eval_metric': get_eval_metric_fn(eval_metric),
            'eval_metric_name': eval_metric,
        }

        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

        self.optimizer = None
        self.lr_scheduler = None
        self.lambda1 = kwargs.get("lambda1", 1e-4)
        self.lambda2 = kwargs.get("lambda2", 1e-4)

    def train(self, eval_data=None):
        # Start training the model, evaluate periodically, and apply early stopping
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            logger.info(f'Set warmup training in initial {args["warmup_steps"]} steps')
            self.lr_scheduler = LinearWarmupScheduler(
                optimizer=self.optimizer, 
                base_lr=self.args['lr'],
                warmup_epochs=args['warmup_steps'],
            )
            self.lr_scheduler.init_optimizer()

        start_time = time.time()
        real_res_list = []

        for epoch in range(args['num_epoch']):
            train_loss_all = 0
            self.model.train()
            # Iterate over multiple train datasets
            for dataindex, trainloader in enumerate(self.trainloader_list):
                for batch_idx, batch in enumerate(trainloader):
                    # Batch contains: inputs, y, df, table_flag
                    inputs, y, df, table_flag = batch
                    x_data = inputs  # The actual batch data
                    
                    self.optimizer.zero_grad()
                    if y is not None:
                        y = torch.tensor(y.values).float()
                    if inputs['x_num'] is not None and torch.isnan(inputs['x_num']).any():
                        logger.error("NaN found in inputs[x_num]")
                    if y is not None and torch.isnan(y).any():
                        logger.error("NaN found in y")
                    
                    # Model forward pass with meta feature generation based on table_flag and df
                    logits, loss = self.model(x_data, y, df, table_flag, sparse=False, M=2, lambda1=self.lambda1, lambda2=self.lambda2)
                    loss.backward()
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    train_loss_all += loss.item()


            if self.lr_scheduler is not None:
                self.lr_scheduler.step(cur_epoch=epoch)

            if self.testloader_list is not None:
                eval_res_list = self.evaluate()
                eval_res = np.mean(eval_res_list)
                if self.early_stopping(-eval_res, self.model) and eval_data:
                    if self.regression_task:
                        ypred = predict_new(self.model, eval_data[0][0], df_test=eval_data[0][2], regression_task=True)
                        ans = evaluate(ypred, eval_data[0][1], metric='rmse')
                    else:
                        ypred = predict_new(self.model, eval_data[0][0], df_test=eval_data[0][2])
                        ans = evaluate(ypred, eval_data[0][1], metric='auc', num_class=self.model.num_class)
                    real_res_list.append(ans[0])
                if self.early_stopping.early_stop:
                    logger.info('Early stopped')
                    break

        # Load best model if specified and save model checkpoint
        if os.path.exists(self.output_dir):
            if self.testloader_list is not None:
                logger.info(f'Load best model from {self.output_dir}')
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

        return real_res_list

    def evaluate(self):
        # Evaluate the model on test datasets and return results
        self.model.eval()
        eval_res_list = []
        for dataindex, testloader in enumerate(self.testloader_list):
            y_test, pred_list, loss_list = [], [], []
            with torch.no_grad():
                for batch in testloader:
                    inputs, y, df, table_flag = batch
                    x_data = inputs  # The actual batch data
                    if y is not None:
                        y_test.append(y)
                    
                    logits, loss = self.model(x_data, y, df, table_flag, sparse=False, M=2, lambda1=self.lambda1, lambda2=self.lambda2)
                    
                    if loss is not None:
                        loss_list.append(loss.item())
                    if logits is not None:
                        if self.regression_task:
                            pred_list.append(logits.detach().cpu().numpy())
                        elif logits.shape[-1] == 1: # binary classification
                            pred_list.append(logits.sigmoid().detach().cpu().numpy())
                        else: # multi-class classification
                            pred_list.append(torch.softmax(logits, -1).detach().cpu().numpy())

            if len(pred_list)>0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()

            # Calculate evaluation metric
            if self.args['eval_metric_name'] == 'val_loss':
                eval_res = np.mean(loss_list)
            else:
                y_test = pd.concat(y_test, axis=0)
                if self.regression_task:
                    eval_res = self.args['eval_metric'](y_test, pred_all)
                else:
                    eval_res = self.args['eval_metric'](y_test, pred_all, self.model.num_class)

            eval_res_list.append(eval_res)

        return eval_res_list

    def create_optimizer(self):
        # Create the optimizer for the model
        decay_parameters = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])

    def save_model(self, output_dir=None): 
        # Save the model and optimizer to the specified directory
        if output_dir is None:
            logger.info('No path assigned for save mode, default saving to ./ckpt/model.pt !')
            output_dir = self.output_dir

        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving model checkpoint to {output_dir}')
        self.model.save(output_dir)

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, constants.OPTIMIZER_NAME))

    def create_scheduler(self, num_training_steps, optimizer):
        # Create the learning rate scheduler
        self.lr_scheduler = get_scheduler(
            'cosine',
            optimizer = optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        # Calculate the total number of training steps
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        total_step *= num_epoch
        return total_step

    def get_warmup_steps(self, num_training_steps):
        """
        Get the number of warmup steps for linear warmup.
        """
        warmup_steps = (
            self.args['warmup_steps'] if self.args['warmup_steps'] is not None else math.ceil(num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=True):
        # Build a DataLoader for a given training set
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            )
        return trainloader