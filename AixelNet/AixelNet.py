import os
import torch
from torch import nn
from .modeling_AixelNet import AixelNetForClassifier, AixelNetForRegression, AixelNetPretrain
from .modeling_AixelNet import AixelNetInputEncoder, AixelNetFeatureExtractor, AixelNetFeatureProcessor
from .regularization import SegmentBilinearAttentionRegularizer, compute_balance_regularization, compute_prediction_diversity
from .evaluator import predict, evaluate, predict_new
from .trainer import Trainer
from .trainer_utils import random_seed

dev = 'cuda'

def build_classifier(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
    num_class=2,
    k_models=3,
    hidden_dim=128,
    num_layer=3,
    num_attention_head=8,
    hidden_dropout_prob=0.1,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=False,
    use_bert=True,
    device=dev,
    checkpoint=None,
    **kwargs
) -> AixelNetForClassifier:
    """
    Build the AixelNetForClassifier model instance.
    Supports loading model weights from a pre-trained checkpoint.
    """
    model = AixelNetForClassifier(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        feature_extractor=feature_extractor,
        num_class=num_class,
        k_models=k_models,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        use_bert=use_bert,
        device=device,
        **kwargs,
    )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model

def build_regressor(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    dataset_paths=None,
    num_classes_list=None,
    feature_extractor=None,
    k_models=3,
    meta_feature_dim=None,
    hidden_dim=128,
    num_layer=3,
    num_attention_head=8,
    hidden_dropout_prob=0.1,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=False,
    use_bert=True,
    device=dev,
    checkpoint=None,
    **kwargs
) -> AixelNetForRegression:
    """
    Build the AixelNetForRegression model instance.
    Supports loading model weights from a pre-trained checkpoint.
    """
    model = AixelNetForRegression(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        feature_extractor=feature_extractor,
        k_models=k_models,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        use_bert=use_bert,
        device=device,
        **kwargs,
    )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model


def build_pretrain_model(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    dataset_paths=None,
    num_classes_list=None,
    feature_extractor=None,
    k_models=3,
    meta_feature_dim=None,
    hidden_dim=128,
    num_layer=3,
    num_attention_head=8,
    hidden_dropout_prob=0.1,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=False,
    use_bert=True,
    device=dev,
    checkpoint=None,
    wo_arg=None,
    lambda1 = 1e-4,
    lambda2 = 1e-4,
    hyper_arg=None,
    **kwargs
) -> AixelNetPretrain:
    """
    Build the AixelNetPretrain pretraining model.
    """
    if dataset_paths is None or num_classes_list is None:
        raise ValueError("dataset_paths and num_classes_list must not be None")

    if len(dataset_paths) != len(num_classes_list):
        raise ValueError("The number of dataset_paths must match the number of num_classes_list")

    model = AixelNetPretrain(
        num_classes_list=num_classes_list,
        k_models=k_models,
        meta_feature_dim=meta_feature_dim,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        feature_extractor=feature_extractor,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        use_bert=use_bert,
        pool_policy='avg',
        device=device,
        dataset_paths=dataset_paths,
        wo_arg=wo_arg,
        lambda1 = lambda1,
        lambda2 = lambda2,
        hyper_arg=hyper_arg,
        **kwargs
    )

    if checkpoint is not None:
        model.load(checkpoint)
    return model


def train(
    model, 
    trainset, 
    valset=None,
    cmd_args=None,
    num_epoch=10,
    batch_size=64,
    eval_batch_size=256,
    lr=1e-4,
    weight_decay=0,
    patience=5,
    warmup_ratio=None,
    warmup_steps=None,
    eval_metric='auc',
    output_dir='./ckpt',
    collate_fn=None,
    num_workers=0,
    balance_sample=False,
    load_best_at_last=True,
    ignore_duplicate_cols=True,
    eval_less_is_better=False,
    flag=0,
    regression_task=False,
    train_method='normal',
    device=None,
    data_weight=None,
    **kwargs,
    ):
    """
    Train the model with provided training and validation sets.
    """
    if isinstance(trainset, tuple): trainset = [trainset]

    train_args = {
        'num_epoch': num_epoch,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'patience': patience,
        'warmup_ratio': warmup_ratio,
        'warmup_steps': warmup_steps,
        'eval_metric': eval_metric,
        'output_dir': output_dir,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'balance_sample': balance_sample,
        'load_best_at_last': load_best_at_last,
        'ignore_duplicate_cols': ignore_duplicate_cols,
        'eval_less_is_better': eval_less_is_better,
        'flag': flag,
        'regression_task': regression_task,
        'device': device,
        'data_weight': data_weight,
    }

    trainer = Trainer(
        model,
        trainset,
        valset,
        **train_args,
    )
    return trainer