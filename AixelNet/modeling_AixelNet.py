import logging
import os, pdb
import math
import collections
import json
from typing import Dict, Optional, Any, Union, Callable, List
from pymfe.mfe import MFE
from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.distributed as dist
from . import constants
from .regularization import SegmentBilinearAttentionRegularizer, compute_balance_regularization, compute_prediction_diversity
dev = 'cuda'

def freeze(layer):
    """
    Freeze the parameters of the layer to prevent them from being updated during backpropagation.
    """
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class AixelNetWordEmbedding(nn.Module):
    """
    A neural network module that initializes word embeddings for both header and value embeddings using pre-trained weights. 
    """
    def __init__(self,
        vocab_size,
        hidden_dim,
        vocab_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        vocab_freeze=False,
        use_bert=True,
        ) -> None:
        """
        Initializes the word embedding layer for both header and value embeddings using pre-trained weights.
        """
        super().__init__()
        word2vec_weight = torch.load('./AixelNet/tokenizer/bert_emb.pt')
        self.word_embeddings_header = nn.Embedding.from_pretrained(word2vec_weight, freeze=vocab_freeze, padding_idx=padding_idx)
        self.word_embeddings_value = nn.Embedding(vocab_size, vocab_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings_value.weight)

        self.norm_header = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)
        weight_emb = torch.load('./AixelNet/tokenizer/bert_layernorm_weight.pt')
        bias_emb = torch.load('./AixelNet/tokenizer/bert_layernorm_bias.pt')
        self.norm_header.weight.data.copy_(weight_emb)
        self.norm_header.bias.data.copy_(bias_emb)
        if vocab_freeze:
            freeze(self.norm_header)
        self.norm_value = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, emb_type) -> Tensor:
        if emb_type == 'header':
            embeddings = self.word_embeddings_header(input_ids)
            embeddings = self.norm_header(embeddings)
        elif emb_type == 'value':
            embeddings = self.word_embeddings_value(input_ids)
            embeddings = self.norm_value(embeddings)
        else:
            raise RuntimeError(f'no {emb_type} word_embedding method!')

        embeddings =  self.dropout(embeddings)
        return embeddings

class AixelNetNumEmbedding(nn.Module):
    """
    A module that generates numerical feature embeddings with normalization and bias, 
    used for embedding numerical columns in the dataset.
    """
    def __init__(self, hidden_dim) -> None:
        """
        Initializes the numerical embedding layer.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb

class AixelNetFeatureExtractor:
    """
    Extracts features from input data (both numerical and categorical) and tokenizes categorical features using a tokenizer. 
    It also provides methods for saving and loading tokenizers.
    """
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        **kwargs,
        ) -> None:
        """
        Initializes the feature extractor, including tokenizer setup.
        """
        if os.path.exists('./AixelNet/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./AixelNet/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./AixelNet/tokenizer')
        self.tokenizer.__dict__['model_max_length'] = 512
        if disable_tokenizer_parallel: # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols


    def __call__(self, x, x_cat=None, table_flag=0) -> Dict:
        encoded_inputs = {
            'x_num':None,
            'num_col_input_ids':None,
            'x_cat_input_ids':None,
            "num_att_mask": None,
            "x_cat_att_mask": None,
            "col_cat_att_mask": None,
            "col_cat_input_ids": None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns[table_flag]] if self.categorical_columns[table_flag] is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns[table_flag]] if self.numerical_columns[table_flag] is not None else []
        
        if len(cat_cols+num_cols) == 0:
            cat_cols = col_names

        # mask out NaN values like done in binary columns
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0) # fill Nan with zero
            x_num_ts = torch.tensor(x_num.values, dtype=torch.float32)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask'] # mask out attention

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_cat = x_cat.fillna('')
            x_cat_str = x_cat.values.tolist()
            encoded_inputs['x_cat_input_ids'] = []
            encoded_inputs['x_cat_att_mask'] = []
            max_y = 0
            cat_cnt = len(cat_cols)
            max_token_len = max(int(1), int(2048/cat_cnt))
            for sample in x_cat_str:
                x_cat_ts = self.tokenizer(sample, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
                x_cat_ts['input_ids'] = x_cat_ts['input_ids'][:,:max_token_len]
                x_cat_ts['attention_mask'] = x_cat_ts['attention_mask'][:,:max_token_len]
                encoded_inputs['x_cat_input_ids'].append(x_cat_ts['input_ids'])
                encoded_inputs['x_cat_att_mask'].append(x_cat_ts['attention_mask'])
                max_y = max(max_y, x_cat_ts['input_ids'].shape[1])
            for i in range(len(encoded_inputs['x_cat_input_ids'])):
                # tmp = torch.zeros((cat_cnt, max_y), dtype=int)
                tmp = torch.full((cat_cnt, max_y), self.pad_token_id, dtype=int)
                tmp[:, :encoded_inputs['x_cat_input_ids'][i].shape[1]] = encoded_inputs['x_cat_input_ids'][i]
                encoded_inputs['x_cat_input_ids'][i] = tmp
                tmp = torch.zeros((cat_cnt, max_y), dtype=int)
                tmp[:, :encoded_inputs['x_cat_att_mask'][i].shape[1]] = encoded_inputs['x_cat_att_mask'][i]
                encoded_inputs['x_cat_att_mask'][i] = tmp
            encoded_inputs['x_cat_input_ids'] = torch.stack(encoded_inputs['x_cat_input_ids'], dim=0)
            encoded_inputs['x_cat_att_mask'] = torch.stack(encoded_inputs['x_cat_att_mask'], dim=0)

            col_cat_ts = self.tokenizer(cat_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['col_cat_input_ids'] = col_cat_ts['input_ids']
            encoded_inputs['col_cat_att_mask'] = col_cat_ts['attention_mask']
        
        return encoded_inputs

    def save(self, path):
        '''
        save the feature extractor configuration to local dir.
        '''
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

    def load(self, path):
        '''load the feature extractor configuration from local dir.
        '''
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)


    def update(self, cat=None, num=None, bin=None):
        if cat is not None:
            self.categorical_columns = cat

        if num is not None:
            self.numerical_columns = num

        if bin is not None:
            self.binary_columns = bin

class AixelNetFeatureProcessor(nn.Module):
    """
    Processes features for embedding by handling both numerical and categorical embeddings, 
    including different pooling strategies (e.g., average, max, self-attention).
    """
    def __init__(self,
        vocab_size=None,
        vocab_dim=768,
        hidden_dim=128,
        hidden_dropout_prob=0,
        pad_token_id=0,
        vocab_freeze=False,
        use_bert=True,
        pool_policy='avg',
        device=dev,
        ) -> None:
        """
        Initializes the feature processor for handling word embeddings and numerical embeddings.
        """
        super().__init__()
        self.word_embedding = AixelNetWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            vocab_dim=vocab_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
        )
        self.num_embedding = AixelNetNumEmbedding(vocab_dim)


        self.align_layer = nn.Linear(vocab_dim, hidden_dim, bias=False)
        
        self.pool_policy=pool_policy
        self.device = device

    def _avg_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is None:
            return embs.mean(-2)
        else:
            embs[att_mask==0] = 0
            embs = embs.sum(-2) / (att_mask.sum(-1,keepdim=True).to(embs.device)+eps)
            return embs
        
    def _max_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is not None:
            embs[att_mask==0] = -1e12
        embs = torch.max(embs, dim=-2)[0]
        return embs
    
    def _sa_block(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)[0]
        return x[:, 0, :]
    
    def _check_nan(self, value):
        return torch.isnan(value).any().item()

    def forward(self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        x_cat_att_mask=None,
        col_cat_input_ids=None,
        col_cat_att_mask=None,
        **kwargs,
        ) -> Tensor:
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None
        other_info = {
            'col_emb' : None,          # [num_fs+cat_fs]
            'num_cnt' : 0,             # num_fs
            'x_num' : x_num,           # [bs, num_fs]
            'cat_bert_emb' : None     # [bs, cat_fs, dim]
        }

        if other_info['x_num'] is not None:
            other_info['x_num'] = other_info['x_num'].to(self.device)

        if self.pool_policy=='avg':
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header') # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
    
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                x_cat_feat_embedding = self._avg_embedding_by_mask(x_cat_feat_embedding, x_cat_att_mask)
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                cat_col_emb = self._avg_embedding_by_mask(col_cat_feat_embedding, col_cat_att_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0],-1,-1))
                
                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._avg_embedding_by_mask(cat_feat_embedding)
    
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
    
                cat_feat_embedding = self.align_layer(cat_feat_embedding)
                cat_col_emb = self.align_layer(cat_col_emb)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
    
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
        elif self.pool_policy=='no':
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header') # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
    
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                col_cat_feat_embedding = col_cat_feat_embedding.unsqueeze(0).expand((x_cat_feat_embedding.shape[0],-1,-1,-1))
                cat_feat_embedding = torch.cat((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                bs,emb_dim = cat_feat_embedding.shape[0], cat_feat_embedding.shape[-1]
                cat_feat_embedding = cat_feat_embedding.reshape(bs, -1, emb_dim)
                cat_feat_embedding = self.align_layer(cat_feat_embedding)

                # mask
                col_cat_att_mask = col_cat_att_mask.unsqueeze(0).expand((x_cat_att_mask.shape[0],-1,-1))
                cat_att_mask = torch.cat((col_cat_att_mask, x_cat_att_mask), dim=-1)
                cat_att_mask = cat_att_mask.reshape(bs, -1)
                
                cat_col_emb = None
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
        elif self.pool_policy=='max':        
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header') # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._max_embedding_by_mask(num_col_emb, num_att_mask)
    
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                x_cat_feat_embedding = self._max_embedding_by_mask(x_cat_feat_embedding, x_cat_att_mask)
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                cat_col_emb = self._max_embedding_by_mask(col_cat_feat_embedding, col_cat_att_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0],-1,-1))
                
                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._max_embedding_by_mask(cat_feat_embedding)
    
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._max_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
    
                cat_feat_embedding = self.align_layer(cat_feat_embedding)
                cat_col_emb = self.align_layer(cat_col_emb)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
    
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
        elif self.pool_policy=='self-attention':        
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header') # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_emb_mask = self.add_cls(num_col_emb, num_att_mask)
                num_col_emb = num_emb_mask['embedding']
                num_att_mask = num_emb_mask['attention_mask'].to(num_col_emb.device)
                num_col_emb = self._sa_block(num_col_emb, num_att_mask)
    
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                col_cat_feat_embedding = col_cat_feat_embedding.unsqueeze(0).expand((x_cat_feat_embedding.shape[0],-1,-1,-1))
                cat_feat_embedding = torch.cat((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                # mask
                col_cat_att_mask = col_cat_att_mask.unsqueeze(0).expand((x_cat_att_mask.shape[0],-1,-1))
                cat_att_mask = torch.cat((col_cat_att_mask, x_cat_att_mask), dim=-1)

                bs, fs, ls = cat_feat_embedding.shape[0], cat_feat_embedding.shape[1], cat_feat_embedding.shape[2]
                cat_feat_embedding = cat_feat_embedding.reshape(bs*fs, ls, -1)
                cat_att_mask = cat_att_mask.reshape(bs*fs, ls)
                cat_embedding_mask = self.add_cls(cat_feat_embedding, cat_att_mask)
                cat_feat_embedding = cat_embedding_mask['embedding'] 
                cat_att_mask = cat_embedding_mask['attention_mask'].to(cat_feat_embedding.device)
                cat_feat_embedding = self._sa_block(cat_feat_embedding, cat_att_mask).reshape(bs, fs, -1)
                cat_feat_embedding = self.align_layer(cat_feat_embedding)

                cat_col_emb = None
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
        else:
            raise RuntimeError(f'no such {self.pool_policy} pooling policy!!!')


        emb_list = []
        att_mask_list = []
        col_emb = []
        if num_feat_embedding is not None:
            col_emb += [num_col_emb]
            other_info['num_cnt'] = num_col_emb.shape[0]
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1]).to(self.device)]

            # emb_list += [num_feat_embedding]
            # att_mask_list += [num_att_mask]
        if cat_feat_embedding is not None:
            col_emb += [cat_col_emb]
            emb_list += [cat_feat_embedding]
            if self.pool_policy=='no':
                att_mask_list += [cat_att_mask.to(self.device)]
            else:
                att_mask_list += [torch.ones(cat_feat_embedding.shape[0], cat_feat_embedding.shape[1]).to(self.device)]
            
        if len(emb_list) == 0: raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        other_info['col_emb'] = torch.cat(col_emb, 0).float()
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}, other_info

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))

class AixelNetTransformerLayer(nn.Module):
    """
    A Transformer layer used in the encoder, implementing multi-head self-attention and a feed-forward network with 
    layer normalization and dropout.
    """
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_layer_norm=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        src = x
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None) -> Tensor:
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else: # do not use layer norm
                x = x + self._sa_block(x, src_mask, src_key_padding_mask)
                x = x + self._ff_block(x)
        return x

class AixelNetInputEncoder(nn.Module):
    """
    Encodes input data using a feature extractor and processor, generating embeddings for the dataset using the specified 
    feature extraction and processing methods.
    """
    def __init__(self,
        feature_extractor,
        feature_processor,
        device=dev,
        ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x):
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds
    
    def load(self, ckpt_dir):
        # load feature extractor
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

        # load embedding layer
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'load model from {ckpt_dir}')

class AixelNetEncoder(nn.Module):
    def __init__(self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
            AixelNetTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            ]
            )
        if num_layer > 1:
            encoder_layer = AixelNetTransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer-1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        return outputs

class AixelNetLinearClassifier(nn.Module):
    def __init__(self, num_class, hidden_dim=128, device='cuda'):
        super().__init__()
        self.num_class = num_class
        self.device = device
        self.fc = nn.Linear(hidden_dim, 1 if num_class <= 2 else num_class).to(device)
        self.norm = nn.LayerNorm(hidden_dim).to(device)

    def forward(self, x):
        x = x[:, 0, :].to(self.device) 
        x = self.norm(x)
        logits = self.fc(x)
        return logits

class AixelNetLinearRegression(nn.Module):
    def __init__(self,
        hidden_dim=128) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activate_fn = nn.ReLU()

    def forward(self, x) -> Tensor:
        x = x[:,0,:]
        x = self.activate_fn(x)
        logits = self.fc(x)
        return logits

class AixelNetProjectionHead(nn.Module):
    def __init__(self,
        hidden_dim=128,
        projection_dim=128):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        h = self.dense(x)
        return h

class AixelNetCLSToken(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask
        return outputs
    
class AixelNetMaskToken(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mask_emb = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.mask_emb, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, embedding, masked_indices, header_emb):
        embedding[masked_indices.bool()] = 0
        bs,fs = embedding.shape[0], embedding.shape[1]
        all_mask_token = self.mask_emb.unsqueeze(0).unsqueeze(0).expand(bs,fs,-1) + header_emb.unsqueeze(0).expand(bs,-1,-1) 
        embedding = embedding + all_mask_token*masked_indices.unsqueeze(-1)

        return embedding

class MetaFeatureExtractor:
    def __init__(self, groups=["statistical", "general", "info-theory"]):
        """
        Initializes the MetaFeatureExtractor for extracting static meta features from the dataset.
        
        Parameters:
            groups (list): The groups of meta features to be extracted.
        """
        self.groups = groups
        self.meta_feature_dim = None  # Dynamically determined meta feature dimension

    def extract_meta_features(self, df):
        """
        Extracts the meta feature vector from the dataset.
        
        Parameters:
            df (DataFrame): The dataset containing features and labels.
        
        Returns:
            Tensor: The extracted meta feature vector.
        """
        df = df.dropna()
        target = df.columns[-1]
        y = np.array(df[target])
        X = np.array(df.drop(columns=[target]))

        mfe = MFE(groups=self.groups)
        mfe.fit(X, y)

        features, values = mfe.extract()

        # Handle NaN values in extracted meta features
        values = np.array(values, dtype=np.float32)
        if np.isnan(values).any():
            values = np.nan_to_num(values, nan=0.0)

        if self.meta_feature_dim is None:
            self.meta_feature_dim = len(values)  # Dynamically determine the dimension

        meta_features = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        return meta_features


class HypernetWeightGenerator(nn.Module):
    def __init__(self, meta_feature_dim, k_models):
        """
        Initializes the hypernetwork for generating weights using a simple MLP.
        
        Parameters:
            meta_feature_dim (int): The dimension of the meta features, i.e., the dataset feature vector dimension.
            k_models (int): The number of generated weights, corresponding to the number of base models in AixelNetEModel.
        """
        super(HypernetWeightGenerator, self).__init__()
        self.k_models = k_models
        
        # MLP to generate K weights
        self.mlp = nn.Sequential(
            nn.Linear(meta_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k_models),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, meta_features):
        """
        Generates normalized weights from the meta features.
        
        Parameters:
            meta_features (Tensor): A tensor of shape [batch_size, meta_feature_dim] containing the dataset feature vectors.
        
        Returns:
            weights (Tensor): A tensor of shape [batch_size, k_models] representing the generated weights.
        """
        weights = self.mlp(meta_features)
        return weights


class AixelNetModel(nn.Module):
    """
    The main model class that integrates various components (feature extraction, embedding layers, and multiple encoders).
    It handles forward passes, loss calculations, and regularization.
    """
    def __init__(self,
                 num_classes_list,
                 k_models,
                 categorical_columns=None,
                 numerical_columns=None,
                 binary_columns=None,
                 feature_extractor=None,
                 hidden_dim=128,
                 num_layer=3,
                 num_attention_head=8,
                 hidden_dropout_prob=0.1,
                 ffn_dim=256,
                 activation='relu',
                 vocab_freeze=False,
                 use_bert=True,
                 pool_policy='avg',
                 device='cuda',
                 meta_feature_groups=["statistical", "general", "info-theory"],
                 wo_arg=None,
                 lambda1 = 1e-4,
                 lambda2 = 1e-4,
                 hyper_arg=None,
                 **kwargs):
        super().__init__()
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.num_classes_list = num_classes_list
        self.k_models = k_models
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device

        # Ablation settings: Enable/Disable specific modules
        self.wo_meta = wo_arg == "wo_meta"
        self.wo_hypernetwork = wo_arg == "wo_hypernetwork"
        self.wo_sparse = wo_arg == "wo_sparse"
        self.wo_regularization = wo_arg == "wo_regularization"

        self.uniform = hyper_arg == "Uniform"
        self.heuristic = hyper_arg == "Heuristic"

        # Initialize feature extractor if not provided
        if feature_extractor is None:
            feature_extractor = AixelNetFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        # Initialize feature processor
        feature_processor = AixelNetFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
        )

        # Initialize input encoder with feature extractor and processor
        self.input_encoder = AixelNetInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        # Initialize multiple base encoders
        self.encoders = nn.ModuleList([
            AixelNetEncoder(
                hidden_dim=hidden_dim,
                num_layer=num_layer,
                num_attention_head=num_attention_head,
                hidden_dropout_prob=hidden_dropout_prob,
                ffn_dim=ffn_dim,
                activation=activation,
            )
            for _ in range(k_models)
        ])

        # Initialize CLS token
        self.cls_token = AixelNetCLSToken(hidden_dim=hidden_dim)

        # Initialize meta feature extractor and weight generator
        self.meta_feature_extractor = MetaFeatureExtractor(groups=meta_feature_groups)
        self.hypernet_weight_generator = None
        self.meta_feature_cache = {}
        self._init_hypernet_if_possible()

        self.to(device)

    def _init_hypernet_if_possible(self):
        if self.wo_hypernetwork:
            logger.info("Skipping Hypernetwork. Using uniform weights across all predictors.")
            self.hypernet_weight_generator = None
            return

        meta_dim = getattr(self.meta_feature_extractor, "meta_feature_dim", None)
        if meta_dim is None:
            # meta_feature_dim may be determined later; delay init
            return

        if self.hypernet_weight_generator is None:
            self.hypernet_weight_generator = HypernetWeightGenerator(
                meta_feature_dim=meta_dim,
                k_models=self.k_models
            ).to(self.device)

    def forward(self, x, y=None, df=None, table_flag=0):
        """
        Perform a forward pass through the model, process input, and generate predictions.
        
        Parameters:
            x (dict or pd.DataFrame): Input data for the model.
            y (Tensor or None): Labels for loss computation, optional.
            df (pd.DataFrame or None): Dataframe used for meta-feature extraction, optional.
            table_flag (int): Identifier for the specific table.
        
        Returns:
            encoder_outputs (list): List of encoder outputs.
            weights (Tensor): Weights generated based on meta features.
        """
        # Check input type and prepare input
        if isinstance(x, dict):
            inputs = x
        elif isinstance(x, pd.DataFrame):
            inputs = self.input_encoder.feature_extractor(x, table_flag=table_flag)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}. Must be dict or pd.DataFrame.")

        # Process inputs using feature processor
        encoder_input, _ = self.input_encoder.feature_processor(**inputs)
        encoder_input = self.cls_token(**encoder_input)

        if self.wo_meta:
            logger.info("Skipping Meta Feature Extraction.")
            weights = torch.ones((1, self.k_models), device=self.device) / self.k_models
        else:
            # Meta feature extraction
            if df is None:
                logger.warning("No DataFrame provided for meta feature extraction. Using uniform weights.")
                weights = torch.ones((1, self.k_models), device=self.device) / self.k_models
            else:
                if table_flag in self.meta_feature_cache:
                    meta_features = self.meta_feature_cache[table_flag]['meta_features']
                else:
                    meta_features = self.meta_feature_extractor.extract_meta_features(df).to(self.device)
                    meta_features = meta_features.detach()
                    self.meta_feature_cache[table_flag] = {'meta_features': meta_features}

                self._init_hypernet_if_possible()

                if self.hypernet_weight_generator is None:
                    # fallback to uniform if hypernet is disabled or not available
                    weights = torch.ones((1, self.k_models), device=self.device) / self.k_models
                else:
                    weights = self.hypernet_weight_generator(meta_features)

        # Obtain encoder outputs from all models
        encoder_outputs = [encoder(**encoder_input) for encoder in self.encoders]

        return encoder_outputs, weights

    def save(self, ckpt_dir):
        """
        Saves the model parameters and configurations to a checkpoint directory.
        
        Parameters:
            ckpt_dir (str): Path to the directory where the model will be saved.
        """
        # Ensure the checkpoint directory exists
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        # Save the state dict of each encoder
        for i, encoder in enumerate(self.encoders):
            torch.save(encoder.state_dict(), os.path.join(ckpt_dir, f'encoder_{i}.pth'))

        # Save the input encoder and its feature extractor
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        self.input_encoder.feature_extractor.save(ckpt_dir)

        # Save meta feature dimension to a file
        meta_feature_dim_path = os.path.join(ckpt_dir, "meta_feature_dim.json")
        if self.meta_feature_extractor.meta_feature_dim is not None:
            with open(meta_feature_dim_path, 'w') as f:
                json.dump({'meta_feature_dim': self.meta_feature_extractor.meta_feature_dim}, f)

        # Save the hypernet weight generator if initialized
        if self.hypernet_weight_generator is not None:
            torch.save(self.hypernet_weight_generator.state_dict(), os.path.join(ckpt_dir, "hypernet_weight_generator.pth"))

        logger.info(f'Model encoders and input encoder saved to {ckpt_dir}.')

    def load(self, ckpt_dir):
        """
        Loads the model parameters and configurations from a checkpoint directory.
        
        Parameters:
            ckpt_dir (str): Path to the directory from which the model will be loaded.
        """
        # Load the main model weights
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')

        # Remove task classifier-related keys if present
        keys_to_remove = [key for key in state_dict.keys() if "task_classifiers" in key]
        for key in keys_to_remove:
            del state_dict[key]

        # Load the model state and handle missing/unexpected keys
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        # Load the encoders from their respective checkpoint files
        for i, encoder in enumerate(self.encoders):
            encoder_path = os.path.join(ckpt_dir, f'encoder_{i}.pth')
            if os.path.exists(encoder_path):
                encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))

        # Load the input encoder
        input_encoder_path = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        if os.path.exists(input_encoder_path):
            state_dict_input_encoder = torch.load(input_encoder_path, map_location=self.device)
            self.input_encoder.load_state_dict(state_dict_input_encoder)
            logger.info("Input encoder loaded from input_encoder.bin.")
        else:
            logger.warning("Input encoder file not found!")

        # Load the meta feature dimension if available
        meta_feature_dim_path = os.path.join(ckpt_dir, "meta_feature_dim.json")
        meta_feature_dim = None
        if os.path.exists(meta_feature_dim_path):
            with open(meta_feature_dim_path, 'r') as f:
                data = json.load(f)
                meta_feature_dim = data.get('meta_feature_dim', None)

        if meta_feature_dim is not None:
            self.meta_feature_extractor.meta_feature_dim = meta_feature_dim

        self._init_hypernet_if_possible()

        # Load the hypernet weight generator if available
        hypernet_path = os.path.join(ckpt_dir, "hypernet_weight_generator.pth")
        if os.path.exists(hypernet_path):
            if self.hypernet_weight_generator is not None:
                self.hypernet_weight_generator.load_state_dict(torch.load(hypernet_path, map_location=self.device))
                logger.info("Hypernet weight generator loaded.")
            else:
                logger.warning("hypernet_weight_generator.pth found but hypernet_weight_generator still not initialized (meta_feature_dim may be None or hypernet disabled).")
        else:
            logger.warning("No hypernet weight generator file found.")

        extractor_dir = os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR)
        if os.path.exists(extractor_dir):
            self.input_encoder.feature_extractor.load(extractor_dir)
        else:
            logger.warning("Feature extractor directory not found!")

    def update(self, config):
        """
        Updates the model configuration, including column mappings and task classifiers.
        
        Parameters:
            config (dict): The new configuration parameters for the model.
        """
        col_map = {}
        for k, v in config.items():
            if k in ['cat', 'num', 'bin']:
                col_map[k] = v

        # Update the feature extractor with new column configurations
        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        # Update the task classifiers if 'num_class' is provided
        if 'num_class' in config:
            num_class = config['num_class']
        return None

class AixelNetPretrain(AixelNetModel):
    """
    A specialized version of the AixelNetModel designed for pre-training with multiple tables.
    It includes meta-feature extraction and regularization terms for fine-tuning during pre-training.
    """
    def __init__(self,
                 num_classes_list, 
                 k_models, 
                 meta_feature_dim=None,
                 hidden_dim=128,
                 num_layer=2,
                 num_attention_head=8,
                 hidden_dropout_prob=0.1,
                 ffn_dim=256,
                 activation='relu',
                 vocab_freeze=False,
                 use_bert=True,
                 pool_policy='avg',
                 device='cuda',
                 dataset_paths=None,
                 wo_arg=None,
                 lambda1=1e-5,
                 lambda2=1e-5,
                 hyper_arg=None,
                 **kwargs):
        super().__init__(
            num_classes_list=num_classes_list,
            k_models=k_models,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
            wo_arg=wo_arg,
            lambda1=lambda1,
            lambda2=lambda2,
            hyper_arg=hyper_arg,
            **kwargs
        )
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_classes_list = num_classes_list
        self.k_models = k_models
        self.task_classifiers = nn.ModuleList([
            nn.ModuleList([
                AixelNetLinearClassifier(num_class=num_classes, hidden_dim=hidden_dim)
                for _ in range(k_models)
            ])
            for num_classes in num_classes_list
        ])
        self.multiclass_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.binaryclass_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

    def forward(self, x, y=None, df=None, table_flag=0,
                sparse=True, M=2, lambda1=None, lambda2=None, return_regularization=False):
        lambda1 = lambda1 if lambda1 is not None else self.lambda1
        lambda2 = lambda2 if lambda2 is not None else self.lambda2
        # Check input type and prepare input
        if isinstance(x, dict):
            inputs = x
        elif isinstance(x, pd.DataFrame):
            inputs = self.input_encoder.feature_extractor(x, table_flag=table_flag)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}. Must be dict or pd.DataFrame.")

        # Process inputs using feature processor
        encoder_input, _ = self.input_encoder.feature_processor(**inputs)
        encoder_input = self.cls_token(**encoder_input)
        
        if not hasattr(self, 'meta_split_indices') or self.meta_split_indices is None:
            encoder_outputs, weights = super().forward(x, y, df=df, table_flag=table_flag)

            # Initialize meta feature extraction only if not ablated
            if not self.wo_meta:
                meta_dim = self.meta_feature_extractor.meta_feature_dim
                if meta_dim is None:
                    raise ValueError("Meta feature dimension has not been computed. Please provide a DataFrame to extract features.")
                self.meta_split_indices = [0, meta_dim//3, 2*meta_dim//3, meta_dim]
            self.sim_regularizer = SegmentBilinearAttentionRegularizer(
                split_indices=self.meta_split_indices,
                k_models=self.k_models
            ).to(self.device)
        else:
            encoder_outputs, weights = super().forward(x, y, df=df, table_flag=table_flag)

        batch_size = encoder_outputs[0].size(0)
        K = self.k_models

        # If sparse selection is enabled, sample weights
        if not self.wo_sparse:
            probs = weights[0].detach().cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.0)
            non_zero_count = np.count_nonzero(probs)
            probs_sum = probs.sum()
            if non_zero_count < M or probs_sum <= 1e-9:
                epsilon = 1e-8
                probs = probs + epsilon
                probs = probs / probs.sum()
            else:
                probs = probs / probs_sum
            sampled_indices = np.random.choice(K, M, replace=False, p=probs)
            sampled_indices = torch.tensor(sampled_indices, device=self.device)
            selected_weights = weights[0, sampled_indices]
            sw_sum = selected_weights.sum()
            if sw_sum > 1e-9:
                normalized_weights = selected_weights / sw_sum
            else:
                normalized_weights = torch.ones_like(selected_weights) / M
            # normalized_weights = selected_weights / selected_weights.sum()
        else:
            sampled_indices = torch.arange(K, device=self.device)
            normalized_weights = weights[0]
        # Placeholder for predictions
        combined_logits = None
        pred_list = []
        raw_preds = []

        # Loop over sampled indices to compute predictions
        for i, idx in enumerate(sampled_indices):
            encoder_output = self.encoders[idx](**encoder_input)
            
#           encoder_output = encoder_outputs[idx]
            classifier = self.task_classifiers[table_flag][idx]
            logits = classifier(encoder_output)
            pred_list.append(logits.detach())  # For R_pred
            raw_preds.append(logits)  # For heuristic weight computation
            
        if self.heuristic:
            error_list = []
            for logits in raw_preds:
                if self.num_classes_list[table_flag] <= 2:
                    logits = logits[:, 0]
                    loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='none')
                else:
                    loss = F.cross_entropy(logits, y.long(), reduction='none')
                error_list.append(loss.mean().item())

            error_tensor = torch.tensor(error_list, device=self.device)
            normalized_weights = torch.softmax(-error_tensor, dim=0)  # Negative error → higher weight for lower error
        else:
            if self.wo_hypernetwork or self.uniform:
                logger.info("Skipping Hypernetwork. Using uniform weights across all predictors.")
                normalized_weights = torch.ones(len(sampled_indices), device=self.device) / len(sampled_indices)
            elif not self.wo_sparse:
                selected_weights = weights[0, sampled_indices]
                normalized_weights = selected_weights / selected_weights.sum()
            else:
                normalized_weights = weights[0]

        combined_logits = None
        pred_list = []

        # Loop over sampled indices to compute predictions
        for i, idx in enumerate(sampled_indices):
            encoder_output = encoder_outputs[idx]
            classifier = self.task_classifiers[table_flag][idx]
            logits = classifier(encoder_output)
            pred_list.append(logits.detach())  # Used for R_pred
            if combined_logits is None:
                combined_logits = normalized_weights[i] * logits
            else:
                combined_logits += normalized_weights[i] * logits

        # Loss calculation
        if y is not None:
            if isinstance(y, pd.Series):
                y = torch.tensor(y.values).to(combined_logits.device)
            else:
                y = y.to(combined_logits.device)

            if self.num_classes_list[table_flag] <= 2:
                combined_logits = combined_logits[:, 0]
                base_loss = self.binaryclass_loss_fn(combined_logits, y.float())
            else:
                base_loss = self.multiclass_loss_fn(combined_logits, y.long())
            base_loss = base_loss.mean()

            meta_features = self.meta_feature_cache[table_flag]['meta_features'].expand(batch_size, -1)
            weight_batch = weights.expand(batch_size, -1)
            
            # print(f"[Predictor GPU] allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | "
            #       f"peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            
            # Regularization terms
            R_sim = R_bal = R_pred = 0
            if not self.wo_regularization:
                R_sim = self.sim_regularizer(meta_features, weights)
                R_bal = compute_balance_regularization(weight_batch)
                R_pred = compute_prediction_diversity(pred_list)
                total_loss = base_loss + lambda1 * R_pred + lambda2 * (R_sim + R_bal)
            else:
                total_loss = base_loss
            if return_regularization:
                return combined_logits, base_loss, R_pred, R_sim, R_bal
            return combined_logits, total_loss
        return combined_logits, None

class AixelNetForClassifier(AixelNetModel):
    def __init__(self,
                 num_class,
                 k_models,
                 hidden_dim=128,
                 num_layer=2,
                 num_attention_head=8,
                 hidden_dropout_prob=0.1,
                 ffn_dim=256,
                 activation='relu',
                 vocab_freeze=False,
                 use_bert=True,
                 pool_policy='avg',
                 device='cuda',
                 **kwargs):
        super().__init__(
            num_classes_list=[num_class],
            k_models=k_models,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
            **kwargs
        )

        self.num_class = num_class
        self.k_models = k_models
        self.task_classifiers = nn.ModuleList([
            nn.ModuleList([
                AixelNetLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
                for _ in range(self.k_models)
            ])
        ])
        self.multiclass_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.binaryclass_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

    def forward(self, x, y=None, df=None, table_flag=0,
                sparse=False, M=2, lambda1=1e-5, lambda2=1e-5, return_regularization=False):

        if not hasattr(self, 'meta_split_indices') or self.meta_split_indices is None:
            encoder_outputs, weights = super().forward(x, y, df=df, table_flag=table_flag)
            meta_dim = self.meta_feature_extractor.meta_feature_dim
            if meta_dim is None:
                raise ValueError("Meta feature dimension has not been computed. Please provide a DataFrame to extract features.")
            self.meta_split_indices = [0, meta_dim // 3, 2 * meta_dim // 3, meta_dim]
            self.sim_regularizer = SegmentBilinearAttentionRegularizer(
                split_indices=self.meta_split_indices,
                k_models=self.k_models
            ).to(self.device)
        else:
            encoder_outputs, weights = super().forward(x, y, df=df, table_flag=table_flag)

        batch_size = encoder_outputs[0].size(0)
        K = self.k_models

        if sparse:
            probs = weights[0].detach().cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.0)
            non_zero_count = np.count_nonzero(probs)
            probs_sum = probs.sum()
            if non_zero_count < M or probs_sum <= 1e-9:
                epsilon = 1e-8
                probs = probs + epsilon
                probs = probs / probs.sum()
            else:
                probs = probs / probs_sum
            sampled_indices = np.random.choice(K, M, replace=False, p=probs)
            sampled_indices = torch.tensor(sampled_indices, device=self.device)
            selected_weights = weights[0, sampled_indices]
            sw_sum = selected_weights.sum()
            if sw_sum > 1e-9:
                normalized_weights = selected_weights / sw_sum
            else:
                normalized_weights = torch.ones_like(selected_weights) / M
            # normalized_weights = selected_weights / selected_weights.sum()
        else:
            sampled_indices = torch.arange(K, device=self.device)
            normalized_weights = weights[0]

        combined_logits = None
        pred_list = []
        for i, idx in enumerate(sampled_indices):
            encoder_output = encoder_outputs[idx]
            classifier = self.task_classifiers[0][idx] 
            logits = classifier(encoder_output)
            pred_list.append(logits.detach())
            if combined_logits is None:
                combined_logits = normalized_weights[i] * logits
            else:
                combined_logits += normalized_weights[i] * logits

        if y is not None:
            if isinstance(y, pd.Series):
                y = torch.tensor(y.values).to(combined_logits.device)
            else:
                y = y.to(combined_logits.device)

            if self.num_class <= 2:
                combined_logits = combined_logits[:, 0]
                base_loss = self.binaryclass_loss_fn(combined_logits, y.float())
            else:
                base_loss = self.multiclass_loss_fn(combined_logits, y.long())
            base_loss = base_loss.mean()

            meta_features = self.meta_feature_cache[table_flag]['meta_features'].expand(batch_size, -1)
            weight_batch = weights.expand(batch_size, -1)
            R_sim = self.sim_regularizer(meta_features, weights)
            R_bal = compute_balance_regularization(weight_batch)
            R_pred = compute_prediction_diversity(pred_list)

            if return_regularization:
                return combined_logits, base_loss, R_pred, R_sim, R_bal

            total_loss = base_loss + lambda1 * R_pred + lambda2 * (R_sim + R_bal)

            return combined_logits, total_loss

        return combined_logits, None
    
class AixelNetForRegression(AixelNetModel):
    """
    A regression model that predicts continuous values by combining predictions from multiple base models. It includes 
    sparse model selection and regularization terms.
    """
    def __init__(self,
                 k_models,
                 categorical_columns=None,
                 numerical_columns=None,
                 binary_columns=None,
                 feature_extractor=None,
                 hidden_dim=128,
                 num_layer=2,
                 num_attention_head=8,
                 hidden_dropout_prob=0.1,
                 ffn_dim=256,
                 activation='relu',
                 vocab_freeze=False,
                 use_bert=True,
                 pool_policy='avg',
                 device='cuda',
                 **kwargs):
        super().__init__(
            num_classes_list=[1],
            k_models=k_models,
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
            pool_policy=pool_policy,
            device=device,
            **kwargs
        )

        self.num_class = 1 
        self.k_models = k_models

        self.task_regressors = nn.ModuleList([
            nn.ModuleList([
                AixelNetLinearRegression(hidden_dim=hidden_dim) for _ in range(self.k_models)
            ])
        ])

        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        self.to(device)

    def forward(self, x, y=None, df=None, table_flag=0,
                sparse=True, M=2, lambda1=1e-5, lambda2=1e-5, return_regularization=False):

        if not hasattr(self, 'meta_split_indices') or self.meta_split_indices is None:
            encoder_outputs, weights = super().forward(x, y, df=df, table_flag=table_flag)
            meta_dim = self.meta_feature_extractor.meta_feature_dim
            if meta_dim is None:
                raise ValueError("Meta feature dimension has not been computed. Please provide a DataFrame to extract features.")
            self.meta_split_indices = [0, meta_dim // 3, 2 * meta_dim // 3, meta_dim]
            self.sim_regularizer = SegmentBilinearAttentionRegularizer(
                split_indices=self.meta_split_indices,
                k_models=self.k_models
            ).to(self.device)
        else:
            encoder_outputs, weights = super().forward(x, y, df=df, table_flag=table_flag)

        batch_size = encoder_outputs[0].size(0)
        K = self.k_models

        if sparse:
            probs = weights[0].detach().cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.0)
            non_zero_count = np.count_nonzero(probs)
            probs_sum = probs.sum()
            if non_zero_count < M or probs_sum <= 1e-9:
                epsilon = 1e-8
                probs = probs + epsilon
                probs = probs / probs.sum()
            else:
                probs = probs / probs_sum
            sampled_indices = np.random.choice(K, M, replace=False, p=probs)
            sampled_indices = torch.tensor(sampled_indices, device=self.device)
            selected_weights = weights[0, sampled_indices]
            sw_sum = selected_weights.sum()
            if sw_sum > 1e-9:
                normalized_weights = selected_weights / sw_sum
            else:
                normalized_weights = torch.ones_like(selected_weights) / M
            # normalized_weights = selected_weights / selected_weights.sum()
        else:
            sampled_indices = torch.arange(K, device=self.device)
            normalized_weights = weights[0]

        combined_preds = None
        pred_list = []
        for i, idx in enumerate(sampled_indices):
            encoder_output = encoder_outputs[idx]
            regressor = self.task_regressors[0][idx]  
            preds = regressor(encoder_output) 
            pred_list.append(preds.detach())
            if combined_preds is None:
                combined_preds = normalized_weights[i] * preds
            else:
                combined_preds += normalized_weights[i] * preds

        if y is not None:
            if isinstance(y, pd.Series):
                y = torch.tensor(y.values, dtype=torch.float32).to(combined_preds.device)
            else:
                y = y.float().to(combined_preds.device)
            y = y.reshape(-1, 1)

            base_loss = self.mse_loss_fn(combined_preds, y).mean()

            meta_features = self.meta_feature_cache[table_flag]['meta_features'].expand(batch_size, -1)
            weight_batch = weights.expand(batch_size, -1)
            R_sim = self.sim_regularizer(meta_features, weights)
            R_bal = compute_balance_regularization(weight_batch)
            R_pred = compute_prediction_diversity(pred_list)

            if return_regularization:
                return combined_preds, base_loss, R_pred, R_sim, R_bal

            total_loss = base_loss + lambda1 * R_pred + lambda2 * (R_sim + R_bal)
            return combined_preds, total_loss

        return combined_preds, None
