#!/usr/bin/env python
# encoding: utf-8


import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
# from fairseq.modules.dlcl import DynamicLinearCombination
# from fairseq.modules import (
#   LearnedPositionalEmbedding, MultiheadAttention, MultiheadAttentionV2,
#  SinusoidalPositionalEmbedding,
# )

# from . import (
#   FairseqIncrementalDecoder, FairseqEncoder, FairseqModel,
#  register_model, register_model_architecture,
# )

# import fairseq.utils as util


from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding,
)

from fairseq.models import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel,
    register_model, register_model_architecture,
)

from fairseq import utils as util

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

try:
    from xformers.components.attention import build_attention
    from xformers.components.attention.utils import maybe_merge_masks

    _xformers_available = True
except ImportError:
    _xformers_available = False

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class MultiheadAttentionV2(nn.Module):
    """support different dimension between encoder and decoder on Multi-headed attention.

    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, kv_dim=None, inner_dim=None):
        super().__init__()
        assert ((kv_dim is None) ^ (inner_dim is None)) == 0
        self.kv_dim = kv_dim if kv_dim else self.embed_dim
        self.inner_dim = inner_dim if inner_dim else self.embed_dim

        self.embed_dim = embed_dim
        self.is_same_dim = kv_dim == inner_dim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.inner_dim // num_heads
        assert self.head_dim * num_heads == self.inner_dim
        self.scaling = self.head_dim ** -0.5
        self._mask = None

        if self.is_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
            if bias:
                self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
            else:
                self.register_parameter('in_proj_bias', None)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.in_proj_q_weight = Parameter(torch.Tensor(self.inner_dim, embed_dim))
            self.in_proj_kv_weight = Parameter(torch.Tensor(2 * self.inner_dim, self.kv_dim))
            if bias:
                self.in_proj_q_bias = Parameter(torch.Tensor(self.inner_dim))
                self.in_proj_kv_bias = Parameter(torch.Tensor(2 * self.inner_dim))
            else:
                self.register_parameter('in_proj_q_bias', None)
                self.register_parameter('in_proj_kv_bias', None)
            self.out_proj = nn.Linear(inner_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.is_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.in_proj_bias is not None:
                nn.init.constant_(self.in_proj_bias, 0.)
                nn.init.constant_(self.out_proj.bias, 0.)
        else:
            nn.init.xavier_uniform_(self.in_proj_q_weight)
            nn.init.xavier_uniform_(self.in_proj_kv_weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.in_proj_q_bias is not None:
                nn.init.constant_(self.in_proj_q_bias, 0.)
                nn.init.constant_(self.in_proj_kv_bias, 0.)
                nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if saved_state is not None:
            if 'prev_key' in saved_state:
                k = torch.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = torch.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.inner_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        if self.is_same_dim:
            return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)
        else:
            return F.linear(key, self.in_proj_kv_weight, self.in_proj_kv_bias).chunk(2, dim=-1)

    def in_proj_q(self, query):
        if self.is_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            return F.linear(query, self.in_proj_q_weight, self.in_proj_q_bias)

    def in_proj_k(self, key):
        if self.is_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=self.embed_dim + self.kv_dim)
        else:
            return F.linear(key,
                            self.in_proj_kv_weight[:self.embed_dim, :],
                            self.in_proj_kv_bias[:self.embed_dim, :])

    def in_proj_v(self, value):
        if self.is_same_dim:
            return self._in_proj(value, start=self.embed_dim + self.kv_dim)
        else:
            return F.linear(value,
                            self.in_proj_kv_weight[self.embed_dim:, :],
                            self.in_proj_kv_bias[self.embed_dim:, :])

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(utils.fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DynamicLinearCombination(nn.Module):
    """Implementation of Dynamic Linear Combination of Layers (DLCL)

        for pre-norm, x_{l+1} = \sum_{k=0}^{l}{W_k^{l+1}LN(y_k)}
        for post-norm, x_{l+1} = LN(\sum_{k=0}^{l}{W_k^{l+1}y_k})
    """

    def __init__(self, args, is_encoder, include_sublayer=False):
        super(DynamicLinearCombination, self).__init__()
        self.normalize_learned_weight = args.normalize_learned_weight
        self.normalized_weight = None
        self.weight_type = args.weight_type
        self.out_dropout = args.history_dropout
        self.normalize_before = args.encoder_normalize_before if is_encoder else args.decoder_normalize_before
        self.dim = args.encoder_embed_dim if is_encoder else args.decoder_embed_dim

        # transformer encoder has 2 sub-layers, decoder has 3 sub-layers
        if include_sublayer:
            layer_num = 1 + (2 * args.encoder_layers if is_encoder else 3 * args.decoder_layers)
        else:
            layer_num = 1 + (args.encoder_layers if is_encoder else args.decoder_layers)

        # init weights and corresponding masks
        learnable = args.encoder_learnable if is_encoder else args.decoder_learnable
        self.weight, self.weight_mask = self._init(layer_num, args.init_value, args.weight_type,
                                                   args.history_window_size, learnable)

        # init triangular layer norm
        if args.normalize_embed:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(layer_num)])
        else:
            self.layer_norms = nn.ModuleList([nn.Sequential()] + [nn.LayerNorm(self.dim) for _ in range(layer_num - 1)])

        # states
        self.count = 0
        self.layers = []

    @staticmethod
    def _init_mask(n_layer, window_size):
        mask = np.zeros([n_layer, n_layer], dtype=np.float32)
        # all preceding layers
        if window_size == -1:
            for i in range(mask.shape[0]):
                mask[i, :(i + 1)] = 1
        else:
            for i in range(mask.shape[0]):
                mask[i, max(0, i + 1 - window_size): (i + 1)] = 1
        return torch.from_numpy(mask)

    @staticmethod
    def _init_weight(np_mask, dim=1, init_value='avg', learnable=True):
        np_weight = np.copy(np_mask)
        if init_value == 'avg':
            np_weight = np_weight / np.sum(np_weight, axis=1, keepdims=True)
        elif init_value == 'one':
            np_weight[:, :] = 1.
        else:
            raise ValueError('unknown init_value:{}'.format(init_value))
        weight_tensor = torch.from_numpy(np_weight).unsqueeze(2)
        if dim > 1:
            weight_tensor = weight_tensor.repeat(1, 1, dim)
        weight_tensor = torch.nn.Parameter(weight_tensor, requires_grad=learnable)
        return weight_tensor

    def _init(self, layer_num, init_value, weight_type, window_size=-1, learnable=True):
        """

        :param layer_num: total layers
        :param init_value: initial weight value
        :param weight_type: granularity of learned weights (scalar, scalar_X, vector)
        :param window_size: past windows size of layers
        :param learnable: if allow to learn weights
        :return:
            weight_tensor:
                1. L x L x 1 if weight type='scalar'
                2. L x L x X if weight type='scalar_X'
                3. L x L x H if weight type='vector'
            weight_mask: L x L, 0 means padding
        """
        """
            weight shape is:
             1. L x L x 1 for weight type='scalar'
             2. L x L x X for weight type='scalar_X'
             3. L x L x H for weight type='vector'
             mask shape is L x L
            :return:
        """
        # L x L
        mask_tensor = self._init_mask(layer_num, window_size)
        if weight_type == 'scalar':
            self.last_dim = 1
        elif weight_type == 'vector':
            self.last_dim = self.dim
        elif weight_type.startswith('scalar_'):
            n = int(weight_type.split('_')[1])
            assert self.dim % n == 0
            self.last_dim = n
        else:
            raise ValueError('unknown weight_type:{}'.format(weight_type))
        weight_tensor = self._init_weight(mask_tensor.numpy(), self.last_dim, init_value,
                                          learnable=learnable)
        return weight_tensor, mask_tensor

    def push(self, layer):
        self.count += 1

        # first layer
        if self.count == 1:
            self.layers.append(self.layer_norms[0](layer))
            # compatible when running on CPU
            if layer.is_cuda and not self.weight_mask.is_cuda:
                self.weight_mask = self.weight_mask.cuda()
            if self.normalize_learned_weight:
                weight = self.weight.masked_fill((self.weight_mask == 0).unsqueeze(2), float('-inf'))
                self.normalized_weight = F.softmax(weight, dim=1)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 1](layer)

        self.layers.append(layer)

    def _pick_weights(self):
        weight = self.normalized_weight if self.normalize_learned_weight else self.weight
        weight = weight[self.count - 1, : self.count, :].view(-1, 1, 1, self.last_dim)
        return weight

    def pop(self):
        assert len(self.layers) > 0

        # D x 1 x 1 x [1, H/G, H]
        weights = self._pick_weights()
        # D x T x B x H
        layers = torch.stack(self.layers, 0)
        # linear combination
        if self.weight_type in ['scalar', 'vector']:
            ret = (layers * weights).sum(0)
        else:
            D, T, B, H = layers.size()
            layers = layers.view(D, T, B, -1, weights.size(-1))
            weights = weights.unsqueeze(3)
            ret = (layers * weights).sum(0).view(T, B, H)

        if self.normalize_before:
            if self.out_dropout > 0:
                return F.dropout(ret, p=self.out_dropout, training=self.training)
            else:
                return ret
        if self.out_dropout > 0:
            return F.dropout(self.layer_norms[self.count - 1](ret), p=self.out_dropout, training=self.training)
        else:
            return self.layer_norms[self.count - 1](ret)

    def clean(self):
        self.count = 0
        self.layers = []

    def forward(self):
        pass


_MODEL_NAME_ = "dlcl_transformer"


@register_model("dlcl_transformer")
class DlclTransformerModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--inspect-grad', type=eval, default='False',
                            help='inspect intermediate gradient in tensorboard. Use more GPU memory')
        # DLCL parameters
        parser.add_argument('--init-value', type=str, default='avg', choices=['avg', 'one'],
                            help='how to init the learned weight matrix')
        parser.add_argument('--weight-type', type=str, default='scalar',
                            help='type of learned weight [scalar, scalar_n(n>1), vector]')
        parser.add_argument('--encoder-learnable', type=eval, default='True',
                            help='enable to learn weights for encoder')
        parser.add_argument('--decoder-learnable', type=eval, default='True',
                            help='enable to learn weights for decoder')
        parser.add_argument('--normalize-learned-weight', type=eval, default='False',
                            help='normalize learned weight by softmax')
        parser.add_argument('--normalize-embedding', type=eval, default='False',
                            help='normalize the input of embedding')
        parser.add_argument('--history-dropout', type=float, default=0.0, metavar='D',
                            help='dropout for history output')
        parser.add_argument('--history-window-size', type=int, default='-1',
                            help='how many past layers are considered. -1 means all')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = DlclTransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = DlclTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return DlclTransformerModel(encoder, decoder)


class DlclTransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.history = DynamicLinearCombination(args, is_encoder=True)
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        self.inspected_grads = OrderedDict() if getattr(args, 'inspect_grad', False) else None

    def forward(self, src_tokens, src_lengths):
        # clean layer history
        self.history.clean()

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        util.inspect_grad("encoder_0", x, self.inspected_grads)

        # push embedding layer into memory
        self.history.push(x)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer_id, layer in enumerate(self.layers):
            # fetch combined input from memory for the next layer
            x = self.history.pop()
            x = layer(x, encoder_padding_mask)
            # push into memory
            self.history.push(x)
            util.inspect_grad("encoder_%d" % (layer_id + 1), x, self.inspected_grads)

        # read from memory
        x = self.history.pop()
        if self.normalize:
            x = self.layer_norm(x)
        util.inspect_grad("encoder_top", x, self.inspected_grads)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            if 'encoder.embed_positions._float_tensor' not in state_dict:
                state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class DlclTransformerDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for i in range(args.decoder_layers)
        ])
        self.history = DynamicLinearCombination(args, is_encoder=False)
        self.normalize = args.decoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

        self.inspected_grads = OrderedDict() if getattr(args, 'inspect_grad', False) else None

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        self.history.clean()

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        util.inspect_grad('decoder_0', x, self.inspected_grads)
        # push embedding layer into memory
        self.history.push(x)

        # decoder layers
        for layer_id, layer in enumerate(self.layers):
            # read from memory
            x = self.history.pop()
            x, attn = layer(
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                incremental_state,
            )
            # write into memory
            self.history.push(x)
            util.inspect_grad('decoder_%d' % (layer_id + 1), x, self.inspected_grads)

        # read from memory
        x = self.history.pop()
        if self.normalize:
            x = self.layer_norm(x)
        util.inspect_grad('decoder_top', x, self.inspected_grads)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

        return x, attn

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before
        if self.embed_dim == args.encoder_embed_dim:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
        else:
            # support different dimension between encoder and decoder
            self.encoder_attn = MultiheadAttentionV2(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
                kv_dim=args.encoder_embed_dim
            )
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, init_size=num_embeddings)
    return m


@register_model_architecture(_MODEL_NAME_, _MODEL_NAME_)
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.inspect_grad = getattr(args, 'inspect_grad', False)

    # setting for DLCL
    args.init_value = getattr(args, 'init_value', 'avg')
    args.weight_type = getattr(args, 'weight_type', 'scalar')
    args.encoder_learnable = getattr(args, 'encoder_learnable', True)
    args.decoder_learnable = getattr(args, 'decoder_learnable', True)
    args.normalize_embed = getattr(args, 'normalize_embed', False)
    args.history_dropout = getattr(args, 'history_dropout', 0.0)
    args.history_window_size = getattr(args, 'history_window_size', -1)


@register_model_architecture("dlcl_transformer", 'dlcl_transformer_postnorm_wmt_en_de')
def dlcl_transformer_postnorm_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("dlcl_transformer", 'dlcl_transformer_postnorm_deep_wmt_en_de')
def dlcl_transformer_postnorm_deep_wmt_en_de(args):
    args.encoder_layers = 25
    dlcl_transformer_postnorm_wmt_en_de(args)


@register_model_architecture("dlcl_transformer", 'dlcl_transformer_prenorm_wmt_en_de')
def dlcl_transformer_prenorm_wmt_en_de(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    base_architecture(args)


@register_model_architecture("dlcl_transformer", 'dlcl_transformer_prenorm_deep_wmt_en_de')
def dlcl_transformer_prenorm_deep_wmt_en_de(args):
    args.encoder_layers = 30
    dlcl_transformer_prenorm_wmt_en_de(args)


@register_model_architecture(_MODEL_NAME_, '%s_iwslt_de_en' %_MODEL_NAME_)
def dlcl_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    base_architecture(args)


@register_model_architecture(_MODEL_NAME_, '%s_prenorm_iwslt_de_en' %_MODEL_NAME_)
def dlcl_transformer_prenorm_iwslt_de_en(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    dlcl_transformer_iwslt_de_en(args)


@register_model_architecture("dlcl_transformer", 'dlcl_transformer_toy')
def dlcl_transformer_toy(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 64)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 64)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    base_architecture(args)

@register_model_architecture("dlcl_transformer", 'dlcl_transformer_prenorm_deeper_wmt_en_de')
def dlcl_transformer_prenorm_deep_wmt_en_de(args):
    args.encoder_layers = 36
    dlcl_transformer_prenorm_wmt_en_de(args)
