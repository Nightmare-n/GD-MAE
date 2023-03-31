# using flat2win_v2 without voxel_drop_level
import torch
import torch.nn as nn
from .cosine_msa import CosineMultiheadAttention
from . import sst_utils


class WindowAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, batch_first=False, layer_cfg=dict()):
        super().__init__()
        self.nhead = nhead
        if layer_cfg.get('cosine', False):
            tau_min = layer_cfg.get('tau_min', 0.01)
            self.self_attn = CosineMultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=False, tau_min=tau_min,
                cosine=True,
                non_shared_tau=layer_cfg.get('non_shared_tau', False)
            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, feat_2d, pos_dict, ind_dict, key_padding_dict):
        '''
        Args:
        Out:
            shifted_feat_dict: the same type as window_feat_dict
        '''
        out_feat_dict = {}

        feat_3d_dict = sst_utils.flat2window_v2(feat_2d, ind_dict)

        for name in feat_3d_dict:
            #  [n, num_token, embed_dim]
            pos = pos_dict[name]

            feat_3d = feat_3d_dict[name]
            feat_3d = feat_3d.permute(1, 0, 2)

            v = feat_3d

            if pos is not None:
                pos = pos.permute(1, 0, 2)
                assert pos.shape == feat_3d.shape, f'pos_shape: {pos.shape}, feat_shape:{feat_3d.shape}'
                q = k = feat_3d + pos
            else:
                q = k = feat_3d

            key_padding_mask = key_padding_dict[name]
            out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)
            out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)

        results = sst_utils.window2flat_v2(out_feat_dict, ind_dict)

        return results


class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, mlp_dropout=0, layer_cfg=dict()):
        super().__init__()
        assert not batch_first, 'Current version of PyTorch does not support batch_first in MultiheadAttention. After upgrading pytorch, do not forget to check the layout of MLP and layer norm to enable batch_first option.'
        self.batch_first = batch_first
        self.win_attn = WindowAttention(d_model, nhead, dropout, layer_cfg=layer_cfg)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos_dict, ind_dict, key_padding_mask_dict):
        src2 = self.win_attn(src, pos_dict, ind_dict, key_padding_mask_dict)  # [N, d_model]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BasicShiftBlockV2(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.'''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, layer_cfg=dict()):
        super().__init__()

        encoder_1 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_cfg=layer_cfg)
        encoder_2 = EncoderLayer(d_model, nhead, dim_feedforward, dropout,
            activation, batch_first, layer_cfg=layer_cfg)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(self, src, pos_dict_list, ind_dict_list, key_mask_dict_list):
        num_shifts = len(pos_dict_list)
        assert num_shifts in (1, 2)

        output = src
        for i in range(2):
            this_id = i % num_shifts
            pos_dict = pos_dict_list[this_id]
            ind_dict = ind_dict_list[this_id]
            key_mask_dict = key_mask_dict_list[this_id]

            layer = self.encoder_list[i]
            output = layer(output, pos_dict, ind_dict, key_mask_dict)

        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
