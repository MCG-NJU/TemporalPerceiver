"""
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/models/transformer.py).

Temporal Perceiver Detector Class Contains:
    * Feature Ranking
    * A Encoder based on Transformer Decoder Structure 
    * A Transformer Decoder
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Dectector(nn.Module):

    def __init__(self, bc_ratio, compression_ratio, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        
        encoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model)

        self.encoder = TransformerDecoder(encoder_layer, num_encoder_layers, encoder_norm, 
                                          return_intermediate=False)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.bc_ratio = bc_ratio
        self.compression_ratio = compression_ratio

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def rearrange_feature(self, scores, src, pos_embed):
        n, b, d = pos_embed.shape
        scores = scores.squeeze()
        scores_sortedIdx = torch.argsort(scores,dim=0,descending=True)
        m = self.compression_ratio * n
        k = int(m * self.bc_ratio)
        
        scores_topkIdx = scores_sortedIdx[:k,:]
        scores_Rest = scores_sortedIdx[k:,:]
        scores_topkIdx = torch.sort(scores_topkIdx, dim=0)[0]
        scores_Rest = torch.sort(scores_Rest, dim=0)[0]
        
        new_pos = torch.zeros((n, b, d))
        new_src = torch.zeros((n, b, d))

        for i in range(b):
            new_pos[:k,i,:] = pos_embed[scores_topkIdx[:,i],i,:]
            new_pos[k:,i,:] = pos_embed[scores_Rest[:,i],i,:]
            new_src[:k,i,:] = src[scores_topkIdx[:,i],i,:]
            new_src[k:,i,:] = src[scores_Rest[:,i],i,:]

        new_src = new_src.to(pos_embed.device)
        new_pos = new_pos.to(pos_embed.device)

        return new_src, new_pos

    def forward(self, src, query_embed, scores, pos_embed, latent_embed):
        bs, t, _ = src.shape
        scores = scores.view(bs, t, -1).permute(1, 0, 2)
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        latent_embed = latent_embed.unsqueeze(1).repeat(1,bs,1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        tgt_enc = torch.zeros_like(latent_embed)
        tgt_dec = torch.zeros_like(query_embed)

        # 1. Rank input features and position embedding by coherence scores.
        src_sorted, pos_embed_sorted = self.rearrange_feature(scores, src, pos_embed)
        
        # 2. Encoder: Compress input feature of length N into latents of length M.
        memory, en_crossattn_map = self.encoder(tgt_enc, src_sorted, memory_key_padding_mask=None, pos=pos_embed_sorted, query_pos=latent_embed)

        # 3. Decoder: Decode boundaries from latents.
        hs, _ = self.decoder(tgt_dec, memory, memory_key_padding_mask=None, pos=latent_embed, query_pos=query_embed)
        
        return hs.transpose(1, 2), en_crossattn_map[-1]

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        crossattn_map = []

        for layer in self.layers:
            output, _, crossattn = layer(output, memory, tgt_mask=tgt_mask, \
                           memory_mask=memory_mask, \
                           tgt_key_padding_mask=tgt_key_padding_mask, \
                           memory_key_padding_mask=memory_key_padding_mask, \
                           pos=pos, query_pos=query_pos)
            
            crossattn_map.append(crossattn)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(crossattn_map)

        return output, torch.stack(crossattn_map)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, selfattn_map = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, crossattn_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, selfattn_map, crossattn_map

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, selfattn_map = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, crossattn_map = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, selfattn_map, crossattn_map

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_detector(args):
    return Dectector(
        bc_ratio=args.bc_ratio,
        compression_ratio=args.compress_ratio,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
