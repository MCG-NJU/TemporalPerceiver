"""
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/models/detr.py)

Temporal Perceiver model, criterion and post-process classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import ops
from util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .detector import build_detector
from .position_embedding import build_position_embedding


class TemporalPerceiver(nn.Module):
    """ This is the TemporalPerceiver module that performs generic boundary detection """
    def __init__(self, position_embedding, detector, num_classes, num_queries, num_input, compress_ratio, aux_loss=False, use_class_dim=False):
        """ Initializes the model.
        Parameters:
            detector: main component of Temporal Perceiver, built on top of torch module of the transformer decoder architecture. See detector.py
            num_classes: number of boundary classes, in the GBD case equals to 1.
            num_queries: number of boundary queries, ie boundary slot. This is the maximal number of boundaries
                         Temporal Perceiver can detect in a single video snippet. 
            num_input: number of input video frames, ie the length of the given video snippet.
            compress_ratio: ratio of feature compression; multiplied with num_input, the ratio decides the number of latents in encoder.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.detector = detector
        input_dim = 2048
        hidden_dim = detector.d_model
        self.hidden_dim = hidden_dim
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)
        self.loc_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.latent_embed = nn.Embedding(int(num_input * compress_ratio), hidden_dim)
        
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        
        self.aux_loss = aux_loss
        self.position_embedding = position_embedding

    def forward(self, locations, samples, coherence_scores):
        """ Parameters:
               - locations: batched location sequences that decodes the relative temporal location of each input frame,
                    of shape [batch_size, num_input, 1]
               - samles: batched video frame features, of shape [batch_size, num_input, 2048], 
                    where 2048 is the number of channels in video features.
               - coherence_scores: batched coherence scores, of shape [batch_size, num_input, 1] 

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boundaries": The normalized boundaries coordinates for all queries. 
                               These values are normalized in [0, 1], relative to the number of input frames.
                               See PostProcess for information on how to retrieve the unnormalized boundary locations.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """        
        bs = locations.shape[0]

        features_flatten = samples.flatten(0,1)
            
        projected_fts = self.input_proj(features_flatten.unsqueeze(-1).unsqueeze(-1))
        features = projected_fts.view((bs, -1, self.hidden_dim))
        
        pos = self.position_embedding(locations)

        hs, crossattn = self.detector(features, self.query_embed.weight, coherence_scores, pos, self.latent_embed.weight)

        outputs_class = self.cls_head(hs)
        outputs_coord = self.loc_head(hs).sigmoid()
        
        out = {'pred_logits': outputs_class[-1], 'pred_boundaries': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out, crossattn

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boundaries': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Temporal Perceiver.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boundaries and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and boundary)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, bc_ratio):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            bc_ratio: ratio of number of boundary queries over the number of context queries, 
                used only in computing loss_alignment.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.bc_ratio = bc_ratio

    def loss_labels(self, outputs, crossattn, targets, indices, num_boundaries, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boundaries]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        if indices is None:
            loss_cls = 0 * src_logits.sum()
            losses = {'loss_cls': loss_cls}
            if log:
                losses['class_error'] = loss_cls
            return losses

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_cls': loss_cls}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_alignment(self, outputs, crossattn, targets, indices, num_boundaries):
        """Alignment loss 
        The loss is emposed only on the encoder cross attention maps, 
            to align the learning for both boundary and context queries.
        """
        crossattn = torch.softmax(crossattn, dim=1)
        mask = torch.zeros_like(crossattn)
        _, m, _ = crossattn.shape

        k = int(m * self.bc_ratio)
        for i in range(k):
            mask[:,i,i] = 1.0

        sumLogits = torch.sum(crossattn * mask, dim=1)
        loss_alignment = torch.mean(-torch.log(sumLogits+1e-4))

        losses = {'loss_alignment': loss_alignment}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, crossattn, targets, indices, num_boundaries):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boundaries
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        if indices is None:
            losses = {'cardinality_error': torch.tensor(0.).to(outputs['pred_logits'].device)}
            return losses
        
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boundaries(self, outputs, crossattn, targets, indices, num_boundaries):
        """Compute the losses related to the boundary locations, the L1 regression loss
           targets dicts must contain the key "boundaries" containing a tensor of dim [n_gt, 1].
           The target boundaries are expected in relative location to the given window snippet.
        """
        assert 'pred_boundaries' in outputs

        if indices is None:
            loss_loc = 0 * outputs['pred_boundaries'].sum()
            return {'loss_loc': loss_loc}
        
        idx = self._get_src_permutation_idx(indices)
        src_boundaries = outputs['pred_boundaries'][idx]
        target_boundaries = torch.cat([t['boundaries'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_loc = F.l1_loss(src_boundaries.view(-1), target_boundaries, reduction='none')

        losses = {}
        losses['loss_loc'] = loss_loc.sum() / num_boundaries

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, crossattn, targets, indices, num_boundaries, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boundaries': self.loss_boundaries,
            'alignment': self.loss_alignment
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, crossattn, targets, indices, num_boundaries, **kwargs)

    def forward(self, outputs, crossattn, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             crossattn: matrix of the last layer of encoder cross-attention, see loss_alignment function.
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boundaries accross all nodes, for normalization purposes
        num_boundaries = sum(len(t["labels"]) for t in targets)
        num_boundaries = torch.as_tensor([num_boundaries], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boundaries)
        num_boundaries = torch.clamp(num_boundaries / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, crossattn, targets, indices, num_boundaries))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, crossattn, targets, indices, num_boundaries, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, args):
        super().__init__()
        self.window_size = args.window_size
        self.interval = args.interval

    @torch.no_grad()
    def forward(self, outputs, num_frames, base):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            num_frames: tensor of dimension [batch_size x 1] containing the frame duration of each videos of the batch.
            base: tensor of dimension [batchsize x 1] containing the starting locations of each video of the batch.
        """
        out_logits, out_boundaries = outputs['pred_logits'], outputs['pred_boundaries']

        assert len(out_logits) == len(num_frames)
        num_frames = num_frames.reshape((len(out_logits), 1))

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boundaries = ops.prop_relative_to_absolute(out_boundaries, base, self.window_size, self.interval)
 
        results = [{'scores': s, 'labels': l, 'boundaries': b} for s, l, b in zip(scores, labels, boundaries)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 1
    device = torch.device(args.device)

    position_embedding = build_position_embedding(args)

    detector = build_detector(args)

    model = TemporalPerceiver(
        position_embedding=position_embedding,
        detector=detector,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_input=args.window_size,
        compress_ratio = args.compress_ratio,
        aux_loss=args.aux_loss
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_cls': args.cls_loss_coef, 'loss_loc': args.loc_loss_coef, 'loss_alignment':args.align_loss_coef}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boundaries', 'cardinality', 'alignment']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, bc_ratio=args.bc_ratio,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'boundaries': PostProcess(args)}

    return model, criterion, postprocessors
