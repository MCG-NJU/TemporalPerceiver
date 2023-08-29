"""
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/models/matcher.py)

Hungarian Matcher: Module to compute the matching cost and solve the corresponding LSAP.

"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_cls: float = 1, cost_loc: float = 1):
        """Creates the matcher

        Params:
            cost_cls: This is the relative weight of the classification error in the matching cost
            cost_loc: This is the relative weight of the L1 error of the boundary frames in the matching cost
        """
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_loc = cost_loc
        assert cost_cls != 0 or cost_loc != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boundaries": Tensor of dim [batch_size, num_queries, 1] with the predicted boundary frames

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boundaries] (where num_target_boundaries is the number of ground-truth
                           objects in the target) containing the class labels
                 "boundaries": Tensor of dim [num_target_boundaries, 1] containing the target boundary frame.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boundaries)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  
        out_loc = outputs["pred_boundaries"].flatten(0, 1) 

        # Also concat the target labels and boundaries
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_loc = torch.cat([v["boundaries"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_cls = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boundaries
        cost_loc = torch.abs(out_loc-tgt_loc)

        # Final cost matrix
        C = self.cost_loc * cost_loc + self.cost_cls * cost_cls
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boundaries"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
def build_matcher(args):
    return HungarianMatcher(cost_cls=args.set_cost_cls, cost_loc=args.set_cost_loc)
