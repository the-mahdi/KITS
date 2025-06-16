import sys
import numpy as np
import torch

from . import TemporalDataset, SpatioTemporalDataset


class ImputationDataset(TemporalDataset):

    def __init__(self, data,
                 index=None,
                 mask=None,
                 eval_mask=None,
                 freq=None,
                 trend=None,
                 scaler=None,
                 window=24,
                 stride=1,
                 exogenous=None):
        # data: (34272, 207)
        # index: (34272, )
        # mask: (34272, 207)
        # eval_mask: (34272, 207)
        # freq: None
        # trend: None
        # scaler: None
        # window: 24
        # stride: 1
        # exogenous: None
        if mask is None:  # ignore
            mask = np.ones_like(data)
        if exogenous is None:  # run
            exogenous = dict()
        exogenous['mask_window'] = mask
        if eval_mask is not None:
            exogenous['eval_mask_window'] = eval_mask
        super(ImputationDataset, self).__init__(data,
                                                index=index,
                                                exogenous=exogenous,
                                                trend=trend,
                                                scaler=scaler,
                                                freq=freq,
                                                window=window,
                                                horizon=window,
                                                delay=-window,
                                                stride=stride)

    def get(self, item, preprocess=False):
        # preprocess: true
        res, transform = super(ImputationDataset, self).get(item, preprocess)
        # res['x']: (24, 207) - unmasked
        # res['mask']: (24, 207)
        # transform['scale']: (1, 207)
        # transform['bias']: (1, 207)
        res['x'] = torch.where(res['mask'], res['x'], torch.zeros_like(res['x']))
        return res, transform


class GraphImputationDataset(ImputationDataset, SpatioTemporalDataset):
    def __init__(self, data,
                 index=None,
                 mask=None,
                 eval_mask=None,
                 freq=None,
                 trend=None,
                 scaler=None,
                 window=24,
                 stride=1,
                 exogenous=None,
                 node_feats=None):
        super(GraphImputationDataset, self).__init__(data,
                                                     index=index,
                                                     mask=mask,
                                                     eval_mask=eval_mask,
                                                     freq=freq,
                                                     trend=trend,
                                                     scaler=scaler,
                                                     window=window,
                                                     stride=stride,
                                                     exogenous=exogenous)

        if node_feats is not None:
            node_feats = self.check_input(node_feats)
            if node_feats.ndim == 2:
                node_feats = node_feats.unsqueeze(0)
            elif node_feats.ndim == 1:
                node_feats = node_feats.view(1, -1, 1)
            node_feats = node_feats.repeat(self.n_steps, 1, 1)
            self.add_exogenous(node_feats, "node_feats", for_window=True, for_horizon=True)
