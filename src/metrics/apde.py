
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
import pandas as pd
import numpy as np
import os
import torch

class MetricStorerAPDE(Metric):

    def __init__(self, output_transform=lambda x: apd(x), return_op='mean', mmapd_gt_path=''):
        self.cumulator = None
        self.count = 0
        self.return_op = return_op
        assert return_op in ['mean', 'avg']
        assert os.path.exists(mmapd_gt_path), f"Cannot find mmapd_GT.csv in {mmapd_gt_path}"
        mmapd_gt = pd.read_csv(mmapd_gt_path, index_col=0)["gt_APD"]
        mmapd_gt = mmapd_gt.replace(0, np.nan)
        self.mmapd_gt = mmapd_gt
        self.index = 0
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.cumulator = 0. if self.return_op != 'min' else 1000000.
        self.count = 0
        self.tot = 0
        super().reset()

    def update(self, output):
        # pred, class_idxs, step = self._output_transform(output) # This will be called by engine if we use it

        output = output.cpu()
        batch_size = output.shape[0]
        mmapde = (output - self.mmapd_gt[self.index:self.index+batch_size].values).abs()
        self.index += batch_size
        
        if self.return_op in ['mean', 'avg']:
            self.cumulator += mmapde.nansum(0)
        
        self.count += (~torch.isnan(mmapde)).sum()
        self.tot = self.cumulator/self.count if self.return_op in ['mean', 'avg'] else self.cumulator

    def compute(self):
        if self.cumulator is None or self.count == None:
            raise NotComputableError('MetricStorer must have at least one example before it can be computed.')
        self.tot = self.cumulator/self.count if self.return_op in ['mean', 'avg'] else self.cumulator
        tot = self.tot
        return tot
