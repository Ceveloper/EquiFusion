
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
import os
import torch

class MetricStorer(Metric):
    def __init__(self, output_transform=lambda x: x, return_op='mean'):
        self.cumulator = None
        self.count = 0
        self.return_op = return_op
        assert return_op in ['mean', 'avg', 'max', 'min']
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.cumulator = 0. if self.return_op != 'min' else 1000000.
        self.count = 0
        self.tot = 0
        super().reset()

    def update(self, output):
        # pred, class_idxs, step = self._output_transform(output) # This will be called by engine if we use it

        output = output.cpu().numpy()
        
        if self.return_op in ['mean', 'avg']:
            self.cumulator += output.sum(0)
        elif self.return_op == 'max':
            current_max = output.max(0)
            self.cumulator = self.cumulator if self.cumulator > current_max else current_max
        elif self.return_op == 'min':
            current_min = output.min(0)
            self.cumulator = self.cumulator if self.cumulator < current_min else current_min
        
        self.count += output.shape[0] if self.return_op in ['mean', 'avg'] else 0
        self.tot = self.cumulator/self.count if self.return_op in ['mean', 'avg'] else self.cumulator

    def compute(self):
        if self.cumulator is None or self.count == None:
            raise NotComputableError('MetricStorer must have at least one example before it can be computed.')
        self.tot = self.cumulator/self.count if self.return_op in ['mean', 'avg'] else self.cumulator
        tot = self.tot
        return tot
    
