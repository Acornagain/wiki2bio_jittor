import jittor as jt
import numpy as np
from jittor.optim import Optimizer

class AdamClip(Optimizer):
    """ Adam Optimizer.
    
    Example::

        optimizer = nn.Adam(model.parameters(), lr, eps=1e-8, betas=(0.9, 0.999))
        optimizer.step(loss)
    """
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0):
        super().__init__(params, lr)
        self.clip = 5.0
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        # assert weight_decay==0, "weight_decay is not supported yet"
        
        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            m = pg["m"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())
                m.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        m = group["m"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
            m.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)


    def step(self, loss=None):
        if loss is not None:
            self.pre_step(loss)
        n = float(self.n_step)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            b0, b1 = pg.get("betas", self.betas)
            # pg["grads"] = self.clip_by_global_norm(pg["grads"], self.clip)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"], pg["m"]):
                if p.is_stop_grad(): continue
                g = p * weight_decay + g
                m.update(b0 * m + (1-b0) * g)
                v.update(b1 * v + (1-b1) * g * g)
                step_size = lr * jt.sqrt(1-b1**n) / (1-b0 ** n)
                p.update(p - m * step_size / (jt.sqrt(v) + eps))
        self.zero_grad()


    def clip_by_global_norm(self, x, clip_norm):
        global_norm = 0
        for g in x:
            global_norm += jt.sum(g*g)
        global_norm = jt.sqrt(global_norm)
        rt = []
        for g in x:
            rt.append(g * clip_norm / jt.maximum(global_norm, clip_norm))
        return rt