import jittor as jt
from jittor import Module
import numpy as np
import math
import pickle
import torch

class MyCrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def execute(self, logits, labels):
        # logits = jt.nn.softmax(logits, dim=-1)
        print(labels)
        print(logits)
        print(logits.view(-1, logits.shape[-1]))
        print(labels.view(-1))
        ori_shape = labels.shape
        losses = jt.nn.cross_entropy_loss(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction=None).view(ori_shape)
        print(losses)
        llogits = torch.from_numpy(np.load('/root/root/Test/logits.npy'))
        llabels = torch.from_numpy(np.load('/root/root/Test/labels.npy'))
        llogits = torch.softmax(llogits, dim=-1)
        oori_shape = llabels.shape
        l = torch.nn.CrossEntropyLoss(reduction='none')
        llosses = l(llogits.view(-1, llogits.shape[-1]), llabels.view(-1)).view(oori_shape)
        print(llosses)
        mask = jt.nn.sign(jt.float32(labels))
        losses = mask * losses
        mean_loss = jt.mean(losses)
        return mean_loss

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    loss = MyCrossEntropyLoss()
    logits = jt.array(np.load('/root/root/Test/logits.npy'), 'float32')
    labels = jt.array(np.load('/root/root/Test/labels.npy'), 'int')
    
    print(loss(logits, labels))