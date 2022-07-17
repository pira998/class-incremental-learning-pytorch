import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utils import freeze_parameters, get_backbone

def freeze_parameters(m, requires_grad=False):
    if m is None:
        return
    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    for p in m.parameters():
        p.requires_grad = requires_grad
    
class CilClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(CilClassifier, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.heads = nn.ModuleList([nn.Linear(self.embedding_size, self.num_classes).cuda()]) # for the first head
    
    def __getitem__(self, i):
        return self.heads[i]

    def __len__(self):
        return len(self.heads)

    def forward(self, x):
        logits = torch.cat([head(x) for head in self.heads], dim=1)
        return logits
    
    def adaption(self, num_classes):
        self.heads.append(nn.Linear(self.embedding_size, num_classes).cuda())
    
class CilModel(nn.Module):
    def __init__(self, backbone):
        super(CilModel, self).__init__()
        self.backbone = get_backbone(backbone)
        self.fc = None
    
    @property
    def feature_dim(self):
        return self.backbone.out_dim
    
    def extract_vector(self, x):
        return self.backbone(x) # extract feature vector

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        return out,x
    
    def copy(self):
        return copy.deepcopy(self)
    
    def freeze(self, names=['all']):
        freeze_parameters(self, requires_grad=True)
        self.train()
        for names in names:
            if names == 'all':
                freeze_parameters(self) # freeze all parameters except the last layer
                self.eval()
            elif names == 'backbone':
                freeze_parameters(self.backbone)
                self.backbone.eval()
            elif names == 'fc':
                freeze_parameters(self.fc)
                self.fc.eval()
            else:
                raise ValueError('Unknown name: {}'.format(names))

        return self
    
    def prev_model_adaption(self, num_classes):
        if self.fc is None:
            self.fc = CilClassifier(self.feature_dim, num_classes).cuda() # CilClassifier
        else:
            self.fc.adaption(num_classes)   # add a new head
        return self
    
    def after_model_adaption(self, num_classes, args):
        if args.task_id > 0:
            self.weight_align(num_classes) # weight align the last layer

    @torch.no_grad()
    def weight_align(self, num_new_classes):
        w = torch.cat([head.weight.data for head in self.fc], dim=0)
        norms = torch.norm(w, dim=1) # norm is a vector of length num_classes

        norm_old = norms[:-num_new_classes] # norm of old classes
        norm_new = norms[-num_new_classes:] # norm of new classes

        gamma = torch.mean(norm_old) / torch.mean(norm_new) # gamma is a scalar
        print('weight align gamma(old norm/new norm): {}'.format(gamma))
        self.fc[-1].weight.data = gamma * w[-num_new_classes:]




