import torch
import torch.nn as nn
import torch.nn.functional as F

class Model1(nn.Module):
    def __init__(self, input_dim):
        # default: He initialization: uniform(-bound, bound) where bound = sqrt(1/num_input_features)
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)  


class Model2(nn.Module):
    def __init__(self, input_dim, num_classes=7): 
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        # CrossEntropyLoss uses raw logits 
        return self.linear(x)

class RLoss(nn.Module):
    def __init__(self, model, alpha=1, lambda_reg=0.01, is_binary=True):
        super(RLoss, self).__init__()
        self.model = model  
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.is_binary = is_binary
    
    def forward(self, preds, labels):
        batch_size = preds.size(0)
        
        if self.is_binary:
            loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='sum') / batch_size
        else:
            loss = F.cross_entropy(preds, labels, reduction='sum') / batch_size
        r = 0.0
        for x in self.model.parameters():
            if x.requires_grad:
                x_sq = x ** 2
                r += torch.sum(self.alpha * x_sq / (1 + self.alpha * x_sq))
        
        return loss + self.lambda_reg * r

        

            