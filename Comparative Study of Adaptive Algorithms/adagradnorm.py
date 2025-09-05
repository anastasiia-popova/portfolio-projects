import torch.optim as optim
import torch 
import torch.nn as nn

# implemented by Pakeeza Ehsan
class AdagradNorm(optim.Optimizer):
    def __init__(self, params, lr=0.01, epsilon=1e-8):
        defaults = dict(lr=lr, epsilon=epsilon)
        super(AdagradNorm, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['sum_sq_grad'] = torch.zeros_like(p.data)
                
                sum_sq_grad = state['sum_sq_grad']
                
                # Compute L2 norm of gradient
                grad_norm = torch.norm(grad, p=2)
                if grad_norm > 0:
                    grad = grad / grad_norm  # Normalize gradient
                
                # Update sum of squared gradients
                sum_sq_grad.addcmul_(grad, grad)
                
                # Compute adaptive learning rate
                adaptive_lr = group['lr'] / (sum_sq_grad.sqrt() + group['epsilon'])
                
                # Update parameters
                p.data.addcmul_(-1, adaptive_lr, grad)
        
        return loss