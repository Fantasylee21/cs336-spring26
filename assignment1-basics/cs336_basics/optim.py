import math
import torch
import torch.nn as nn
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                state['step'] += 1

                alpha = lr * (math.sqrt(1.0 - beta2 ** state['step']) / (1.0 - beta1 ** state['step']))
                
                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # Update biased second raw moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                p.data.addcdiv_(m, v.sqrt().add(eps), value=-alpha)
                # Update parameters

        return loss

def learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        return min_learning_rate


