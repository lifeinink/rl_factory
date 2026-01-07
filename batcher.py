import numpy as np
from sklearn.utils.random import sample_without_replacement
import time
import torch
#Depricated
#class Batcher:
#    def __init__(self,batch_size: int):
#        self.batch_size = batch_size
#    
#    def mini_batch(self,x: torch.Tensor,y: torch.Tensor = None, extras: list[torch.Tensor]=None) -> tuple[torch.Tensor,torch.Tensor, list[torch.Tensor]]:
#        seed = int(time.time())
#        batches = sample_without_replacement(x.shape[0],self.batch_size,random_state=seed)
#        mini_batched_x = x[batches]
#        mini_batched_y = None
#        if y != None:
#            mini_batched_y = y[batches]
#        if extras != None:
#            batched_extras = []
#            for extra in extras:
#                batched_extras.append(extra[batches])
#            return mini_batched_x, mini_batched_y, batched_extras
#        return mini_batched_x, mini_batched_y
