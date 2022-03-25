import torch
t = torch.tensor([1.0]) # create tensor with just a 1 in it
t = t.cuda() # Move t to the gpu
print(t) # Should print something like tensor([1], device='cuda:0')
print(t.mean()) # Test an operation just to be sure
