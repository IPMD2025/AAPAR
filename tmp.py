import torch
x1 = torch.rand((2,10))
x2 = torch.rand((2,10))
x1 = torch.unsqueeze(x1, dim= 1)
x2 = torch.unsqueeze(x2, dim= 1)
x_max = torch.cat([x1,x2], dim = 1)
print(x1)
print(x2)

print(torch.max(x_max, dim = 1)[0])