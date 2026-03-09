"""
    如果一个张量开启自动微分，那么它就不能直接转为numpy数组，需要detach函数来解决
"""

import torch

# 创建一个开启自动微分的张量
t = torch.tensor([10, 20], requires_grad=True, dtype=torch.float)

# 直接转会报错
# t.numpy()
# print(f't:{t.type()}')

# 调用detach，再转就可以了
n = t.detach().numpy()
print(f'n:{type(n)}')