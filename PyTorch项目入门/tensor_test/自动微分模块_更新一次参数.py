"""
    演示自动微分模块如何求导
    w新 = w旧 - 学习率 * 梯度，梯度 = 损失函数的导数

    注意：只有标量张量才能求导，且大多数底层操作的都是浮点型
"""

import torch

# 定义变量，初始的权重
# 参数：(w初始值， 自动微分开启， 值类型改为torch.float32)
w = torch.tensor(10, requires_grad=True, dtype=torch.float32)

# 定义loss函数
loss = 2 * w ** 2

# 计算梯度，也就是损失函数的导数
# loss.backward()     这种写法可以，因为loss本身就是标量
loss.sum().backward() # 保证loss是标量，进行求导，结果保存在w的grad成员中

# 代入权重更新公式
w.data = w.data - 0.01 * w.grad

# 打印结果
print(f'更新后的权重:{w}')