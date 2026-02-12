"""
    多次更新参数，利用反向传播就可以找到最适合的参数，使得损失函数值最小
"""

import torch

# 定义变量，初始的权重
# 参数：(w初始值， 自动微分开启， 值类型改为torch.float32)
w = torch.tensor(10, requires_grad=True, dtype=torch.float32)

# 定义loss函数
loss = w ** 2 + 20

# 利用梯度下降法，循环迭代100次，求最优解
# 打印初始数值
print(f'开始\n 权重初始值:{w}, (0.01 * grad):无, loss:{loss}')

for i in range(1, 101):
    # 正向计算（前向传播）
    loss = w ** 2 + 20  # 每次更新w之后都需要重新计算损失值

    # 判断grad是否为空，因为第一次更新时grad为None，调用清0函数会报错，只有反向传播后，grad才不为None
    if w.grad is not None:
        w.grad.zero_()  # 为什么要清0？因为grad会默认做累加

    # 反向传播
    loss.sum().backward()

    # 更新参数
    w.data = w.data - 0.01 * w.grad

    # 打印信息查看
    print(f'第{i}次更新，权重初始值:{w}, loss:{loss:.5f}')

# 打印最终结果
print(f'最终结果 权重:{w}, 梯度:{w.grad}, loss:{loss:.5f}')