"""
    神经网络的底层都是在进行张量的计算
    同一张量中的类型必须一致，且必须都是数值

    张量的基本创建方式：
        1、torch.tensor 根据指定数据创建张量
        2、torch.Tensor 根据形状创建张量，也可用来创建指定数据的张量
        3、troch.IntTensor、troch.FloatTensor、troch.DoubleTensor 创建指定类型的张量
"""

import torch

# 1、torch.tensor 根据指定数据创建张量
def dm01():
    # 标量张量
    t1 = torch.tensor(10)
    print(f't1: {t1}, type: {type(t1)}')

# 2、torch.Tensor 根据形状创建张量，也可用来创建指定数据的张量

# 3、troch.IntTensor、troch.FloatTensor、troch.DoubleTensor 创建指定类型的张量

if __name__ == '__main__':
    dm01()