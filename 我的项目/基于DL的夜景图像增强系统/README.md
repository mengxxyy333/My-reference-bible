# 夜景图像增强系统

## 项目简介
本项目实现了一个基于深度学习的夜景图像增强系统，包含数据预处理与加载、Retinex 启发增强网络训练、单张/批量推理以及 PyQt 图形交互界面。

## 数据集结构
默认使用 LOL 数据集，目录结构如下：

- `data/train/low`
- `data/train/high`
- `data/val/low`
- `data/val/high`

程序会自动按文件名进行低照度图像和高质量图像配对。

## 功能模块
1. 数据预处理：保持宽高比缩放、边缘填充、RGB 转换、归一化、配对增强。
2. 模型训练：采用 ImageNet 预训练 VGG-16 前 13 层特征作为编码器浅层主干，结合 Retinex 启发的编码器-解码器增强网络。
3. 组合损失：感知损失 + SSIM 损失 + RGB 三通道 L1 颜色损失。
4. GPU 训练/推理：自动优先选择 CUDA。
5. GUI 系统：支持单图/批量导入、结果切换、直方图显示、滚轮缩放、中间轮次预览、结果导出。

## 安装依赖
```bash
pip install -r requirements.txt
```

## 训练模型
```bash
python train.py
```

训练结果会保存在：
- `checkpoints/`：模型权重
- `outputs/progress/`：0、20、50、80、100 轮预览图
- `outputs/logs/train_history.json`：训练日志

## 单张/批量推理
```bash
python predict.py "待增强图片或文件夹路径"
```

可选参数：
```bash
python predict.py "输入路径" --checkpoint "checkpoints/latest.pth" --output "outputs/batch"
```

## 启动图形界面
```bash
python app.py
```

## GUI 使用说明
- 点击“导入单张”或“导入文件夹”加载待增强图像。
- 点击“加载模型”选择训练好的 `.pth` 权重。
- 点击“增强当前图像”生成结果。
- 原图和增强图通过下拉框单独切换，不会互相覆盖。
- 支持查看 0/20/50/80/100 轮的中间效果与训练指标。
- 双击或滚轮可缩放预览图。
- 点击“导出当前结果”或“批量导出”保存增强图片，文件名自动添加 `_enhanced` 后缀。

## 说明
- 第一次训练/推理时会自动下载 VGG16 预训练权重，需要网络连接。
- 若没有 GPU，程序会自动退回 CPU，但训练速度会明显变慢。
