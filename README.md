# README

## 项目概述
本项目使用带有注意力机制的ResNet-18模型进行CIFAR-10图像分类。主要包含数据预处理、模型定义、训练和测试等步骤。

## 目录结构
- `data/`: 数据存储目录
- `models/`: 模型定义和存储目录
- `scripts/`: 数据处理和训练脚本
- `outputs/`: 训练后的模型和结果存储目录

## 依赖项
运行此项目需要安装以下依赖项：
- `pandas`
- `torch`
- `torchvision`
- `d2l`

安装方式：
```bash
pip install pandas torch torchvision d2l
```

## 数据预处理
从CIFAR-10数据集中读取图像和标签，并进行数据增强和标准化处理。将数据划分为训练集、验证集和测试集。

## 模型定义
定义带有注意力机制的ResNet-18模型，包括模型的各个层次和注意力模块。

## 训练与验证
使用AdamW优化器和学习率调度器进行模型训练，并在每个epoch结束时计算验证集的准确率。

## 测试与保存结果
训练完成后，对测试集进行预测，并将预测结果保存为CSV文件。

## 运行步骤
1. 下载并解压CIFAR-10数据集。
2. 运行数据预处理脚本，组织数据结构。
3. 运行训练脚本，进行模型训练和验证。
4. 运行测试脚本，进行模型预测并保存结果。

## 示例
```bash
# 下载数据集
python scripts/download_data.py

# 数据预处理
python scripts/preprocess_data.py

# 训练模型
python scripts/train_model.py

# 测试模型并保存结果
python scripts/test_model.py
```

