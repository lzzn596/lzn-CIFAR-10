!pip install pandas
!pip install torch
!pip install torchvision
!pip install d2l

import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

#@save
# 下载数据集
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip', '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# 使用完整数据集时设置demo为False
demo = False
if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../sagemaker-studiolab-notebooks/cifar-10/'

#@save
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels('../sagemaker-studiolab-notebooks/trainLabels.csv')
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))

#@save
def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

#@save
def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))


def reorg_cifar10_data(data_dir, valid_ratio):
    # 读取训练集标签
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    # 拆分训练集和验证集，并整理测试集
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
# 重新组织 CIFAR-10 数据集，包括拆分训练集和验证集，以及整理测试集的目录结构
reorg_cifar10_data(data_dir, valid_ratio)

# 数据增强
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),  # 将图像大小调整为40x40像素
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),  # 随机裁剪并缩放到32x32像素
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.ToTensor(),  # 转换为Tensor格式
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # 标准化
])

# 测试集的转换操作
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 转换为Tensor格式
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # 标准化
])

# 训练集和训练验证集的数据集定义，应用训练数据增强的转换
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder), transform=transform_train) for folder in ['train', 'train_valid']]

# 验证集和测试集的数据集定义，应用测试数据的转换
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder), transform=transform_test) for folder in ['valid', 'test']]

# 训练集和训练验证集的数据迭代器定义，每次从数据集中加载一个批次数据，打乱并丢弃最后一个不完整的批次
train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
                                for dataset in (train_ds, train_valid_ds)]

# 验证集的数据迭代器定义，每次从验证集中加载一个批次数据，不打乱顺序，并且不丢弃最后一个不完整的批次
valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)

# 测试集的数据迭代器定义，每次从测试集中加载一个批次数据，不打乱顺序，并且不丢弃最后一个不完整的批次
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

# 导入 torchvision 中的模型模块
import torchvision.models as models


class Attention(nn.Module):
    """
    定义注意力机制模块，用于增强ResNet-18的性能
    """

    def __init__(self, in_channels):
        super(Attention, self).__init__()
        # 三个卷积层用于生成注意力权重
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播函数

        Args:
        - x (tensor): 输入张量，尺寸为(batch_size, channels, height, width)

        Returns:
        - tensor: 输出张量，增强后的特征表示，尺寸与输入相同
        """
        batch_size, channels, height, width = x.size()
        f = self.conv1(x).view(batch_size, -1, height * width)
        g = self.conv2(x).view(batch_size, -1, height * width)
        h = self.conv3(x).view(batch_size, -1, height * width)
        attention = self.softmax(torch.bmm(f.permute(0, 2, 1), g))
        o = torch.bmm(h, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        return o + x


class ResNet18WithAttention(nn.Module):
    """
    带注意力机制的ResNet-18模型，用于分类任务
    """

    def __init__(self, num_classes):
        super(ResNet18WithAttention, self).__init__()
        # 加载预训练的ResNet-18模型
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 替换最后的全连接层
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
        # 添加注意力机制
        self.attention = Attention(self.resnet.layer4[1].conv2.out_channels)

    def forward(self, x):
        """
        前向传播函数

        Args:
        - x (tensor): 输入张量，尺寸为(batch_size, channels, height, width)

        Returns:
        - tensor: 输出张量，经过ResNet-18和注意力机制后的预测结果，尺寸为(batch_size, num_classes)
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.attention(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


def get_net():
    """
    获取带有注意力机制的ResNet-18模型

    Returns:
    - ResNet18WithAttention: 带有注意力机制的ResNet-18模型实例
    """
    num_classes = 10
    net = ResNet18WithAttention(num_classes)
    return net


loss = nn.CrossEntropyLoss(reduction="none")


def get_optimizer_and_scheduler(net, lr, wd, lr_period, lr_decay):
    """
    获取优化器和学习率调度器

    Args:
    - net (nn.Module): 神经网络模型
    - lr (float): 学习率
    - wd (float): 权重衰减
    - lr_period (int): 学习率衰减周期
    - lr_decay (float): 学习率衰减率

    Returns:
    - optimizer: 优化器
    - scheduler: 学习率调度器
    """
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    return optimizer, scheduler


# 导入 torchvision 中的模型模块
import torchvision.models as models


class Attention(nn.Module):
    """
    定义注意力机制模块，用于增强ResNet-18的性能
    """

    def __init__(self, in_channels):
        super(Attention, self).__init__()
        # 三个卷积层用于生成注意力权重
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播函数

        Args:
        - x (tensor): 输入张量，尺寸为(batch_size, channels, height, width)

        Returns:
        - tensor: 输出张量，增强后的特征表示，尺寸与输入相同
        """
        batch_size, channels, height, width = x.size()
        f = self.conv1(x).view(batch_size, -1, height * width)
        g = self.conv2(x).view(batch_size, -1, height * width)
        h = self.conv3(x).view(batch_size, -1, height * width)
        attention = self.softmax(torch.bmm(f.permute(0, 2, 1), g))
        o = torch.bmm(h, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        return o + x


class ResNet18WithAttention(nn.Module):
    """
    带注意力机制的ResNet-18模型，用于分类任务
    """

    def __init__(self, num_classes):
        super(ResNet18WithAttention, self).__init__()
        # 加载预训练的ResNet-18模型
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 替换最后的全连接层
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
        # 添加注意力机制
        self.attention = Attention(self.resnet.layer4[1].conv2.out_channels)

    def forward(self, x):
        """
        前向传播函数

        Args:
        - x (tensor): 输入张量，尺寸为(batch_size, channels, height, width)

        Returns:
        - tensor: 输出张量，经过ResNet-18和注意力机制后的预测结果，尺寸为(batch_size, num_classes)
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.attention(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


def get_net():
    """
    获取带有注意力机制的ResNet-18模型

    Returns:
    - ResNet18WithAttention: 带有注意力机制的ResNet-18模型实例
    """
    num_classes = 10
    net = ResNet18WithAttention(num_classes)
    return net


loss = nn.CrossEntropyLoss(reduction="none")


def get_optimizer_and_scheduler(net, lr, wd, lr_period, lr_decay):
    """
    获取优化器和学习率调度器

    Args:
    - net (nn.Module): 神经网络模型
    - lr (float): 学习率
    - wd (float): 权重衰减
    - lr_period (int): 学习率衰减周期
    - lr_decay (float): 学习率衰减率

    Returns:
    - optimizer: 优化器
    - scheduler: 学习率调度器
    """
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    return optimizer, scheduler


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 获取优化器和学习率调度器
    optimizer, scheduler = get_optimizer_and_scheduler(net, lr, wd, lr_period, lr_decay)

    # 获取训练数据集批次数量和初始化计时器
    num_batches, timer = len(train_iter), d2l.Timer()

    # 初始化图表显示的标签
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')

    # 初始化动画器，用于实时显示训练过程
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)

    # 将模型并行化，使用所有可用的 GPU 设备
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):
        # 设置模型为训练模式
        net.train()

        # 初始化累加器，用于累加每个批次的损失和准确率
        metric = d2l.Accumulator(3)

        for i, (features, labels) in enumerate(train_iter):
            timer.start()

            # 训练一个批次，计算损失和准确率
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, optimizer, devices)

            # 累加损失和准确率
            metric.add(l, acc, labels.shape[0])

            timer.stop()

            # 每处理完 num_batches // 5 个批次或者最后一个批次时，更新动画图
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))

        # 如果提供了验证数据集，在每个 epoch 结束时计算验证准确率
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))

        # 更新学习率
        scheduler.step()

    # 打印训练损失和准确率
    measures = (f'train loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[2]:.3f}')

    # 如果有验证数据集，也打印验证准确率
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'

    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

    # 显示训练过程的图表
    d2l.plt.show()

# devices, num_epochs, lr, wd = d2l.try_all_gpus(), 100, 0.1e-4, 5e-4
# lr_period, lr_decay, net = 50, 0.1, get_net()

# 获取所有可用的 GPU，如果没有 GPU，则使用 CPU
devices = d2l.try_all_gpus()
if len(devices) == 0:
    devices = ['cpu']
print(f"Using devices: {devices}")

# 设置训练参数
# num_epochs, lr, wd = 50, 2e-4, 5e-4
# lr_period, lr_decay = 4, 0.9
num_epochs, lr, wd = 100, 1e-4, 5e-4
lr_period, lr_decay = 50, 0.1
net = get_net()

# 训练模型
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 定义保存模型的函数
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 示例路径和文件名
model_path = '../sagemaker-studiolab-notebooks/trained_model.pth'

# 保存模型
save_model(net, model_path)

import pandas as pd

# 加载训练好的模型和初始化预测结果列表
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 对测试集进行预测
for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

# 创建排序后的ID列表
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))

# 将预测结果整理成DataFrame，并保存为CSV文件
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
