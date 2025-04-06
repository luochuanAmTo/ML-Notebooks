```python
# imports
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

# use gpu if available a
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```



```python
# download the data
!wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
!unzip hymenoptera_data.zip #下载并解压一个图像分类任务的数据集（hymenoptera_data）
```



```python
data_dir = 'hymenoptera_data'

# custom transformer to flatten the image tensors
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        result = torch.reshape(img, self.new_size)
        return result

# transformations used to standardize and normalize the datasets定义图像预处理流程
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224), #将图像缩放为 224x224 像素。
        transforms.CenterCrop(224), #从图像中心裁剪出 224x224 像素的区域。
        transforms.ToTensor(), #将图像转换为 PyTorch 张量（形状为 [C, H, W]，其中 C 是通道数，H 和 W 是高度和宽度）。
        ReshapeTransform((-1,)) # flattens the data 将图像张量展平为一维向量。
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ReshapeTransform((-1,)) # flattens the data
    ]),
}

# load the correspoding folders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# load the entire dataset; we are not using minibatches here
train_dataset = torch.utils.data.DataLoader(image_datasets['train'],
                                            batch_size=len(image_datasets['train']),
                                            shuffle=True)

test_dataset = torch.utils.data.DataLoader(image_datasets['val'],
                                           batch_size=len(image_datasets['val']),
                                           shuffle=True)解释代码作用
```

```python
# build the LR model
class LR(nn.Module):
    def __init__(self, dim):
        super(LR, self).__init__()
        self.linear = nn.Linear(dim, 1) #定义一个全连接层（线性层），输入维度为 dim，输出维度为 1。
        nn.init.zeros_(self.linear.weight) #将线性层的权重矩阵初始化为全零
        nn.init.zeros_(self.linear.bias) #将线性层的偏置项初始化为零

    def forward(self, x):
        x = self.linear(x) #将输入数据通过线性层，计算线性变换：y = x @ weight + bias。输出形状为 [batch_size, 1]。
        x = torch.sigmoid(x) #对线性层的输出应用 Sigmoid 函数，将值映射到 [0, 1] 之间。
        return x 
```

```py
# predict function
def predict(yhat, y):#hat = torch.tensor([0.2, 0.6, 0.4, 0.8])（模型预测的概率值）。  y = torch.tensor([0, 1, 0, 1])（真实标签）
    yhat = yhat.squeeze()
    y = y.unsqueeze(0) 
    y_prediction = torch.zeros(y.size()[1]) #创建一个全零张量 y_prediction，形状为 [batch_size]，用于存储每个样本的预测类别
    for i in range(yhat.shape[0]):
        if yhat[i] <= 0.5:
            y_prediction[i] = 0
        else:
            y_prediction[i] = 1
    return 100 - torch.mean(torch.abs(y_prediction - y)) * 100  #取差值的绝对值，得到每个样本的预测是否正确（0 表示正确，1 表示错误）
```



```py
dim = train_dataset.dataset[0][0].shape[0] #例如，如果图像被展平为一维向量，dim 的值可能是 224 * 224 * 3 = 150528（假设图像大小为 224x224，3 个通道）。

lrmodel = LR(dim).to(device) #将模型移动到指定的设备（GPU 或 CPU）。
criterion = nn.BCELoss() #使用二元交叉熵损失（Binary Cross Entropy Loss）
optimizer = torch.optim.SGD(lrmodel.parameters(), lr=0.0001) #使用随机梯度下降（Stochastic Gradient Descent, SGD）优化器。用于更新模型的参数，以最小化损失函数。
```



```py
# training the model
costs = []

for ITER in range(200):
    lrmodel.train()
    x, y = next(iter(train_dataset))
    test_x, test_y = next(iter(test_dataset))

    # forward
    yhat = lrmodel.forward(x.to(device))

    cost = criterion(yhat.squeeze(), y.type(torch.FloatTensor))
    #criterion:计算二元交叉熵损失（BCELoss），衡量模型预测值与真实标签之间的差异
    train_pred = predict(yhat, y) #调用 predict 函数，计算模型在训练数据上的准确率。

    # backward
    optimizer.zero_grad()
    cost.backward() 计算损失函数关于模型参数的梯度。
    optimizer.step() 根据梯度更新模型参数。
    
    # evaluate
    lrmodel.eval()
    with torch.no_grad():
        yhat_test = lrmodel.forward(test_x.to(device))
        test_pred = predict(yhat_test, test_y)

    if ITER % 10 == 0:
        costs.append(cost)

    if ITER % 10 == 0:
        print("Cost after iteration {}: {} | Train Acc: {} | Test Acc: {}".format(ITER, 
                                                                                    cost, 
                                                                                    train_pred,
                                                                                    test_pred))
   
```

