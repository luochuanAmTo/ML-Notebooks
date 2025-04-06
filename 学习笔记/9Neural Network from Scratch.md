```py
import torch
import torch.nn as nn

X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor
  #创建一个 3x2 的张量 X，表示输入数据。每一行是一个样本，每一列是一个特征。数据类型为 torch.float（浮点数）。
y = torch.tensor(([92], [100], [89]), dtype=torch.float) # 3 X 1 tensor
  #创建一个 3x1 的张量 y，表示目标值（标签）。每个值对应 X 中相应样本的目标输出。数据类型为 torch.float。
xPredicted = torch.tensor(([4, 8]), dtype=torch.float) # 1 X 2 tensor
  #创建一个 1x2 的张量 xPredicted，表示一个待预测的样本。数据类型为 torch.float。 a


print(X.size())
print(y.size())  

# scale units
X_max, _ = torch.max(X, 0) #计算 X 中每一列的最大值。torch.max(X, 0) 返回一个元组，第一个元素是每一列的最大值，第二个元素是对应的索引（这里用 _ 忽略）。
xPredicted_max, _ = torch.max(xPredicted, 0)

X = torch.div(X, X_max) #对 X 进行归一化处理，将每个元素除以其所在列的最大值。这样做的目的是将数据缩放到 [0, 1] 范围内，以便于神经网络训练
xPredicted = torch.div(xPredicted, xPredicted_ma x)
y = y / 100  # max test score is 100 对 y 进行归一化处理，将每个目标值除以 100（假设最大目标值是 100），将其缩放到 [0, 1] 范围内。
print(xPredicted)    #归一化后的 xPredicted 张量
```



```py 
class Neural_Network(nn.Module): #定义一个神经网络类 Neural_Network，继承自 PyTorch 的 nn.Module 类 
    def __init__(self, ): 
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 2 #输入层的大小（输入特征数为 2）
        self.outputSize = 1  #：输出层的大小（输出特征数为 1）
        self.hiddenSize = 3  #隐藏层的大小（隐藏层神经元数为 3
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 3 X 2 tensor    输入层到隐藏层的权重矩阵，大小为 (inputSize, hiddenSize)，即 (2, 3)。
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor    隐藏层到输出层的权重矩阵，大小为 (hiddenSize, outputSize)，即 (3, 1)。
        
    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorchnmamul是矩阵乘法   隐藏层到输出层的权重矩阵，大小为 (hiddenSize, outputSize)，即 (3, 1)。 X 的形状为 (batch_size, inputSize)，W1 的形状为 (inputSize, hiddenSize)，因此 z 的形状为 (batch_size, hiddenSize)。

        self.z2 = self.sigmoid(self.z) # activation function 对隐藏层的输入 z 应用激活函数 sigmoid，得到隐藏层的输出 z2
        self.z3 = torch.matmul(self.z2, self.W2)  #计算隐藏层输出 z2 与权重 W2 的矩阵乘法，得到输出层的输入 z3
        o = self.sigmoid(self.z3) # final activation function
        return o #计算隐藏层输出 z2 与权重 W2 的矩阵乘法，得到输出层的输入 z3   对输出层的输入 z3 应用激活函数 sigmoid，得到网络的最终输出 o，并返回。 
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid #用于反向传播中的梯度计算
        return s * (1 - s)
    #s 是 Sigmoid 函数的输出，即 s=σ(x),返回值是 Sigmoid 函数的导数，即 s⋅(1−s)
    
    def backward(self, X, y, o):
        self.o_error = y - o # error in output 计算输出层的误差 o_error，即目标值 y 与网络输出 o 的差值。
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error 计算输出层的误差梯度 o_delta，即输出误差乘以 sigmoid 的导数
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2)) #计算隐藏层的误差 z2_error，即输出层误差梯度 o_delta 与权重 W2 的转置的矩阵乘法
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) #计算隐藏层的误差梯度 z2_delta，即隐藏层误差乘以 sigmoid 的导数。
        self.W1 += torch.matmul(torch.t(X), self.z2_delta) #输入 X 的转置与隐藏层误差梯度 z2_delta 的矩阵乘法
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        #隐藏层输出 z2 的转置与输出层误差梯度 o_delta 的矩阵乘法。
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X) #进行前向传播，得到输出 o 
        self.backward(X, y, o) #进行反向传播，更新权重
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN") #定义保存权重的函数 saveWeights，使用 torch.save 将模型保存到文件 "NN" 中。
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))
        
```







