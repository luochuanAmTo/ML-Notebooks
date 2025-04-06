```python
# Find the CUDA version PyTorch was installed with
!python -c "import torch; print(torch.version.cuda)" #查找 PyTorch 安装时使用的 CUDA 版本

# PyTorch version
!python -c "import torch; print(torch.__version__)" #获取 PyTorch 版本

!pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html #torch-scatter：一个用于处理图数据的 PyTorch 扩展库，通常用于图神经网络（GNN）。

!pip install torch-geometric #安装 torch-geometric 库
```



```py
import torch

# print torch version
print(torch.__version__)

from torch_geometric.data import Data

# define edge list
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

'''第一行 [0, 1, 1, 2] 表示边的起始节点。

边 1 的起始节点是 0
边 2 的起始节点是 1
边 3 的起始节点是 1
边 4 的起始节点是 2
第二行 [1, 0, 2, 1] 表示边的结束节点。

边 1 的结束节点是 1
边 2 的结束节点是 0
边 3 的结束节点是 2
边 4 的结束节点是 1
'''
# define node features
x = torch.tensor([[-1], [0], [1]])#这个 x 张量的形状是 (3, 1)，表示图中有 3 个节点，每个节点有 1 个特征

# create graph data object
data = Data(x=x, edge_index=edge_index)
print(data)
#Data(x=[3, 1], edge_index=[2, 4])
#这表示：图有 3 个节点，每个节点有 1 个特征（形状为 (3, 1)）。图有 4 条边，边的信息通过 edge_index 给出，形状为 (2, 4)。

# check number of edges of the graph
print(data.num_edges)
# check number of nodes of the graph
print(data.num_nodes)

# check if graph is directed
print(data.is_directed())

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='tmp/Cora', name='Cora') 

'''
节点（Nodes）：数据集包含 2708 个节点，每个节点代表一篇论文。每篇论文都有一个对应的特征向量，通常是论文的词袋模型（Bag-of-Words, BoW）表示。

特征（Features）：每个节点（论文）都有一个 1433 维的特征向量，这些特征是由论文的词袋模型（BoW）生成的。每个特征表示词汇表中的一个单词，特征的值表示该单词在论文中的出现频率。每篇论文的特征向量是稀疏的，大多数元素为零。

边（Edges）：数据集包含 5429 条边，每条边表示论文之间的引用关系。具体来说，如果论文 A 引用了论文 B，则在图中存在一条从 A 到 B 的有向边。由于是学术论文网络，所以边是有向的（虽然在实际应用中有时会考虑无向边）。

标签（Labels）：数据集中的每个节点（论文）都有一个标签，表示该论文所属的主题（类别）。Cora 数据集包含 7 个类别，分别是：


'''

# number of graphs
print("Number of graphs: ", len(dataset))

# number of features
print("Number of features: ", dataset.num_features)

# number of classes
print("Number of classes: ", dataset.num_classes)

# select the first graph
data = dataset[0]  #选择数据集中的第一个图（Cora 数据集只有一个图）。

# number of nodes
print("Number of nodes: ", data.num_nodes)

# number of edges
print("Number of edges: ", data.num_edges)

# check if directed
print("Is directed: ", data.is_directed())

# sample nodes from the graph
print("Shape of sample nodes: ", data.x[:5].shape)
  """Shape of sample nodes:  torch.Size([5, 1433])"""

# check training nodes
print("# of nodes to train on: ", data.train_mask.sum().item())

# check test nodes
print("# of nodes to test on: ", data.test_mask.sum().item())

# check validation nodes
print("# of nodes to validate on: ", data.val_mask.sum().item())



"""# of nodes to train on:  140
# of nodes to test on:  1000
# of nodes to validate on:  500"""


from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset = Planetoid(root='tmp/Cora', name='Cora')
data = dataset[0].to(device)
print("X shape: ", data.x.shape)
print("Edge shape: ", data.edge_index.shape)
print("Y shape: ", data.y.shape)

"""X shape:  torch.Size([2708, 1433])    2708 个节点，每个节点代表一篇论文, 每个节点（论文）都有一个 1433 维的特征向量，这些特征是由论文的词袋模型（BoW）生成的。每个特征表示词汇表中的一个单词，特征的值表示该单词在论文中的出现频率
Edge shape:  torch.Size([2, 10556])   数据集包含 5429 条边，每条边表示论文之间的引用关系
Y shape:  torch.Size([2708])"""  
```



```
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ GCNConv layers """
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```

