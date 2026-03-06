# 方向选择

请选择提交的方向 (可多选):

* \[ √] 学术向
* \[ ] 工具向
* \[ ] 机器学习向
* \[ ] 开发向
* \[ ] 爬虫向

目前进度:

* \[ ] 计划中
* \[ ] 进行中
* \[√ ] 已完成



架构：数据集封装→数据预处理→模型定义→训练 / 测试







源码



import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader





class CIFAR10Preloaded(Dataset):

    def \_\_init\_\_(self, root, train=True, transform=None, download=False):

        self.original\_dataset = torchvision.datasets.CIFAR10(

            root=root, train=train, transform=transform, download=download

        )

 

        self.images = \[]

        self.labels = \[]

 



        for img, label in self.original\_dataset:

            self.images.append(img)

            self.labels.append(label)

 

 

        self.images = torch.stack(self.images)

        self.labels = torch.tensor(self.labels)



    def \_\_len\_\_(self):

        return len(self.images)



    def \_\_getitem\_\_(self, idx):

        return self.images\[idx], self.labels\[idx]



\# 定义图像预处理

transform = transforms.Compose(\[

    transforms.ToTensor(),

    transforms.Normalize(

        mean=(0.4914, 0.4822, 0.4465),

        std=(0.2023, 0.1994, 0.2010)

    )

])



\# 加载预加载的训练集

trainset = CIFAR10Preloaded(

    root=r'C:\\Users\\Lenovo\\python\_study\\py311\\day3\\data\\cifar-10-batches-py', train=True, transform=transform, download=False

)

trainloader = DataLoader(

    trainset, batch\_size=64, shuffle=True,

    num\_workers=0

)



\# 加载预加载的测试集

testset = CIFAR10Preloaded(

    root=r'C:\\Users\\Lenovo\\python\_study\\py311\\day3\\data\\cifar-10-batches-py', train=False, transform=transform, download=False

)

testloader = DataLoader(

    testset, batch\_size=64, shuffle=False,

    num\_workers=0

)





\#定义简单的CNN模型



class SimpleCNN(nn.Module):

    def \_\_init\_\_(self):

        super(SimpleCNN, self).\_\_init\_\_()

        # 卷积层1: 3通道->32通道, 3x3卷积

        self.conv1 = nn.Conv2d(3, 32, kernel\_size=3, padding=1)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2, 2)

 

        # 卷积层2: 32通道->64通道

        self.conv2 = nn.Conv2d(32, 64, kernel\_size=3, padding=1)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(2, 2)

 

        # 全连接层

        self.fc1 = nn.Linear(64 \* 8 \* 8, 512)

        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(512, 10)	#10个分类



    def forward(self, x):

        x = self.pool1(self.relu1(self.conv1(x)))

        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(-1, 64 \* 8 \* 8)			#展平为一维向量

        x = self.relu3(self.fc1(x))

        x = self.fc2(x)

        return x





\#训练与测试



device = torch.device("cuda")



model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()  # 分类任务用交叉熵损失

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器



\# 训练函数

def train\_model(epochs=10):

    model.train()

    for epoch in range(epochs):

        running\_loss = 0.0

        for inputs, labels in trainloader:

            inputs, labels = inputs.to(device), labels.to(device)

 

            # 1. 梯度清零

            optimizer.zero\_grad()

            # 2. 前向传播

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # 3. 反向传播

            loss.backward()

            # 4. 更新参数

            optimizer.step()

 

            running\_loss += loss.item() \* inputs.size(0)

 

        epoch\_loss = running\_loss / len(trainset)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch\_loss:.4f}")



\# 测试函数

def test\_model():

    model.eval()

    correct = 0

    total = 0

    with torch.no\_grad():  # 测试时不需要计算梯度

        for inputs, labels in testloader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            \_, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 \* correct / total:.2f}%")



\# 执行训练和测试

if \_\_name\_\_ == "\_\_main\_\_":

    print("开始训练...")

    train\_model(epochs=10)

    print("训练完成，开始测试...")

    test\_model()

