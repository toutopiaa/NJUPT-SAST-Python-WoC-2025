# 方向选择

请选择提交的方向 (可多选):

* \[√ ] 学术向
* \[ ] 工具向
* \[ ] 机器学习向
* \[ ] 开发向
* \[ ] 爬虫向

目前进度:

* \[ ] 计划中
* \[ ] 进行中
* \[√ ] 已完成







import os

import torch

from torch.utils.data import Dataset, DataLoader

from torch import nn, optim

import numpy as np

import h5py

import glob

import re



class DnCNN(nn.Module):

    def \_\_init\_\_(self, channels=1, num\_of\_layers=17):

        super(DnCNN, self).\_\_init\_\_()

        kernel\_size = 3

        padding = 1

        features = 64

        layers = \[]

        layers.append(nn.Conv2d(channels, features, kernel\_size, padding=padding, bias=False))

        layers.append(nn.ReLU(inplace=True))

        for \_ in range(num\_of\_layers - 2):

            layers.append(nn.Conv2d(features, features, kernel\_size, padding=padding, bias=False))

            layers.append(nn.BatchNorm2d(features))

            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, channels, kernel\_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(\*layers)



    def forward(self, x):

        residual = self.dncnn(x)

        return x - residual











\# 内存预加载dataset

class SIDD\_Raw\_Dataset\_Preloaded(Dataset):

    def \_\_init\_\_(self, root\_dir, patch\_size=128):

        self.root\_dir = root\_dir

        self.patch\_size = patch\_size

        if not os.path.exists(root\_dir):

            raise FileNotFoundError(f"Data文件夹不存在！请检查路径：{root\_dir}")

 

        #遍历场景文件夹

        scene\_instances\_path = os.path.join(os.path.dirname(root\_dir), "Scene\_Instances.txt")

        if not os.path.exists(scene\_instances\_path):

            raise FileNotFoundError(f"Scene\_Instances.txt不存在！请检查是否和Data文件夹在同一级目录")

        scene\_dirs = \[]



 	for line in open(scene\_instances\_path, 'r'):

 	    cleaned\_line = line.strip()

 	    if cleaned\_line:

 	        scene\_dirs.append(cleaned\_line)

        valid\_scene\_dirs = \[]

 	for d in scene\_dirs:

    	scene\_full\_path = os.path.join(root\_dir, d)

    	if os.path.isdir(scene\_full\_path):

        	valid\_scene\_dirs.append(d)

 	scene\_dirs = valid\_scene\_dirs

        print(f"成功匹配Scene\_Instances.txt的有效场景文件夹总数：{len(scene\_dirs)} 个")



        self.noisy\_paths = \[]

        self.clean\_paths = \[]

 

        #配对图像

        for scene in scene\_dirs:

            scene\_path = os.path.join(root\_dir, scene)

            all\_noisy\_files = sorted(glob.glob(os.path.join(scene\_path, "\*NOISY\_RAW\*.MAT")))

            all\_gt\_files = sorted(glob.glob(os.path.join(scene\_path, "\*GT\_RAW\*.MAT")))

 

            if len(all\_noisy\_files) == 2 and len(all\_gt\_files) == 2:

                self.noisy\_paths.extend(all\_noisy\_files)

                self.clean\_paths.extend(all\_gt\_files)

            else:

                print(f"场景 {scene} 下文件数量不匹配，跳过")

 



        print(f"成功配对的噪声-真值图像对总数：{len(self.noisy\_paths)} 对")



 

        #预加载

        print("正在预加载所有数据到内存")

        self.noisy\_data\_list = \[]

        self.clean\_data\_list = \[]

        key = 'x'

 

        for idx, (noisy\_path, clean\_path) in enumerate(zip(self.noisy\_paths, self.clean\_paths)):

            try:

                with h5py.File(noisy\_path, 'r') as f:

                    noisy\_img = np.array(f\[key], dtype=np.float32)

                with h5py.File(clean\_path, 'r') as f:

                    clean\_img = np.array(f\[key], dtype=np.float32)

 

                #转置维度

                noisy\_img = np.transpose(noisy\_img)

                clean\_img = np.transpose(clean\_img)

 

                self.noisy\_data\_list.append(noisy\_img)

                self.clean\_data\_list.append(clean\_img)

 

                if (idx + 1) % 20 == 0:

                    print(f"已预加载 {idx+1}/{len(self.noisy\_paths)} 对图像")

            except Exception as e:

                print(f"预加载文件失败 {noisy\_path}，错误：{e}")

                continue

 

        print(f"数据预加载完成！成功加载 {len(self.noisy\_data\_list)} 对图像")











    def \_\_len\_\_(self):

        return len(self.noisy\_data\_list)



    def \_\_getitem\_\_(self, idx):

        # 直接从内存读取数据

        noisy\_img = self.noisy\_data\_list\[idx]

        clean\_img = self.clean\_data\_list\[idx]



————————————————————————————————————————————————————————————————————————————————————————————————————————————————好像没有正确填充

 	#先填充再裁剪

        h, w = noisy\_img.shape

        patch\_size = self.patch\_size

 

        if h < patch\_size or w < patch\_size:

            pad\_h = max(patch\_size - h, 0)

            pad\_w = max(patch\_size - w, 0)

            noisy\_img = np.pad(noisy\_img, ((0, pad\_h), (0, pad\_w)), mode='constant')

            clean\_img = np.pad(clean\_img, ((0, pad\_h), (0, pad\_w)), mode='constant')

            h, w = noisy\_img.shape

 

        #生成随机裁剪的起始坐标

        start\_h = np.random.randint(0, h - patch\_size + 1)

        start\_w = np.random.randint(0, w - patch\_size + 1)

 

        #同步裁剪

        noisy\_patch = noisy\_img\[start\_h:start\_h+patch\_size, start\_w:start\_w+patch\_size]

        clean\_patch = clean\_img\[start\_h:start\_h+patch\_size, start\_w:start\_w+patch\_size]

 

        #适配PyTorch的CHW张量格式

        noisy\_patch = np.expand\_dims(noisy\_patch, axis=0)

        clean\_patch = np.expand\_dims(clean\_patch, axis=0)

 

        #转为PyTorch浮点张量

        noisy\_tensor = torch.from\_numpy(noisy\_patch)

        clean\_tensor = torch.from\_numpy(clean\_patch)

 

        return noisy\_tensor, clean\_tensor













\#数据加载器

def get\_sidd(data\_dir, batch\_size=16, patch\_size=128):

    dataset = SIDD\_Raw\_Dataset\_Preloaded(data\_dir, patch\_size=patch\_size)

    # Windows系统强制num\_workers=0

    return DataLoader(

        dataset,

        batch\_size=batch\_size,

        shuffle=True,

        num\_workers=0,  #Windows下不要改大于0的数

        pin\_memory=True,	#内存锁定

        drop\_last=False

    )









\#主函数

def main():

    torch.manual\_seed(168)

    np.random.seed(168)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

 

    device = torch.device('cuda:0')

    torch.cuda.set\_device(device)

 

    data\_dir = r'D:\\SIDD\_Medium\_Raw\\Data'



    batch\_size = 16

    patch\_size = 128

    epochs = 60



    lr = 1e-3

    step\_size = 5

    gamma = 0.5



    # 加载数据集

    try:

        train\_loader = get\_sidd(data\_dir, batch\_size=batch\_size, patch\_size=patch\_size)

        print(f"数据集总迭代批次：{len(train\_loader)}")

    except Exception as e:

        print(f"数据加载失败：{e}")

        return

 

    # 初始化模型、优化器、损失函数

    model = DnCNN().to(device)

    print(f"模型已加载到GPU：{next(model.parameters()).device}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss().to(device)

    scheduler = optim.lr\_scheduler.StepLR(optimizer, step\_size=step\_size, gamma=gamma)

 

    #训练循环

    print("\\n==================== 开始训练 ====================")

    model.train()

    for epoch in range(epochs):

        total\_loss = 0.0

        # 本轮GPU计时

        if torch.cuda.is\_available():

            start\_time = torch.cuda.Event(enable\_timing=True)

            end\_time = torch.cuda.Event(enable\_timing=True)

            start\_time.record()

 

        for batch\_idx, (noisy\_imgs, clean\_imgs) in enumerate(train\_loader):

            # 数据迁移到GPU，开启异步拷贝

            noisy\_imgs = noisy\_imgs.to(device, non\_blocking=True)

            clean\_imgs = clean\_imgs.to(device, non\_blocking=True)

 

            #前向传播

            pred\_clean = model(noisy\_imgs)

            loss = criterion(pred\_clean, clean\_imgs)

 

            #反向传播与参数更新

            optimizer.zero\_grad()

            loss.backward()

            optimizer.step()

 

            total\_loss += loss.item()

 

            if batch\_idx % 5 == 0:

                print(f"Epoch \[{epoch+1}/{epochs}] | Batch \[{batch\_idx}/{len(train\_loader)}] | Loss: {loss.item():.6f}")

 

        epoch\_time = 0.0

        end\_time.record()

        torch.cuda.synchronize()

        epoch\_time = start\_time.elapsed\_time(end\_time) / 1000

 

        #学习率更新

        scheduler.step()

        #打印每轮结果

        avg\_loss = total\_loss / len(train\_loader)

        current\_lr = optimizer.param\_groups\[0]\['lr']

        print(f"\\nEpoch \[{epoch+1}/{epochs}] 完成 | 耗时：{epoch\_time:.2f}秒 | 平均Loss: {avg\_loss:.6f} | 当前LR: {current\_lr:.6f}\\n")

 

    #保存模型

    torch.save(model.state\_dict(), "dncnn\_sidd\_raw\_final.pth")

    print("==================== 训练全部完成！模型已保存为 dncnn\_sidd\_raw\_final.pth ====================")



if \_\_name\_\_ == "\_\_main\_\_":

    main()

