# 方向选择

请选择提交的方向 (可多选):

* \[√ ] 学术向
* \[ ] 工具向
* \[ ] 机器学习向
* \[ ] 开发向
* \[ ] 爬虫向

目前进度:

* \[ ] 计划中
* \[√ ] 进行中
* \[ ] 已完成







import os

import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import numpy as np

import h5py

import glob

import torch.nn.functional as F

import random



\# ======================================

\#1. 全链路随机数固定

\# ======================================

def set\_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual\_seed(seed)

    if torch.cuda.is\_available():

        torch.cuda.manual\_seed(seed)

        torch.cuda.manual\_seed\_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print(f"随机种子已固定为: {seed}")



set\_seed(168)



\# ======================================

\#2. 数据集类（100%复用你提供的原代码，仅修复负步长bug）

\# ======================================

\# ---------------------------

\#分类任务：CIFAR10预加载数据集（原task2代码完全复用）

\# ---------------------------

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

        self.images = torch.stack(self.images).float()

        self.labels = torch.as\_tensor(self.labels, dtype=torch.long)

    def \_\_len\_\_(self):

        return len(self.images)

    def \_\_getitem\_\_(self, idx):

        return self.images\[idx], self.labels\[idx]



\# ---------------------------

\#去噪任务：SIDD RAW数据集（原task1代码完全复用，仅修复负步长bug）

\# ---------------------------

class SIDD\_Raw\_Dataset\_Preloaded(Dataset):

    def \_\_init\_\_(self, root\_dir, patch\_size=128):

        self.root\_dir = root\_dir

        self.patch\_size = patch\_size

        if not os.path.exists(root\_dir):

            raise FileNotFoundError(f"Data文件夹不存在！请检查路径：{root\_dir}")



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

        noisy\_img = self.noisy\_data\_list\[idx]

        clean\_img = self.clean\_data\_list\[idx]

        h, w = noisy\_img.shape

        patch\_size = self.patch\_size



        if h < patch\_size or w < patch\_size:

            pad\_h = max(patch\_size - h, 0)

            pad\_w = max(patch\_size - w, 0)

            noisy\_img = np.pad(noisy\_img, ((0, pad\_h), (0, pad\_w)), mode='constant')

            clean\_img = np.pad(clean\_img, ((0, pad\_h), (0, pad\_w)), mode='constant')

            h, w = noisy\_img.shape



        start\_h = np.random.randint(0, h - patch\_size + 1)

        start\_w = np.random.randint(0, w - patch\_size + 1)



        noisy\_patch = noisy\_img\[start\_h:start\_h+patch\_size, start\_w:start\_w+patch\_size]

        clean\_patch = clean\_img\[start\_h:start\_h+patch\_size, start\_w:start\_w+patch\_size]



        #修复负步长bug：翻转/旋转后加.copy()

        if np.random.rand() > 0.5:

            noisy\_patch = np.fliplr(noisy\_patch).copy()

            clean\_patch = np.fliplr(clean\_patch).copy()

        if np.random.rand() > 0.5:

            noisy\_patch = np.flipud(noisy\_patch).copy()

            clean\_patch = np.flipud(clean\_patch).copy()

        rot\_k = np.random.randint(0, 4)

        noisy\_patch = np.rot90(noisy\_patch, k=rot\_k).copy()

        clean\_patch = np.rot90(clean\_patch, k=rot\_k).copy()



        noisy\_patch = np.expand\_dims(noisy\_patch, axis=0).copy()

        clean\_patch = np.expand\_dims(clean\_patch, axis=0).copy()



        noisy\_tensor = torch.from\_numpy(noisy\_patch)

        clean\_tensor = torch.from\_numpy(clean\_patch)



        return noisy\_tensor, clean\_tensor



\# ======================================

\#3. 多任务模型（核心：参数共享+原结构复用）

\# ======================================

class MultiTask\_DnCNN\_CNN(nn.Module):

    def \_\_init\_\_(self, in\_channels=1, num\_classes=10, dncnn\_layers=17):

        super().\_\_init\_\_()

        self.in\_channels = in\_channels

        features = 64

        kernel\_size = 3

        padding = 1



        # ======================================

        #共享编码器：100%复用原DnCNN前3层，无任何修改

        #作用：提取通用底层特征（边缘、纹理），两个任务共用

        # ======================================

        self.shared\_encoder = nn.Sequential(

            # 原DnCNN第1层

            nn.Conv2d(in\_channels, features, kernel\_size, padding=padding, bias=False),

            nn.ReLU(inplace=True),

            # 原DnCNN第2层

            nn.Conv2d(features, features, kernel\_size, padding=padding, bias=False),

            nn.BatchNorm2d(features),

            nn.ReLU(inplace=True),

            # 原DnCNN第3层

            nn.Conv2d(features, features, kernel\_size, padding=padding, bias=False),

            nn.BatchNorm2d(features),

            nn.ReLU(inplace=True),

        )



        # ======================================

        #去噪分支：100%复用原DnCNN剩余层，和原单任务结构完全一致

        # ======================================

        self.denoise\_branch = nn.Sequential()

        #原DnCNN剩余的中间层（17-3-1=13层）

        for i in range(dncnn\_layers - 3 - 1):

            self.denoise\_branch.add\_module(

                f'den\_conv\_{i+1}',

                nn.Conv2d(features, features, kernel\_size, padding=padding, bias=False)

            )

            self.denoise\_branch.add\_module(f'den\_bn\_{i+1}', nn.BatchNorm2d(features))

            self.denoise\_branch.add\_module(f'den\_relu\_{i+1}', nn.ReLU(inplace=True))

        #原DnCNN最后一层输出层

        self.denoise\_branch.add\_module(

            'den\_out\_conv',

            nn.Conv2d(features, in\_channels, kernel\_size, padding=padding, bias=False)

        )



        # ======================================

        #分类分支：95%复用原SimpleCNN结构，仅调整输入通道

        #原SimpleCNN结构：Conv→ReLU→Pool→Conv→ReLU→Pool→FC→FC

        # ======================================

        #轻量特征适配：1x1卷积调整通道，参数量极小，不改变原结构

        self.classify\_adapt = nn.Conv2d(features, 32, kernel\_size=1, padding=0, bias=False)

        # 完全复用原SimpleCNN的核心结构

        self.classify\_branch = nn.Sequential(

            # 原SimpleCNN第1组卷积+池化

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            # 原SimpleCNN第2组卷积+池化

            nn.Conv2d(32, 64, kernel\_size=3, padding=1, bias=False),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

        )

        #完全复用原SimpleCNN的全连接层

        self.classify\_fc = nn.Sequential(

            nn.Linear(64 \* 8 \* 8, 512),  # 原SimpleCNN完全一致

            nn.ReLU(inplace=True),

            nn.Linear(512, num\_classes)

        )



    # ---------------------------

    #去噪前向传播（和原DnCNN完全一致）

    # ---------------------------

    def forward\_denoise(self, x):

        # 共享编码器提取特征

        shared\_feat = self.shared\_encoder(x)

        # 去噪分支预测残差

        residual = self.denoise\_branch(shared\_feat)

        # 原DnCNN残差学习逻辑完全保留

        return x - residual



    # ---------------------------

    #分类前向传播（和原SimpleCNN逻辑完全一致）

    # ---------------------------

    def forward\_classify(self, x):

        # 3通道RGB转1通道（和去噪输入格式统一，适配共享编码器）

        if x.shape\[1] == 3:

            x = x.mean(dim=1, keepdim=True)

        # 共享编码器提取特征

        shared\_feat = self.shared\_encoder(x)

        # 特征适配+分类分支

        x = self.classify\_adapt(shared\_feat)

        x = self.classify\_branch(x)

        # 展平+全连接，和原SimpleCNN完全一致

        x = x.view(-1, 64 \* 8 \* 8)

        x = self.classify\_fc(x)

        return x



    # ---------------------------

    #联合前向传播（多任务训练用）

    # ---------------------------

    def forward(self, x, task\_type="denoise"):

        if task\_type == "denoise":

            return self.forward\_denoise(x)

        elif task\_type == "classify":

            return self.forward\_classify(x)

        elif task\_type == "both":

            # 同时输出两个任务结果，联合训练用

            denoise\_out = self.forward\_denoise(x)

            classify\_out = self.forward\_classify(x)

            return denoise\_out, classify\_out

        else:

            raise ValueError("task\_type仅支持'denoise'/'classify'/'both'")



\# ======================================

\#4. 指标计算函数

\# ======================================

def calculate\_psnr(img1, img2, data\_range=1.0):

    mse = torch.mean((img1 - img2) \*\* 2, dim=\[1, 2, 3])

    psnr = 10 \* torch.log10((data\_range \*\* 2) / (mse + 1e-8))

    return psnr.mean().item()



def calculate\_ssim(img1, img2, data\_range=1.0, window\_size=11, sigma=1.5):

    assert img1.shape\[1] == 1 and img2.shape\[1] == 1, "SSIM仅支持单通道图像"

    device = img1.device

    gauss = torch.Tensor(

        np.exp(-((np.arange(window\_size) - window\_size//2)\*\*2) / (2 \* sigma\*\*2))

    ).to(device)

    gauss = gauss / gauss.sum()

    window\_1d = gauss.unsqueeze(1)

    window\_2d = window\_1d @ window\_1d.T

    window = window\_2d.unsqueeze(0).unsqueeze(0)



    C1 = (0.01 \* data\_range) \*\* 2

    C2 = (0.03 \* data\_range) \*\* 2



    mu1 = F.conv2d(img1, window, padding=window\_size//2, groups=1)

    mu2 = F.conv2d(img2, window, padding=window\_size//2, groups=1)

    mu1\_sq = mu1 \*\* 2

    mu2\_sq = mu2 \*\* 2

    mu1\_mu2 = mu1 \* mu2



    sigma1\_sq = F.conv2d(img1\*img1, window, padding=window\_size//2, groups=1) - mu1\_sq

    sigma2\_sq = F.conv2d(img2\*img2, window, padding=window\_size//2, groups=1) - mu2\_sq

    sigma12 = F.conv2d(img1\*img2, window, padding=window\_size//2, groups=1) - mu1\_mu2



    ssim\_map = ((2 \* mu1\_mu2 + C1) \* (2 \* sigma12 + C2)) / ((mu1\_sq + mu2\_sq + C1) \* (sigma1\_sq + sigma2\_sq + C2))

    return ssim\_map.mean().item()



\# ======================================

\#5. 数据加载配置

\# ======================================

\# ---------------------------

\#分类数据：CIFAR10复用task2配置

\# ---------------------------

transform\_cifar = transforms.Compose(\[

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

])



\#本地路径

cifar\_root = r'C:\\Users\\Lenovo\\python\_study\\py311\\day3\\data\\cifar-10-batches-py'

sidd\_root = r'D:\\SIDD\_Medium\_Raw\\Data'



\#分类数据集加载

trainset\_cifar = CIFAR10Preloaded(root=cifar\_root, train=True, transform=transform\_cifar, download=False)

trainloader\_cifar = DataLoader(trainset\_cifar, batch\_size=64, shuffle=True, num\_workers=0)

testset\_cifar = CIFAR10Preloaded(root=cifar\_root, train=False, transform=transform\_cifar, download=False)

testloader\_cifar = DataLoader(testset\_cifar, batch\_size=64, shuffle=False, num\_workers=0)



\#去噪数据集加载

try:

    trainset\_sidd = SIDD\_Raw\_Dataset\_Preloaded(root\_dir=sidd\_root, patch\_size=128)

    trainloader\_sidd = DataLoader(trainset\_sidd, batch\_size=16, shuffle=True, num\_workers=0, pin\_memory=True)

except Exception as e:

    print(f"去噪数据集加载失败：{e}")

    trainloader\_sidd = None



\# ======================================

\#6. 训练与测试配置

\# ======================================

device = torch.device("cuda" if torch.cuda.is\_available() else "cpu")

print(f"训练设备：{device}")



\#模型初始化

model = MultiTask\_DnCNN\_CNN().to(device)



\#损失函数：完全复用原单任务的损失函数

criterion\_denoise = nn.MSELoss().to(device)  # 原task1用MSE

criterion\_classify = nn.CrossEntropyLoss().to(device)  # 原task2用交叉熵



\# ======================================

\#7. 核心训练策略：两阶段训练（解决偏科问题）

\# ======================================

\# ---------------------------

\#阶段1：去噪预训练（保住高PSNR）

\# ---------------------------

def train\_denoise\_pretrain(epochs=60):

    print("\\n===== 阶段1/2：去噪任务预训练 =====")

    if trainloader\_sidd is None:

        print("去噪数据集加载失败，跳过预训练")

        return

 

    #优化器复用原task1

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr\_scheduler.StepLR(optimizer, step\_size=5, gamma=0.5)

 

    best\_psnr = 0.0

    for epoch in range(epochs):

        model.train()

        total\_loss = 0.0

 

        for batch\_idx, (noisy\_imgs, clean\_imgs) in enumerate(trainloader\_sidd):

            noisy\_imgs = noisy\_imgs.to(device, non\_blocking=True)

            clean\_imgs = clean\_imgs.to(device, non\_blocking=True)

 

            optimizer.zero\_grad()

            pred\_clean = model.forward\_denoise(noisy\_imgs)

            loss = criterion\_denoise(pred\_clean, clean\_imgs)

            loss.backward()

            optimizer.step()

 

            total\_loss += loss.item()

            if batch\_idx % 10 == 0:

                print(f"Epoch \[{epoch+1}/{epochs}] | Batch \[{batch\_idx}/{len(trainloader\_sidd)}] | Loss: {loss.item():.6f}")

 

        scheduler.step()

        avg\_loss = total\_loss / len(trainloader\_sidd)

        current\_lr = optimizer.param\_groups\[0]\['lr']

 

        #验证PSNR

        model.eval()

        with torch.no\_grad():

            val\_psnr = 0.0

            for noisy\_imgs, clean\_imgs in trainloader\_sidd:

                noisy\_imgs = noisy\_imgs.to(device)

                clean\_imgs = clean\_imgs.to(device)

                denoised\_imgs = model.forward\_denoise(noisy\_imgs)

                val\_psnr += calculate\_psnr(denoised\_imgs, clean\_imgs)

            val\_psnr /= len(trainloader\_sidd)

 

        print(f"\\nEpoch \[{epoch+1}/{epochs}] 完成 | 平均Loss: {avg\_loss:.6f} | PSNR: {val\_psnr:.2f} dB | LR: {current\_lr:.6f}\\n")

 

        #保存最优去噪模型

        if val\_psnr > best\_psnr:

            best\_psnr = val\_psnr

            torch.save(model.state\_dict(), "denoise\_pretrain\_best.pth")

            print(f"-> 保存最优预训练模型，PSNR: {best\_psnr:.2f} dB")

 

    print(f"去噪预训练完成，最高PSNR: {best\_psnr:.2f} dB")



\# ---------------------------

\#阶段2分层冻结+联合微调（解决偏科，同时保住两个任务）

\# ---------------------------

def train\_joint\_finetune(epochs=20, denoise\_weight=1.0, cls\_weight=0.1):

    print("\\n===== 阶段2/2：分层冻结+多任务联合微调 =====")

    if trainloader\_sidd is None or trainloader\_cifar is None:

        print("数据集加载失败，退出微调")

        return

 

    #加载预训练的去噪模型

    if os.path.exists("denoise\_pretrain\_best.pth"):

        print("加载去噪预训练权重...")

        model.load\_state\_dict(torch.load("denoise\_pretrain\_best.pth"))

 

    #核心：冻结共享编码器的前2层（保护底层细节特征，不破坏去噪效果）

    print("冻结共享编码器前2层，保护去噪底层特征")

    for name, param in model.named\_parameters():

        if "shared\_encoder.0" in name or "shared\_encoder.1" in name:

            param.requires\_grad = False

        else:

            param.requires\_grad = True

 

    #优化器：仅更新未冻结的参数，用更小的学习率避免破坏预训练特征

    optimizer = optim.Adam(filter(lambda p: p.requires\_grad, model.parameters()), lr=5e-4, weight\_decay=1e-4)

    scheduler = optim.lr\_scheduler.CosineAnnealingLR(optimizer, T\_max=epochs, eta\_min=1e-6)

 

    best\_joint\_score = 0.0

    best\_psnr = 0.0

    best\_acc = 0.0

 

    for epoch in range(epochs):

        model.train()

        total\_denoise\_loss = 0.0

        total\_cls\_loss = 0.0

 

        #迭代器对齐

        sidd\_iter = iter(trainloader\_sidd)

        cifar\_iter = iter(trainloader\_cifar)

        max\_iter = min(len(trainloader\_sidd), len(trainloader\_cifar))

 

        for i in range(max\_iter):

            #加载数据

            noisy\_imgs, clean\_imgs = next(sidd\_iter)

            cls\_imgs, cls\_labels = next(cifar\_iter)

 

            noisy\_imgs = noisy\_imgs.to(device, non\_blocking=True)

            clean\_imgs = clean\_imgs.to(device, non\_blocking=True)

            cls\_imgs = cls\_imgs.to(device, non\_blocking=True)

            cls\_labels = cls\_labels.to(device, non\_blocking=True)

 

            #前向传播

            denoised\_out = model.forward\_denoise(noisy\_imgs)

            cls\_out = model.forward\_classify(cls\_imgs)

 

            #加权联合损失

            loss\_denoise = criterion\_denoise(denoised\_out, clean\_imgs)

            loss\_cls = criterion\_classify(cls\_out, cls\_labels)

            joint\_loss = denoise\_weight \* loss\_denoise + cls\_weight \* loss\_cls

 

            #反向传播

            optimizer.zero\_grad()

            joint\_loss.backward()

            optimizer.step()

 

            total\_denoise\_loss += loss\_denoise.item()

            total\_cls\_loss += loss\_cls.item()

 

        scheduler.step()

        avg\_denoise\_loss = total\_denoise\_loss / max\_iter

        avg\_cls\_loss = total\_cls\_loss / max\_iter

        current\_lr = optimizer.param\_groups\[0]\['lr']

 

        #验证两个任务的性能

        model.eval()

        with torch.no\_grad():

            #去噪验证

            val\_psnr = 0.0

            val\_ssim = 0.0

            for noisy\_imgs, clean\_imgs in trainloader\_sidd:

                noisy\_imgs = noisy\_imgs.to(device)

                clean\_imgs = clean\_imgs.to(device)

                denoised\_imgs = model.forward\_denoise(noisy\_imgs)

                val\_psnr += calculate\_psnr(denoised\_imgs, clean\_imgs)

                val\_ssim += calculate\_ssim(denoised\_imgs, clean\_imgs)

            val\_psnr /= len(trainloader\_sidd)

            val\_ssim /= len(trainloader\_sidd)

 

            #分类验证

            val\_correct = 0

            val\_total = 0

            for imgs, labels in testloader\_cifar:

                imgs = imgs.to(device)

                labels = labels.to(device)

                logits = model.forward\_classify(imgs)

                \_, pred = torch.max(logits, 1)

                val\_total += labels.size(0)

                val\_correct += (pred == labels).sum().item()

            val\_acc = 100 \* val\_correct / val\_total

 

        #打印日志

        print(f"\\nEpoch \[{epoch+1}/{epochs}] | LR: {current\_lr:.6f}")

        print(f"训练损失：去噪={avg\_denoise\_loss:.6f}, 分类={avg\_cls\_loss:.6f}")

        print(f"验证性能：PSNR={val\_psnr:.2f} dB, SSIM={val\_ssim:.4f}, 分类准确率={val\_acc:.2f}%")

 

        #保存最优联合模型

        joint\_score = 0.6 \* val\_psnr + 0.3 \* val\_acc

        if joint\_score > best\_joint\_score:

            best\_joint\_score = joint\_score

            best\_psnr = val\_psnr

            best\_acc = val\_acc

            torch.save(model.state\_dict(), "multitask\_final\_best.pth")

            print(f"  -> 保存最优多任务模型，综合得分={best\_joint\_score:.2f}")

 

    print("\\n===== 多任务训练全部完成 =====")

    print(f"最终最优性能：去噪PSNR={best\_psnr:.2f} dB, 分类准确率={best\_acc:.2f}%")



\# ======================================

\# 8. 测试函数

\# ======================================

def test\_denoise\_final():

    if trainloader\_sidd is None:

        print("无去噪数据集，跳过测试")

        return

    model.eval()

    with torch.no\_grad():

        final\_psnr = 0.0

        final\_ssim = 0.0

        for noisy\_imgs, clean\_imgs in trainloader\_sidd:

            noisy\_imgs = noisy\_imgs.to(device)

            clean\_imgs = clean\_imgs.to(device)

            denoised\_imgs = model.forward\_denoise(noisy\_imgs)

            final\_psnr += calculate\_psnr(denoised\_imgs, clean\_imgs)

            final\_ssim += calculate\_ssim(denoised\_imgs, clean\_imgs)

        final\_psnr /= len(trainloader\_sidd)

        final\_ssim /= len(trainloader\_sidd)

    print(f"去噪最终性能：PSNR={final\_psnr:.2f} dB, SSIM={final\_ssim:.4f}")

    return final\_psnr, final\_ssim



def test\_classify\_final():

    model.eval()

    with torch.no\_grad():

        final\_correct = 0

        final\_total = 0

        for imgs, labels in testloader\_cifar:

            imgs = imgs.to(device)

            labels = labels.to(device)

            logits = model.forward\_classify(imgs)

            \_, pred = torch.max(logits, 1)

            final\_total += labels.size(0)

            final\_correct += (pred == labels).sum().item()

        final\_acc = 100 \* final\_correct / final\_total

    print(f"分类最终性能：测试准确率={final\_acc:.2f}%")

    return final\_acc





if \_\_name\_\_ == "\_\_main\_\_":

    #去噪预训练（保住高PSNR）

    train\_denoise\_pretrain(epochs=20)

 

    #联合微调（同时学好分类，不破坏去噪）

    train\_joint\_finetune(epochs=40, denoise\_weight=1.0, cls\_weight=2.0)

 

    #最终测试

    print("\\n===== 最终模型测试结果 =====")

    if os.path.exists("multitask\_final\_best.pth"):

        model.load\_state\_dict(torch.load("multitask\_final\_best.pth"))

        print("已加载最优多任务模型")

 

    test\_denoise\_final()

    test\_classify\_final()

