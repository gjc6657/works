import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
import torch.utils.data as Data
from torchinfo import summary
import os
torch.manual_seed(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "dataset/train")
test1_path = os.path.join(BASE_DIR, "dataset/test1")
test2_path = os.path.join(BASE_DIR, "dataset/test2")
PTH_DIR = os.path.join(BASE_DIR, "pth")

"""
参考自https://mofanpy.com/tutorials/machine-learning/torch/CNN
这是本文重要的代码风格参考
我在这些基础上进行了以下优化：
数据增强（图象转换中，对图形进行了随机的翻转）
加上了batchnorm2d，对数据作了标准化
"""
"""
有关本文使用ai的一些说明
训练部分，由于我想用mgbd进行训练，输出结构与sgd还是有差别的
所以我让gpt帮我优化了一下可能的bug
在打乱顺序的问题上，gpt提醒我最好测试集用False以便于复现
在文件导入问题上，遇到了路径与vs终端路径不一致的问题，所以gpt帮我添加了一项路径的宏加在开头，其实有点像cmake的内容
"""

class cnn_net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, #RGB图像默认3通道
                out_channels = 16, #输出16张特征图
                kernel_size = 3, #3*3卷积核
                stride = 1, #每次移动1步
                padding = 1 #在外围加上一层0
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(), #激活函数
            nn.MaxPool2d(2) #池化
        )   
        # input(B,3,64,64)
        # output(B,16,32,32)
        """
        Sequential：按顺序一次自动执行整个cnn块中的所有语句
        打算把第一次从输入到激活再到卷积最后到池化全过程封装成一个函数
        """
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16, 
                out_channels = 32,
                kernel_size = 3, 
                stride = 1,
                padding = 1, 
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        # input(B,16,32,32)
        # output(B,32,16,16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1, 
                padding = 1, 
            ),
            nn.BatchNorm2d(64),
            nn.ReLU() ,
            nn.MaxPool2d(2) 
        )
        #input(B,32,16,16)
        #output(B,64,8,8)
        self.out = nn.Linear(64*8*8,3) #特征图由3张变为64张，尺寸经过三次池化已经变成8*8，即(B,4096)
        #这是一个三分类问题，于是输出是三种分类
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output
    
if __name__ == "__main__":#如果程序是主动执行的，就执行以下操作，如果是被引用的，就不执行，只引用函数和类
    #图像转换
    transform_style = transforms.Compose(
        [
            transforms.Resize([64,64]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    )

    #超参数
    BATCH_SIZE = 32 #采用MBGD，一次训练一小批
    EPOCH = 200

    #dataloader
    trainset = datasets.ImageFolder(root=train_path, transform=transform_style)
    testset1 = datasets.ImageFolder(root=test1_path, transform=transform_style)
    testset2 = datasets.ImageFolder(root=test2_path, transform=transform_style)
    
    print(f"训练集图片数量: {len(trainset)}")
    print(f"测试集1图片数量: {len(testset1)}")
    print(f"测试集2图片数量: {len(testset2)}")
    
    """
    准备用dataloader进行传入，batchsize确定每次传入几个数据，
    shuffle确定是否打乱，pinmemory在gpu有用
    """
    train_loader = Data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader1 = Data.DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader2 = Data.DataLoader(testset2, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    """
    为什么测试集选择false？
    因为为了方便复现
    """
    #初始化网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = cnn_net().to(device)

    summary(net, input_size=(1, 3, 64, 64), device=device)
    print(f'标签对应的ID: {trainset.class_to_idx}')

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    #weightdecay就是正则化，正则化确实是很值得研究的一个项目
    """
    在这里介绍几个常见的优化器，并解释一下为什么我会选择Adam
    优化器其实就是根据反向传播过程得到的grad计算新的参数，因为新的参数是根据梯度反向方向计算的，所以一定是让loss更小的
    SGD：每次就用一张图进行计算，想这次的数据集，要计算1298次
    确实很精细，容易跳出局部最小值，但是噪声对其影响非常大
    BGD：一次性计算1298张图，超级慢，对gpu的利用率太差了
    MBGD：一次性计算一小批次的图，比较合理的利用gpu，同时拟合精度还算可以。比如我这里选择一次计算32张
    上述内容其实都是经典版本，现代pytorch中的例如SGD已经融合了Momentum，Weight Deca这些算法，其实挺全面的。

    不过基于学的更多点的考虑，我觉得还是需要介绍和使用一下ada系列优化器
    ada系列优化器其实是自适应参数的，比如adagrad，利用基于历史参数的变化轨迹来自由变更学习率，
    前期学习率相对较大，下降很快，比较适合参数间距较大较为稀疏的场景。
    但是他的缺点也很明显，由于历史参数的影响，中后期学习率会因为持续的下降变得很小，可能提前收敛。
    于是对adagrad进行优化，产生了adadelta。adadelta通过一个参数p（其实是rou，但是不是md文档打不出来）
    对最近的数据进行了比重上的加强，对历史久远的数据比重削弱了。这样，学习率基本只受到最近一段时间的影响。
    但由于训练后期数据梯度不够明显，可能导致在局部最小值附近抖动。
    于是产生了rmsprop，让p=0.5，成为计算梯度均方根，比较适合非平稳的目标，对于rnn效果比较好
    以上的ada系列优化器都依赖于设置一个全局学习率。那有没有对学习率没有那么的敏感的算法呢？
    接下来就要请出我们的adam。他相当于RMSprop + Momentum，利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率
    所以他对学习率没那么敏感，因为他调节学习率的方式很多样。但不代表就不需要选择合适学习率了
    你一次调整到合适学习率的概率变高了！
    adam对新手更加友好，尤其在数据没那么大的情况下，用adam更合适，收敛更快
    """

    print("Start")#开始训练！
    max_correct = 98 #用来决定是否保存模型
    for epoch in range(EPOCH):
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_id, (datas, labels) in enumerate(train_loader):#取用数据,标签和id
            datas, labels = datas.to(device),labels.to(device)#把数据放到gpu里
            optimizer.zero_grad()#总的loss已经被记录到trainloss里面了，优化器里这个就不重要了
            #因为pytorch梯度默认会累加，不清空会把上一次batch结果叠加进去
            outputs = net(datas)#这里其实就是自动调用forward函数
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()#loss.item()会把tensor变成python里的普通数字
            _, predicted = torch.max(outputs.data, dim=1)#即每张图对应一个预测类别
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            if (batch_id + 1) % 10 == 0:#每10个batch打印一次loss。因为MBGD本身波动就比较大，所以这里只看大概趋势
                print(
                    f"Epoch [{epoch+1}/{EPOCH}] "
                    f"Batch [{batch_id+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
        avg_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        #进入验证阶段，因为是用mbgd，不应该在每个batch都检测一次，会大大拖慢训练时间，所以把检测放到了batch之外
        if epoch > 50 and (epoch % 10) == 0:
            os.makedirs(PTH_DIR, exist_ok=True)
            PATH = os.path.join(PTH_DIR, "modeltemp.pth")
            torch.save(net.state_dict(), PATH)#保存参数
            model = cnn_net()
            model.load_state_dict(torch.load(PATH))
            model.eval()
            model.to(device)

            #初始化正确数量
            correct1 = 0
            correct2 = 0

            #初始化总数量
            total1 = 0
            total2 = 0

            #分别测试两个数据集
            with torch.no_grad():
                for i ,(datas1, labels1) in enumerate(test_loader1):
                    datas1, labels1 = datas1.to(device), labels1.to(device)
                    output_test1 = model(datas1)
                    _, predicted1 = torch.max(output_test1.data, dim=1)#找概率最大的类别
                    total1 += predicted1.size(0)
                    correct1 += (predicted1 == labels1).sum().item()

                for i ,(datas2, labels2) in enumerate(test_loader2):
                    datas2, labels2 = datas2.to(device), labels2.to(device)
                    output_test2 = model(datas2)
                    _, predicted2 = torch.max(output_test2.data, dim=1)
                    total2 += predicted2.size(0)
                    correct2 += (predicted2 == labels2).sum().item()

                c1 = 0
                c2 = 0
                #计算正确率
                c2 = correct2 / total2 * 100
                c1 = correct1 / total1 * 100
                #输出整体训练情况
                print(
                    f"\nEpoch [{epoch+1}/{EPOCH}] "
                    f"平均损失: {avg_loss:.4f} "
                    f"训练集准确率: {train_acc:.2f}% "
                    f"Test1准确率: {c1:.2f}% "
                    f"Test2准确率: {c2:.2f}%"
                )
                #如果当前正确率更高，就保存模型，这样最后留下来的基本就是训练过程中效果最好的模型
                if (c1 > max_correct):
                    max_correct = c1
                    MAX_PATH = os.path.join(PTH_DIR, f"model_best_{max_correct}.pth")
                    print(f"save {MAX_PATH}")
                    torch.save(net.state_dict(),MAX_PATH)
        
"""
后面测试以及输出的地方没什么想法，于是我就大致借鉴了实例代码，因为我觉得实例代码出错概率不高，看起来就比较可信
写学习过程的时候，还是试了一下自己调用函数自己写中间的过程，其实这个调用函数的过程已经在forwadr定义过了，
其他的函数已经封装好。
为了写这个还专门看api，但是很多没找到就会去搜索方法
"""