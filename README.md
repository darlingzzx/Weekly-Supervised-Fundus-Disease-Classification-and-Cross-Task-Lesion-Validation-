

# 病灶感知的跨任务统一学习框架

# 0. 最终要做成什么样


- **任务统一**：ODIR（多标签眼病识别）+ DDR（糖网分级/病灶相关任务）
- **病灶感知**：模型不仅看全图，还学习关键病灶区域
- **跨任务迁移**：DDR 提供更强的结构性病灶知识，帮助 ODIR
- **方法创新**：“统一表征 + 病灶约束 + 跨任务对齐”
- **可解释性**：模型输出的注意力/热图能对应病灶区域
- **泛化性**：在不同数据分布下保持稳定

---

# 1. 论文核心问题定义

## 1.1 研究动机
眼底图像分析存在两个现实问题：

1. **图像级标签太粗**
   - 很多数据集只有疾病标签或分级标签
   - 模型容易学到伪相关特征
   - 缺少对病灶区域的显式建模

2. **不同数据集的任务粒度不同**
   - ODIR 更偏多标签疾病筛查
   - DDR 更偏 DR 分级和病灶相关结构
   - 两者如果单独训练，知识无法共享

## 1.2 研究目标
你要提出一个框架，使得：

- ODIR 的多标签识别受益于 DDR 中的病灶结构知识
- DDR 的分级能力受益于 ODIR 中更丰富的多标签语义
- 统一表征能更好地泛化
- 模型能提供更好的解释和更强的鲁棒性

---

# 2. 数据集规划

---

## 2.1 ODIR 数据集
### 任务性质
- 多标签分类
- 典型标签：正常、糖网、白内障、青光眼、黄斑病变等
- 适合做共病识别与筛查

### 任务
1. 多标签疾病识别
2. 类别不平衡学习
3. 公平性分析
4. 可解释性分析
5. 低质量图像鲁棒性分析

### 预处理要点
- 裁剪黑边
- 统一尺寸
- 亮度/对比度规范化
- 训练时增强：
  - 随机旋转
  - 水平翻转
  - Color jitter
  - Random crop
  - Gaussian blur
  - Cutout / Random erasing

---

## 2.2 DDR 数据集
### 任务性质
- 糖网分级
- 可能包含更细的病灶相关标注或类别信息
- 适合做 DR 分级与病灶感知学习

### 任务
1. DR 分级
2. 病灶敏感分类
3. 弱监督病灶定位
4. 多任务学习
5. 迁移到 ODIR

### 预处理要点
- 同样做黑边裁剪和尺寸统一
- 若存在严重类别不平衡，必须做重采样或损失重加权
- 病灶类常常稀少，训练时要特别注意增强策略

---

## 2.3 数据划分建议
### ODIR
- Train / Val / Test = 70 / 10 / 20
- 确保同一患者图像不会泄漏到不同集合

### DDR
- 使用官方划分优先
- 如果没有，建议按患者级别划分
- 尽量保持类别分布稳定

### 外部验证集
额外引入一个公开数据集，会极大增强论文说服力，例如：
- EyePACS
- Messidor / Messidor-2
- APTOS
- IDRiD
- RFMiD
---

# 3. 论文方法设计

---

## 3.1 总体框架
四个核心模块：

1. **共享视觉编码器**
2. **任务特定预测头**
3. **病灶感知模块**
4. **跨任务对齐模块**

---

## 3.2 共享视觉编码器

### 目的
提取统一的 retinal representation，让 ODIR 和 DDR 共用底层视觉特征。

### 推荐 backbone
按优先级建议：

#### 方案 A：Swin Transformer
优点：
- 全局建模能力强
- 适合医学图像
- 在分类任务上通常比普通 CNN 更有潜力

#### 方案 B：ViT
优点：
- 结构清晰
- 易于做 token 级分析和可解释性
- 适合统一建模叙事

#### 方案 C：ConvNeXt
优点：
- 训练稳定
- 工程上相对容易
- 兼具 CNN 的局部归纳偏置

### 建议
如果你想更像“顶会论文”，建议最终版本用 **Swin Transformer 或 ViT**。  
如果你担心训练不稳定，可以先用 ConvNeXt 做 baseline。

---

## 3.3 任务特定预测头

### ODIR 头
- 多标签分类头
- 输出每个疾病的概率

建议损失：
- BCEWithLogitsLoss
- Focal Loss
- Asymmetric Loss
- Class-balanced BCE

### DDR 头
- 分级分类头
- 如果是有序类别，建议加 ordinal learning

建议损失：
- Cross Entropy
- Ordinal Regression Loss
- Label Distribution Learning

---

## 3.4 病灶感知模块

这是你论文的核心创新点之一。

### 目标
让模型不仅输出疾病类别，还显式关注与病灶相关的区域。

### 可选实现方式

---

### 方案 1：注意力图 / CAM 伪病灶监督
#### 思路
- 利用分类头生成 attention map 或 CAM
- 将高响应区域视作“伪病灶区域”
- 训练中约束模型关注这些区域

#### 优点
- 简单可实现
- 不需要像素级标注
- 很适合弱监督设置

#### 实现方式
1. 前向传播得到分类 logits
2. 从中间层提取 attention map
3. 对高响应区域生成 soft mask
4. 用一致性损失约束模型在增强前后关注区域稳定

#### 相关损失
- Attention consistency loss
- CAM sparsity loss
- Region activation regularization

---

### 方案 2：病灶原型学习
#### 思路
- 为不同病灶学一个 prototype 向量
- 图像特征靠近对应 prototype
- 让模型具备病灶语义空间

#### 优点
- 解释性更强
- 跨任务对齐更自然
- 有一定论文创新味道

#### 设计方式
- 为每类病灶定义一个或多个 prototype
- 通过对比学习让病灶 patch 向对应 prototype 聚合
- 非病灶区域远离这些 prototype

---

### 方案 3：局部-全局对比学习
#### 思路
- 从图像中采样局部 patch
- 病灶区域 patch 与图像级标签做对比对齐
- 背景 patch 与标签解耦

#### 优点
- 更符合“病灶感知”概念
- 有助于提升定位与分类

---

## 3.5 跨任务对齐模块

这是你的第二个核心创新点。

### 目标
让 ODIR 和 DDR 的知识相互促进，而不是并行训练。

### 你可以采用的对齐策略

#### 策略 A：特征空间对齐
- 将两个任务的中间层特征映射到统一 embedding space
- 用 cosine loss / contrastive loss 拉近语义相近样本


#### 策略 B：局部-全局一致性对齐
- DDR 中病灶感知较强
- 用 DDR 产生的病灶 attention 作为 ODIR 学习的辅助监督
- 强制 ODIR 的注意力区域与 DDR 学到的病灶区域分布一致


---

# 4. 损失函数设计

你可以把总损失写成：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{ODIR}
+
\lambda_1 \mathcal{L}_{DDR}
+
\lambda_2 \mathcal{L}_{lesion}
+
\lambda_3 \mathcal{L}_{align}
+
\lambda_4 \mathcal{L}_{ssl}
$$

其中：

- $$\mathcal{L}_{ODIR}$$：ODIR 多标签分类损失
- $$\mathcal{L}_{DDR}$$：DDR 分级损失
- $$\mathcal{L}_{lesion}$$：病灶感知约束
- $$\mathcal{L}_{align}$$：跨任务对齐
- $$\mathcal{L}_{ssl}$$：自监督预训练损失

---

## 4.1 ODIR 损失
推荐：
- BCEWithLogitsLoss
- Asymmetric Loss
- Focal Loss

如果类别极不平衡，优先：
- Asymmetric Loss
- Class-balanced loss

---

## 4.2 DDR 损失
如果是普通分级：
- Cross Entropy

如果你希望利用等级序关系：
- Ordinal regression loss
- CORAL
- Label distribution learning

---

## 4.3 病灶感知损失
可以设计成以下任一组合：

### 方式 A：Attention consistency
$$
\mathcal{L}_{lesion} = \|A(x) - A(\tilde{x})\|
$$
其中 $$x$$ 和 $$\tilde{x}$$ 是同一图像的不同增强视图。

### 方式 B：Sparse activation
让注意力更集中：
$$
\mathcal{L}_{lesion} = \|A\|_1
$$

### 方式 C：伪掩码监督
如果你有伪病灶 mask：
$$
\mathcal{L}_{lesion} = \mathrm{BCE}(A, M)
$$

---

## 4.4 跨任务对齐损失
### 对比学习式
$$
\mathcal{L}_{align} = - \log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\mathrm{sim}(z_i, z_k)/\tau)}
$$

其中：
- $$z_i$$ 和 $$z_j$$ 是语义一致样本的特征
- $$\tau$$ 是温度系数

### 也可以用
- cosine similarity loss
- KL divergence
- MSE on projected embeddings

---

## 4.5 自监督损失
如果你做预训练，可选：
- MAE reconstruction loss
- DINO contrastive distillation
- MoCo v3
- SimCLR

### 推荐
如果你时间有限：
- **MAE** 比较适合视觉基础模型叙事
- **DINO** 比较适合学表征和 attention map
- 两者都可以，但先做一个即可

---

# 5. 训练流程设计

建议三阶段训练。

---

## 5.1 阶段 1：自监督预训练
### 数据
- 使用 ODIR + DDR 中所有可用图像
- 不依赖标签

### 目标
- 学一个通用 retinal encoder
- 提高后续监督训练的收敛与泛化

### 训练方式
- 随机裁剪
- 掩码重建或对比学习
- 多视图增强

### 产出
- 一个预训练 backbone
- 后续作为初始化

---

## 5.2 阶段 2：单任务监督预训练
分别训练：

### ODIR 分支
- 多标签分类
- 学习疾病共现语义

### DDR 分支
- DR 分级
- 学习严重程度与病灶敏感特征

### 目标
- 先让各自任务收敛
- 方便后面联合学习更稳定

---

## 5.3 阶段 3：联合训练
把两个任务放到一个框架内联合优化。

### 训练策略
- 共享 backbone
- 任务特定 head
- 病灶分支共享
- 对齐模块开启
- 损失按权重加和

### 训练技巧
- 先冻结 backbone 前几层，稳定训练
- 再逐步解冻
- 使用 cosine learning rate schedule
- 采用 warmup

---

# 6. 实验设计

---

## 6.1 主实验
### 在 ODIR 上
测试：
- 多标签分类性能
- AUC
- mAP
- F1
- precision / recall

### 在 DDR 上
测试：
- 分级准确率
- quadratic weighted kappa
- macro-F1
- AUC

---

## 6.2 对比方法
你至少要和这些基线比：

### CNN 基线
- ResNet50
- DenseNet121
- EfficientNet-B0/B3

### Transformer 基线
- ViT-B
- Swin-T / Swin-S
- DeiT

### 医学影像分割/分类基线
- Attention U-Net
- UNet++
- nnU-Net

### 预训练基线
- MAE
- DINO
- RETFound 初始化

### 多任务基线
- 普通 shared-backbone multi-task learning
- hard parameter sharing
- uncertainty weighting
- PCGrad / GradNorm

---

## 6.3 消融实验
这是必做项。

### 消融项建议
1. 去掉病灶感知模块
2. 去掉跨任务对齐模块
3. 去掉自监督预训练
4. 去掉注意力一致性
5. 去掉原型学习
6. 仅用 ODIR 训练
7. 仅用 DDR 训练
8. 联合训练但无共享病灶分支

### 你要回答的问题
- 病灶模块到底有没有用？
- 跨任务对齐是不是带来提升？
- 自监督预训练是不是关键？
- ODIR 和 DDR 的知识是否真的互补？

---

## 6.4 泛化实验

### 跨数据集测试
- 在 ODIR 训练，测试到其他 fundus 数据集
- 在 DDR 训练，测试到 EyePACS / Messidor / APTOS

### 跨质量测试
- 高质量 vs 低质量图像
- 不同相机来源
- 不同增强程度

### 跨子群测试
- 性别
- 年龄
- 标签稀有度
- 视图侧别

---

## 6.5 可解释性实验
### 方法
- Grad-CAM
- Attention rollout
- Score-CAM
- Proto-based visualization

### 验证
- 定性图可视化
- 如果有局部标注，计算热图与病灶区域重合程度
- 定量指标：
  - overlap ratio
  - pointing game
  - localization accuracy

---

# 7. 结果分析应该怎么写

---

## 7.1 性能提升分析
你要展示：
- 分类指标显著提升
- 少数类提升更明显
- 病灶类和难例上提升更大

---

## 7.2 泛化提升分析
说明：
- 在不同数据集上你的方法掉点更少
- 对域偏移更稳定
- 对低质量输入更鲁棒

---

## 7.3 可解释性分析
要说明：
- 模型关注区域更接近病灶
- 不是盯着边缘、黑边、伪相关区域
- 注意力图更符合临床直觉

---

## 7.4 公平性分析
如果你要冲更高水平，建议加：
- subgroup performance gap
- worst-group accuracy
- calibration error

这会让论文更像“真实世界医学 AI”。
---
# 8. 论文创新点表述
## 创新点 1：跨任务统一视网膜学习
我们首次将 ODIR 的多标签筛查和 DDR 的分级任务统一到一个病灶感知框架中，实现不同粒度监督的协同学习。

## 创新点 2：病灶感知与注意力一致性
我们设计了病灶感知约束，使模型在仅有图像级标签的条件下也能显式学习关键病灶区域，提高可解释性和定位能力。

## 创新点 3：提升泛化与鲁棒性
我们在多数据集和多子群设置下验证了方法对类别不平衡、域偏移和低质量图像的鲁棒性，展示了更强的临床应用潜力。

---