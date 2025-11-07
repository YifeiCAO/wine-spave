# Meta-Learning 实现说明

## 概述

本实现基于**In-Context Meta-Learning**方法，训练RNN成为一个"快速学习器"，能够从1D规则快速适应并泛化到2D组合任务，模拟人类的组合泛化能力。

## 核心思想

### Meta-Learning vs 传统训练

- **传统训练**: 训练模型学习特定的wine space
- **Meta-Learning**: 训练模型学习"如何学习"任何新的wine space

### 关键概念

1. **Task/Episode**: 每个任务包含：
   - 一个新的4x4地图（随机生成的wine space）
   - **Support Set**: 1D规则样本（用于适应）
   - **Query Set**: 2D规则样本（用于评估）

2. **Meta-Training (外循环)**:
   - 通过大量随机任务训练RNN
   - RNN通过隐藏状态适应新任务，而不改变权重θ
   - 基于query set的表现更新权重θ

3. **Meta-Testing (内循环)**:
   - 冻结权重θ_final
   - 在真实任务上测试（模拟人类实验）
   - 通过1D支持集适应，在2D查询集上测试

## 实现架构

### 文件结构

```
meta_learning.py          # Meta-learning核心实现
├── MetaTask              # 任务类（包含support和query sets）
├── MetaTaskGenerator     # 任务生成器
├── SequentialRNN          # 支持序列输入的RNN包装器
├── meta_train()          # Meta-training函数
├── meta_test()           # Meta-testing函数
└── create_meta_learning_args()  # 参数创建函数
```

### 主要组件

#### 1. SequentialRNN
- 包装基础RNN模型
- 支持序列输入，维护隐藏状态
- 允许在多个样本间传递隐藏状态

#### 2. MetaTaskGenerator
- 生成随机meta-learning任务
- 每个任务包含新的4x4地图
- Support set: 1D规则样本
- Query set: 2D规则样本

#### 3. meta_train()
Meta-Training流程：
```
For each meta-iteration:
  1. Sample batch of tasks
  2. For each task:
     a. Process support set (adapt hidden state)
     b. Process query set (evaluate on 2D)
     c. Calculate meta-loss
  3. Meta-update: Update θ based on aggregated meta-loss
```

#### 4. meta_test()
Meta-Testing流程：
```
1. Freeze model weights θ_final
2. For each country (real task):
   a. Adaptation: Process 1D support set
   b. Evaluation: Process 2D query set
3. Compare with human baseline
```

## 使用方法

### 1. 在Notebook中使用

参见 `notebooks/test_pipeline.ipynb` 的 Section 9。

### 2. 基本使用流程

```python
from meta_learning import meta_train, meta_test, create_meta_learning_args

# 1. 创建meta-learning参数
meta_args = create_meta_learning_args(args)
meta_args.meta_lr = 0.001
meta_args.n_support = 16
meta_args.n_query = 32
meta_args.n_meta_iterations = 10000

# 2. Meta-Training
model = get_model(meta_args)
meta_trained_model, meta_losses = meta_train(model, meta_args)

# 3. Meta-Testing
meta_test_results = meta_test(
    meta_trained_model,
    meta_args,
    grid_data_gen,
    country_data_gen,
    country_loader
)
```

## 参数说明

### Meta-Learning参数

- `meta_lr`: Meta-learning学习率（通常0.001）
- `n_support`: 每个任务的support样本数（建议16-32）
- `n_query`: 每个任务的query样本数（建议32-64）
- `n_meta_iterations`: Meta-training迭代次数（建议≥10000）
- `n_tasks_per_batch`: 每批任务数（建议4-8）

## 预期结果

### 成功标准

如果meta-learning成功，RNN应该能够：
- 在**从未见过任何2D规则训练**的情况下（零样本）
- 在2D国家任务上表现出**远高于随机水平**的准确率
- 成功模拟人类的组合泛化能力

### 性能基线

- **随机猜测**: 33.33% (3选1)
- **人类基线**: ~66-80% (根据论文)
- **目标**: >66% (超过人类基线下限)

## 实现细节

### 1. 序列处理

`SequentialRNN`通过维护LSTM的隐藏状态(h_n, c_n)来实现序列处理：
- Support set样本依次输入，隐藏状态逐步适应
- Query set使用适应后的隐藏状态进行预测

### 2. 任务生成

`MetaTaskGenerator`生成任务时：
- Support set: 只包含1D规则（cong != -1）
- Query set: 只包含2D规则（cong == -1，需要2D推理）

### 3. Meta-Update

Meta-loss基于query set的表现计算，然后更新模型的初始权重θ，使其在未来的任务中能够更快适应。

## 注意事项

1. **训练时间**: Meta-training需要较长时间（数千到数万次迭代）
2. **内存使用**: 每个任务需要生成新的grid，注意内存管理
3. **任务多样性**: 确保生成的任务足够多样化
4. **超参数调优**: 可能需要调整meta_lr、n_support等参数

## 与论文的对应关系

- **1D训练数据**: Support set（模拟人类学习阶段）
- **2D国家任务**: Query set（模拟人类测试阶段）
- **组合元算法**: Meta-trained RNN的"学习策略"
- **零样本学习**: Meta-testing阶段，模型从未见过2D规则训练

## 未来改进方向

1. **更高效的任务生成**: 缓存或预生成任务
2. **更好的支持集设计**: 针对2D国家的1D支持集设计
3. **多步适应**: 支持多轮adaptation
4. **注意力机制**: 增强对关键样本的关注
5. **任务嵌入**: 显式学习任务表示



