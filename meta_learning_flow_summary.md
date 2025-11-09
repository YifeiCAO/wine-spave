# Meta-Learning 流程总结：查询集和支持集划分

## 📋 整体流程概览

```
Meta-Training (外循环)
├── 生成新任务 (新的4x4 grid)
├── 划分 Support Set (1D规则)
├── 划分 Query Set (2D规则)
├── In-Context Learning (处理Support Set，适应hidden state)
├── In-Context Testing (处理Query Set，评估2D泛化)
└── Meta-Update (根据Query Set的loss更新模型权重)
```

## 🔍 详细划分逻辑

### 1. 数据生成阶段 (`data.py`)

#### 1.1 基础数据生成 (`generate_ungrouped_samples`)
```python
# 对于每个wine pair和每个context
for idx1, idx2 in permutations(idxs, 2):
    for ctx in range(2):
        r1, r2 = f1[ctx], f2[ctx]  # 当前context的rank
        d = r1 - r2  # rank差
        
        if d != 0:
            y = int(d > 0)  # y基于当前context计算
            
            if abs(d) == 1:
                train.append((ctx, f1, f2, y))  # 1-level difference
            elif abs(d) > 1:
                test.append((ctx, f1, f2, y))   # 需要泛化
```

#### 1.2 Congruency计算 (`utils.py`)
```python
def get_congruency(loc1, loc2):
    (x1, y1), (x2, y2) = loc1, loc2
    if (x1==x2) or (y1==y2):
        cong = 0  # 中性：某个维度相同
    else:
        cong = 1 if (x1<x2) == (y1<y2) else -1
    return cong
```

**Congruency含义**：
- `cong = 1` (一致): 两个维度方向相同
  - 例如: wine1=(2,3), wine2=(1,2) → x1>x2且y1>y2
- `cong = -1` (不一致): 两个维度方向相反  
  - 例如: wine1=(2,1), wine2=(1,2) → x1>x2但y1<y2
- `cong = 0` (中性): 某个维度相同
  - 例如: wine1=(2,1), wine2=(1,1) → y1==y2

### 2. Support Set (支持集) 划分 (`meta_learning.py:56-94`)

#### 2.1 代码实现
```python
def generate_task(self):
    # 创建新的4x4 grid
    grid = GridDataGenerator(...)
    
    # Support Set: 从grid.train中筛选
    support_set = []
    for sample in grid.train:
        ctx, loc1, loc2, y, info = sample
        cong = info.get('cong', 0)
        
        # 条件1: 排除incongruent样本
        if cong != -1:
            # 条件2: 只包含1-level difference (rank差=1)
            rank1 = loc1[ctx]
            rank2 = loc2[ctx]
            rank_diff = abs(rank1 - rank2)
            
            if rank_diff == 1:
                support_set.append(sample)
    
    # 使用所有满足条件的样本（不使用n_support参数随机采样）
    return MetaTask(grid, support_set, query_set)
```

#### 2.2 划分条件总结
| 条件 | 说明 | 目的 |
|------|------|------|
| 来源 | `grid.train` | 使用训练集中的样本 |
| Congruency | `cong != -1` | 排除需要2D推理的样本 |
| Rank差 | `abs(rank1 - rank2) == 1` | 只包含相邻pair（1-level difference） |
| 采样策略 | 使用所有满足条件的样本 | 确保全面学习所有相邻关系 |

#### 2.3 示例
```
Context 0的样本:
- wine1=(2,1), wine2=(1,1)
  - rank差 = |2-1| = 1 ✓
  - cong: y1==y2 → cong=0 ✓
  - ✅ 加入Support Set

- wine1=(2,1), wine2=(1,2)  
  - rank差 = |2-1| = 1 ✓
  - cong: x1>x2但y1<y2 → cong=-1 ✗
  - ❌ 排除（incongruent）
```

### 3. Query Set (查询集) 划分 (`meta_learning.py:95-110`)

#### 3.1 代码实现
```python
# Query Set: 从grid.test中筛选
query_set = []
for sample in grid.test:
    ctx, loc1, loc2, y, info = sample
    cong = info.get('cong', -1)
    
    # 只包含incongruent样本
    if cong == -1:
        query_set.append(sample)

# 如果样本数 > n_query，随机采样
if len(query_set) > self.n_query:
    query_set = random.sample(query_set, self.n_query)
```

#### 3.2 划分条件总结
| 条件 | 说明 | 目的 |
|------|------|------|
| 来源 | `grid.test` | 使用测试集中的样本 |
| Congruency | `cong == -1` | 只包含incongruent样本（需要2D推理） |
| Rank差 | `abs(rank1 - rank2) > 1` | 非相邻pair（需要泛化） |
| 采样策略 | 如果>n_query则随机采样 | 控制Query Set大小 |

#### 3.3 为什么Query Set只用incongruent？

**核心思想**：
- Support Set学习的是**1D规则**：在当前context下，rank高的更好
- Query Set测试的是**在干扰下坚持1D规则**：
  - 虽然另一个维度给出相反信号（incongruent）
  - 但y仍然基于当前context计算
  - 模型需要忽略另一个维度的干扰，坚持当前context的规则

**示例**：
```
Context 0的样本:
- wine1=(2,1), wine2=(1,2)
  - 在Context 0: rank差=2-1=1 → y=1 (wine1更好)
  - 在Context 1: rank差=1-2=-1 → 如果只看Context 1，wine2更好
  - cong: x1>x2但y1<y2 → cong=-1 ✓
  - ✅ 加入Query Set
  
  测试目标: 模型需要在Context 1给出相反信号时，
           仍然遵循Context 0的规则（wine1更好）
```

### 4. Meta-Training 流程 (`meta_learning.py:197-361`)

#### 4.1 完整流程
```python
for meta_iter in range(n_meta_iterations):
    # 1. 生成一批任务
    tasks = [task_generator.generate_task() for _ in range(n_tasks_per_batch)]
    
    total_meta_loss = 0.0
    
    for task in tasks:
        # 2. 准备Support Set和Query Set
        support_samples = prepare_samples(task.support_set)
        query_samples = prepare_samples(task.query_set)
        
        # 3. In-Context Learning: 处理Support Set
        #    - Hidden state适应，但模型权重θ不变
        support_outputs, adapted_hidden = seq_model.forward_sequence(support_samples)
        
        # 4. In-Context Testing: 处理Query Set
        #    - 使用适应后的hidden state
        query_outputs, _ = seq_model.forward_sequence(query_samples, adapted_hidden)
        
        # 5. 计算Query Set的loss（meta-loss）
        query_preds = torch.cat(query_outputs, dim=0)
        query_labels = torch.cat([y for _, _, _, y in query_samples], dim=0)
        task_loss = loss_fn(query_preds, query_labels)
        total_meta_loss += task_loss
    
    # 6. Meta-Update: 根据平均meta-loss更新模型权重
    avg_meta_loss = total_meta_loss / n_tasks_per_batch
    optimizer.zero_grad()
    avg_meta_loss.backward()
    optimizer.step()
```

#### 4.2 关键点
- **Hidden State适应**: 处理Support Set时，hidden state会适应1D规则
- **权重冻结**: 在单个任务内，模型权重θ不变，只有hidden state变化
- **Meta-Loss**: 基于Query Set的loss，用于更新模型权重
- **Meta-Update**: 更新模型权重，使其成为"快速学习器"

### 5. Meta-Testing 流程 (`meta_learning.py:581-696`)

#### 5.1 简化版测试 (`meta_test_simple`)
```python
# 1. 冻结模型权重
model.eval()

# 2. 对于每个测试任务
for task_idx in range(n_test_tasks):
    task = task_generator.generate_task()
    
    # 3. 适应: 处理Support Set
    support_outputs, adapted_hidden = seq_model.forward_sequence(support_samples)
    
    # 4. 测试: 处理Query Set
    query_outputs, _ = seq_model.forward_sequence(query_samples, adapted_hidden)
    
    # 5. 计算准确率
    preds = torch.argmax(query_preds, dim=1)
    accuracy = (preds == query_labels).float().mean()
```

## 📊 数据流图

```
┌─────────────────────────────────────────────────────────┐
│             数据生成 (GridDataGenerator)                 │
├─────────────────────────────────────────────────────────┤
│  generate_ungrouped_samples()                            │
│  ├── train: rank_diff == 1 (1-level difference)         │
│  └── test:  rank_diff > 1  (需要泛化)                    │
│                                                           │
│  append_info()                                            │
│  └── 为每个样本添加cong信息                                │
│      ├── cong = 1:  congruent (一致)                      │
│      ├── cong = -1: incongruent (不一致)                 │
│      └── cong = 0:  neutral (中性)                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│         Meta-Task生成 (MetaTaskGenerator)                │
├─────────────────────────────────────────────────────────┤
│  Support Set (从grid.train筛选)                          │
│  ├── cong != -1 (排除incongruent)                        │
│  └── rank_diff == 1 (相邻pair)                           │
│                                                           │
│  Query Set (从grid.test筛选)                              │
│  └── cong == -1 (只包含incongruent)                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Meta-Training (外循环)                       │
├─────────────────────────────────────────────────────────┤
│  1. 生成任务批次                                           │
│  2. 对每个任务:                                            │
│     ├── In-Context Learning (Support Set)               │
│     │   └── 适应hidden state，学习1D规则                  │
│     ├── In-Context Testing (Query Set)                   │
│     │   └── 测试2D泛化能力（在干扰下坚持1D规则）           │
│     └── 计算meta-loss                                     │
│  3. Meta-Update: 更新模型权重θ                           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Meta-Testing (内循环)                        │
├─────────────────────────────────────────────────────────┤
│  1. 冻结模型权重                                           │
│  2. 对每个测试任务:                                        │
│     ├── 适应: 处理Support Set                             │
│     └── 测试: 处理Query Set，计算准确率                    │
└─────────────────────────────────────────────────────────┘
```

## 🎯 关键理解

### 1. Support Set的作用
- **学习1D规则**: 在当前context下，rank高的更好
- **相邻pair**: 最容易学习的规则（rank差=1）
- **排除干扰**: 排除incongruent样本，确保规则清晰

### 2. Query Set的作用
- **测试2D推理**: 在另一个维度给出相反信号时，能否坚持当前context的规则
- **Incongruent样本**: 两个维度方向相反，需要忽略干扰
- **泛化能力**: 测试模型能否从1D规则泛化到2D场景

### 3. 为什么这样设计？
- **模拟人类学习**: 先学习简单的1D规则，再测试在复杂场景下的应用
- **组合泛化**: 测试模型能否从1D规则组合到2D场景
- **快速适应**: 通过meta-learning训练模型成为"快速学习器"

## 📝 代码位置总结

| 功能 | 文件 | 行数 |
|------|------|------|
| Support Set划分 | `meta_learning.py` | 72-94 |
| Query Set划分 | `meta_learning.py` | 95-110 |
| Meta-Training | `meta_learning.py` | 197-361 |
| Meta-Testing | `meta_learning.py` | 581-696 |
| 数据生成 | `data.py` | 270-316 |
| Congruency计算 | `utils.py` | 1-7 |

