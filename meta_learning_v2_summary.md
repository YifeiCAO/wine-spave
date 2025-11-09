# Meta-Learning V2 实现总结

## 📋 新设计概述

根据你的需求，我重新实现了meta learning流程，主要变化：

### 1. 规则向量系统
- **1D规则**：
  - `[1,0,0,0]` - Sweet（使用Context 0的rank）
  - `[0,1,0,0]` - Dry（Sweet的反向，使用 `grid_size-1 - Context 0的rank`）
  - `[0,0,1,0]` - Light（使用Context 1的rank）
  - `[0,0,0,1]` - Full（Light的反向，使用 `grid_size-1 - Context 1的rank`）

- **2D规则**：
  - `[1,0,1,0]` - Sweet + Light
  - `[1,0,0,1]` - Sweet + Full
  - `[0,1,1,0]` - Dry + Light
  - `[0,1,0,1]` - Dry + Full

### 2. 输入格式
- **旧格式**: `(ctx, f1, f2)` - ctx是0或1的整数
- **新格式**: `(rule_vector, wine_id1, wine_id2)` - rule_vector是4维向量

### 3. 输出格式
- **旧格式**: 2类 `[Wine1更好, Wine2更好]`
- **新格式**: 3类 `[Wine1胜, Wine2胜, 平局]`

### 4. 支持集（Support Set）
- **内容**: 只包含1D规则的样本
- **生成**: 为每个1D规则生成多个wine pair样本
- **目的**: 让RNN学习当前任务的"认知地图"（每个维度的规则）

### 5. 查询集（Query Set）
- **内容**: 主要包含2D规则的样本
- **生成**: 为每个2D规则生成wine pair样本
- **目的**: 测试RNN是否能零样本泛化到2D组合规则

## 🔧 实现细节

### 核心函数

#### `get_wine_attribute_value(wine_loc, rule_vector, grid_size=4)`
根据规则向量计算wine的属性值：
- 1D规则：直接返回对应维度的rank（或反向rank）
- 2D规则：返回两个维度的rank之和

#### `get_label(wine1_loc, wine2_loc, rule_vector, grid_size=4)`
计算标签：
- `value1 > value2` → 0 (Wine1胜)
- `value1 < value2` → 1 (Wine2胜)
- `value1 == value2` → 2 (平局)

#### `MetaTaskGeneratorV2.generate_task()`
生成新任务：
1. 创建随机4x4 grid
2. 为每个1D规则生成支持集样本
3. 为每个2D规则生成查询集样本

#### `SequentialRNNV2`
- 包装RNN模型以处理序列输入
- 添加规则向量embedding层（4维 → state_dim）
- 使用规则向量替代context

#### `meta_train_v2()`
Meta-training流程：
1. 生成任务批次
2. 对每个任务：
   - 处理支持集（适应hidden state）
   - 处理查询集（评估2D泛化）
   - 计算meta-loss
3. 更新模型权重

## 📊 数据流

```
任务生成
├── 创建随机4x4 grid
├── 支持集生成
│   ├── Sweet规则: n_support_per_rule个样本
│   ├── Dry规则: n_support_per_rule个样本
│   ├── Light规则: n_support_per_rule个样本
│   └── Full规则: n_support_per_rule个样本
└── 查询集生成
    ├── Sweet+Light规则: 多个样本
    ├── Sweet+Full规则: 多个样本
    ├── Dry+Light规则: 多个样本
    └── Dry+Full规则: 多个样本

Meta-Training
├── 对每个任务:
│   ├── In-Context Learning (支持集)
│   │   └── 适应hidden state，学习1D规则
│   └── In-Context Testing (查询集)
│       └── 测试2D泛化能力
└── Meta-Update: 根据查询集loss更新权重
```

## 🎯 关键特性

### 1. 规则向量embedding
- 使用`nn.Linear(4, state_dim)`将4维规则向量映射到state_dim
- 替代原来的context embedding

### 2. 3类输出
- 模型输出维度改为3
- Loss函数使用`CrossEntropyLoss`（支持3类）

### 3. 属性值计算
- **Sweet**: `wine_loc[0]` (Context 0的rank)
- **Dry**: `grid_size - 1 - wine_loc[0]` (Sweet的反向)
- **Light**: `wine_loc[1]` (Context 1的rank)
- **Full**: `grid_size - 1 - wine_loc[1]` (Light的反向)
- **2D规则**: 两个维度的值相加

### 4. 平局处理
- 当两个wine的属性值相等时，标签为2（平局）
- 这模拟了真实场景中的平局情况

## 📝 使用示例

```python
from meta_learning_v2 import meta_train_v2, meta_test_v2, create_meta_learning_args
from models import get_model

# 创建参数
args = Args()
args.output_dim = 3  # 3类输出
args.n_support_per_rule = 16
args.n_query = 32

# 创建模型
model = get_model(args)

# Meta-training
meta_trained_model, meta_losses = meta_train_v2(
    model, args,
    n_meta_iterations=10000,
    n_tasks_per_batch=4
)

# Meta-testing
final_acc, accuracies = meta_test_v2(
    meta_trained_model, args,
    n_test_tasks=20
)
```

## ⚠️ 注意事项

1. **模型输出维度**: 需要确保模型输出维度为3（在`meta_train_v2`中会自动修改）
2. **Grid引用**: 需要在`SequentialRNNV2`中存储grid引用以访问`idx2tensor`
3. **图像vs索引**: 根据`use_images`参数正确处理wine embeddings
4. **规则向量**: 确保规则向量是4维的`[sweet, dry, light, full]`

## 🔄 与旧版本的对比

| 特性 | 旧版本 | 新版本 V2 |
|------|--------|-----------|
| 输入 | `(ctx, f1, f2)` | `(rule_vector, wine_id1, wine_id2)` |
| Context | 0或1的整数 | 4维规则向量 |
| 输出 | 2类 | 3类（包含平局） |
| 支持集 | 1D规则（rank_diff=1） | 1D规则（4个规则） |
| 查询集 | Incongruent样本 | 2D规则（4个规则） |
| 标签计算 | 基于rank差 | 基于属性值比较 |

## 📂 文件位置

- 新实现: `meta_learning_v2.py`
- 旧实现: `meta_learning.py` (保留作为参考)

