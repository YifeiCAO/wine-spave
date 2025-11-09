# Meta-Learning 完整流程和数据详解

## 1. 数据生成流程（ungrouped regime）

### 1.1 基础数据结构
- **4x4 Grid**: 16个wine，每个wine有2个维度的rank
  - 例如：wine0的位置是(0,0)，表示在Context 0中rank=0，在Context 1中rank=0
  - wine1的位置是(0,1)，表示在Context 0中rank=0，在Context 1中rank=1

### 1.2 样本生成逻辑（generate_ungrouped_samples）
对于每个wine pair (idx1, idx2)：
- 对于每个context (0或1)：
  - 计算在当前context下的rank差：`d = r1 - r2`
  - 如果 `d != 0`：
    - 生成样本：`(ctx, loc1, loc2, y)`
    - `y = int(d > 0)`：如果r1 > r2，y=1（wine1更好），否则y=0（wine2更好）
    - **关键**：这个y是基于**当前context**的rank差计算的
    - 如果 `abs(d) == 1`：放入`train`（相邻pair，1-level difference）
    - 如果 `abs(d) > 1`：放入`test`（非相邻pair，需要泛化）

### 1.3 Congruency（一致性）计算
对于样本 `(ctx, loc1, loc2, y)`：
- `loc1 = (x1, y1)`：wine1在Context 0的rank=x1，在Context 1的rank=y1
- `loc2 = (x2, y2)`：wine2在Context 0的rank=x2，在Context 1的rank=y2

**Cong计算**：
```python
if (x1==x2) or (y1==y2):
    cong = 0  # 某个维度相同，中性
else:
    cong = 1 if (x1<x2) == (y1<y2) else -1
```

**含义**：
- `cong = 1`（一致）：两个维度方向相同
  - 例如：wine1在两个维度上都比wine2高，或都低
  - `(x1<x2) == (y1<y2)`：两个维度都指向同一方向
- `cong = -1`（不一致）：两个维度方向相反
  - 例如：wine1在Context 0上比wine2高，但在Context 1上比wine2低
  - `(x1<x2) != (y1<y2)`：两个维度指向相反方向
- `cong = 0`（中性）：某个维度相同
  - `x1==x2`或`y1==y2`：在某个维度上rank相同

## 2. Support Set（支持集）生成

### 2.1 来源
- 从`grid.train`中筛选

### 2.2 筛选条件
1. `cong != -1`：排除incongruent样本（排除需要2D推理的样本）
2. 在当前context下，`abs(rank1 - rank2) == 1`：只包含相邻pair（1-level difference）

### 2.3 含义
- **1D规则**：只关心当前context的rank差
- **相邻pair**：rank差为1，最容易学习的规则
- **排除incongruent**：确保这些样本可以用单一维度判断

**示例**：
- Context 0: wine1=(2,1), wine2=(1,1)
  - rank差 = |2-1| = 1 ✓
  - cong: x1=2, x2=1, y1=1, y2=1 → y1==y2 → cong=0 ✓
  - 这是1D规则：只关心Context 0，两个wine在Context 1上相同

## 3. Query Set（查询集）生成

### 3.1 来源
- 从`grid.test`中筛选

### 3.2 筛选条件
- `cong == -1`：只包含incongruent样本（不一致样本）

### 3.3 关键理解：为什么incongruent需要2D推理？

**重要**：虽然样本格式是`(ctx, loc1, loc2, y)`，但`y`是基于**当前context**计算的。

**示例1**：Context 0的样本
- wine1=(2,1), wine2=(1,2)
- 在Context 0上：rank差 = 2-1 = 1 → y=1（wine1更好）
- 在Context 1上：rank差 = 1-2 = -1 → 如果只看Context 1，wine2更好
- cong: x1=2>x2=1, y1=1<y2=2 → (x1<x2) != (y1<y2) → cong=-1
- **问题**：虽然y是基于Context 0计算的，但两个维度给出相反信号
- **2D推理**：模型需要理解"在Context 0上，即使Context 1给出相反信号，也要遵循Context 0的规则"

**示例2**：Context 1的样本
- wine1=(1,3), wine2=(2,1)
- 在Context 1上：rank差 = 3-1 = 2 → y=1（wine1更好）
- 在Context 0上：rank差 = 1-2 = -1 → 如果只看Context 0，wine2更好
- cong: x1=1<x2=2, y1=3>y2=1 → (x1<x2) != (y1<y2) → cong=-1
- **2D推理**：模型需要理解"在Context 1上，即使Context 0给出相反信号，也要遵循Context 1的规则"

### 3.4 为什么Query Set只用incongruent？

**核心思想**：
- Support Set：学习1D规则（只关心当前context，忽略另一个维度）
- Query Set：测试2D组合推理（在当前context下，即使另一个维度给出相反信号，也要遵循当前context的规则）

**这不是"组合两个维度"**，而是：
- **在给定context下，忽略另一个维度的干扰信号**
- **坚持当前context的规则，即使另一个维度指向相反方向**

## 4. Meta-Learning流程

### 4.1 Meta-Training（外循环）
对于每次meta-iteration：
1. 生成一批新任务（每个任务是一个新的4x4 grid）
2. 对于每个任务：
   - **Support Set**：1D规则（相邻pair，cong!=1）
     - 模型通过hidden state学习"在当前context下，rank高的更好"
   - **Query Set**：2D规则（incongruent，cong=-1）
     - 模型需要在另一个维度给出相反信号时，仍然遵循当前context的规则
   - 计算query set的loss（meta-loss）
3. 根据meta-loss更新模型权重

### 4.2 Meta-Testing（内循环）
1. 冻结模型权重
2. 对于每个测试任务：
   - 用Support Set适应hidden state
   - 在Query Set上测试
   - 计算准确率

## 5. 关键理解

### 5.1 1D规则 vs 2D规则
- **1D规则**：只关心当前context的rank，另一个维度相同或一致
- **2D规则**：在当前context下，另一个维度给出相反信号，需要忽略干扰

### 5.2 为什么100%准确率可能有问题？
如果Query Set只包含incongruent样本，且这些样本的y都是基于当前context计算的，那么：
- 模型只需要学习"在当前context下，rank高的更好"
- 即使另一个维度给出相反信号，y仍然是基于当前context的
- 所以模型可能只是学会了"忽略另一个维度，只看当前context"

**这不是真正的"2D组合推理"**，而是"在干扰下坚持1D规则"。

### 5.3 真正的2D组合推理应该是什么？
如果要做真正的2D组合推理，应该：
- Query Set包含需要**同时考虑两个维度**的样本
- 例如：wine1=(2,1), wine2=(1,2)
  - 在Context 0上：wine1更好
  - 在Context 1上：wine2更好
  - **需要组合判断**：哪个wine更好？
  - 这需要定义组合规则（如：两个维度加权平均，或优先某个维度）

**但当前的实现中，y是基于当前context计算的，所以不是真正的2D组合推理。**

