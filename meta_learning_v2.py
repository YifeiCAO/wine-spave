"""
Meta-Learning Implementation V2: 使用规则向量和3类输出

新的设计：
- 支持集：1D规则（Sweet, Dry, Light, Full）
- 查询集：2D规则（Sweet+Light, Sweet+Full, Dry+Light, Dry+Full）
- 输入格式：(rule_vector, wine_id1, wine_id2)
- 输出格式：3类 [Wine1胜, Wine2胜, 平局]
"""

import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from itertools import permutations

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

from data import GridDataGenerator
from models import get_model


# 定义规则向量
# 1D规则
RULE_SWEET = [1, 0, 0, 0]  # Sweet
RULE_DRY = [0, 1, 0, 0]    # Dry (Sweet的反向)
RULE_LIGHT = [0, 0, 1, 0]  # Light
RULE_FULL = [0, 0, 0, 1]   # Full (Light的反向)

# 2D规则
RULE_SWEET_LIGHT = [1, 0, 1, 0]   # Sweet + Light
RULE_SWEET_FULL = [1, 0, 0, 1]    # Sweet + Full
RULE_DRY_LIGHT = [0, 1, 1, 0]     # Dry + Light
RULE_DRY_FULL = [0, 1, 0, 1]       # Dry + Full

# 所有1D规则
ONE_D_RULES = [RULE_SWEET, RULE_DRY, RULE_LIGHT, RULE_FULL]

# 所有2D规则
TWO_D_RULES = [RULE_SWEET_LIGHT, RULE_SWEET_FULL, RULE_DRY_LIGHT, RULE_DRY_FULL]


def get_wine_attribute_value(wine_loc, rule_vector, grid_size=4):
    """
    根据规则向量获取wine的属性值
    
    Args:
        wine_loc: (x, y) 位置，x是Context 0的rank，y是Context 1的rank
        rule_vector: [sweet, dry, light, full] 规则向量
        grid_size: 网格大小（默认4）
    
    Returns:
        属性值（用于比较）
    """
    sweet, dry, light, full = rule_vector
    
    # Sweet: 使用Context 0的rank
    if sweet == 1:
        return wine_loc[0]
    
    # Dry: Sweet的反向，使用 (grid_size - 1 - Context 0的rank)
    if dry == 1:
        return (grid_size - 1) - wine_loc[0]
    
    # Light: 使用Context 1的rank
    if light == 1:
        return wine_loc[1]
    
    # Full: Light的反向，使用 (grid_size - 1 - Context 1的rank)
    if full == 1:
        return (grid_size - 1) - wine_loc[1]
    
    # 2D规则：组合两个维度
    if sum(rule_vector) == 2:
        value = 0
        if sweet == 1:
            value += wine_loc[0]
        if dry == 1:
            value += (grid_size - 1) - wine_loc[0]
        if light == 1:
            value += wine_loc[1]
        if full == 1:
            value += (grid_size - 1) - wine_loc[1]
        return value
    
    return 0


def get_label(wine1_loc, wine2_loc, rule_vector, grid_size=4):
    """
    根据规则向量和两个wine的位置计算标签
    
    Args:
        wine1_loc: (x1, y1) wine1的位置
        wine2_loc: (x2, y2) wine2的位置
        rule_vector: [sweet, dry, light, full] 规则向量
        grid_size: 网格大小
    
    Returns:
        标签: 0=Wine1胜, 1=Wine2胜, 2=平局
    """
    value1 = get_wine_attribute_value(wine1_loc, rule_vector, grid_size)
    value2 = get_wine_attribute_value(wine2_loc, rule_vector, grid_size)
    
    if value1 > value2:
        return 0  # Wine1胜
    elif value1 < value2:
        return 1  # Wine2胜
    else:
        return 2  # 平局


class MetaTaskV2:
    """
    表示一个meta-learning任务/episode
    每个任务包含：
    - 一个随机生成的4x4 wine空间（grid）
    - Support Set: 1D规则样本
    - Query Set: 2D规则样本
    """
    def __init__(self, grid, support_set, query_set):
        self.grid = grid
        self.support_set = support_set  # List of (rule_vector, wine_id1, wine_id2, label) tuples
        self.query_set = query_set      # List of (rule_vector, wine_id1, wine_id2, label) tuples


class MetaTaskGeneratorV2:
    """
    生成新的meta-learning任务
    每个任务是一个新的随机4x4 wine空间，包含1D支持集和2D查询集
    """
    def __init__(self, args, n_support_per_rule=16, n_query=32):
        self.args = args
        self.n_support_per_rule = n_support_per_rule  # 每个1D规则的样本数
        self.n_query = n_query  # 查询集样本数
        self.grid_size = args.grid_size
        
    def generate_task(self):
        """
        生成一个新的meta-learning任务：
        1. 创建新的随机4x4 wine空间
        2. 生成支持集（1D规则）
        3. 生成查询集（2D规则）
        """
        # 创建新的随机grid
        grid = GridDataGenerator(
            training_regime=self.args.training_regime,
            size=self.args.grid_size,
            use_images=self.args.use_images,
            image_dir=self.args.image_dir,
            inner_4x4=self.args.inner_4x4
        )
        
        # 生成支持集：1D规则
        support_set = []
        
        # 为每个1D规则生成样本
        for rule_vector in ONE_D_RULES:
            rule_samples = []
            
            # 生成所有可能的wine pair
            for wine_id1, wine_id2 in permutations(range(16), 2):
                loc1 = grid.idx2loc[wine_id1]
                loc2 = grid.idx2loc[wine_id2]
                
                # 计算标签
                label = get_label(loc1, loc2, rule_vector, self.grid_size)
                
                # 添加样本
                rule_samples.append((rule_vector, wine_id1, wine_id2, label))
            
            # 随机采样n_support_per_rule个样本
            if len(rule_samples) > self.n_support_per_rule:
                rule_samples = random.sample(rule_samples, self.n_support_per_rule)
            
            support_set.extend(rule_samples)
        
        # 生成查询集：2D规则
        query_set = []
        
        # 为每个2D规则生成样本
        for rule_vector in TWO_D_RULES:
            rule_samples = []
            
            # 生成所有可能的wine pair
            for wine_id1, wine_id2 in permutations(range(16), 2):
                loc1 = grid.idx2loc[wine_id1]
                loc2 = grid.idx2loc[wine_id2]
                
                # 计算标签
                label = get_label(loc1, loc2, rule_vector, self.grid_size)
                
                # 添加样本
                rule_samples.append((rule_vector, wine_id1, wine_id2, label))
            
            query_set.extend(rule_samples)
        
        # 如果查询集样本数 > n_query，随机采样
        if len(query_set) > self.n_query:
            query_set = random.sample(query_set, self.n_query)
        
        return MetaTaskV2(grid, support_set, query_set)


class SequentialRNNV2(nn.Module):
    """
    Wrapper around RNN to handle sequential input for meta-learning V2
    使用规则向量而不是context
    """
    def __init__(self, base_rnn):
        super(SequentialRNNV2, self).__init__()
        self.base_rnn = base_rnn
        self.hidden_dim = base_rnn.hidden_dim
        
        # 规则向量embedding层应该在base_rnn中
        # 如果不存在，说明应该在meta_train_v2中已经创建了，这里不应该创建新的
        if not hasattr(base_rnn, 'rule_embedding'):
            raise RuntimeError("rule_embedding should be added to model before creating SequentialRNNV2!")
        
        # 直接使用base_rnn中的rule_embedding（不创建新引用）
        # 这样确保训练和测试使用同一个实例
        
    def forward_sequence(self, samples, hidden_state=None):
        """
        处理样本序列，保持hidden state
        
        Args:
            samples: List of (rule_vector, wine_id1, wine_id2, label) tuples
            hidden_state: 初始hidden state (h_n, c_n) 或 None
        
        Returns:
            outputs: 每个样本的模型输出列表
            final_hidden: 处理完所有样本后的最终hidden state
        """
        outputs = []
        device = next(self.base_rnn.parameters()).device
        
        # 初始化hidden state
        if hidden_state is None:
            h_n = torch.zeros(1, 1, self.hidden_dim).to(device)
            c_n = torch.zeros(1, 1, self.hidden_dim).to(device)
        else:
            h_n, c_n = hidden_state
        
        # 处理每个样本
        for sample in samples:
            rule_vector, wine_id1, wine_id2, label = sample
            
            # 转换为tensor
            if isinstance(rule_vector, list):
                rule_vector = torch.tensor(rule_vector, dtype=torch.float32).to(device)
            if isinstance(wine_id1, int):
                wine_id1 = torch.tensor([wine_id1]).to(device)
            if isinstance(wine_id2, int):
                wine_id2 = torch.tensor([wine_id2]).to(device)
            
            # 处理batch维度
            if rule_vector.dim() == 1:
                rule_vector = rule_vector.unsqueeze(0)  # [1, 4]
            if wine_id1.dim() == 0:
                wine_id1 = wine_id1.unsqueeze(0)
            if wine_id2.dim() == 0:
                wine_id2 = wine_id2.unsqueeze(0)
            
            batch_size = rule_vector.shape[0]
            
            # 获取embeddings（使用base_rnn中的rule_embedding）
            rule_embed = self.base_rnn.rule_embedding(rule_vector)  # [batch, state_dim]
            
            # 获取wine embeddings
            # 从grid获取idx2tensor（在meta_train_v2中设置）
            grid = getattr(self.base_rnn, 'grid', None)
            if grid is not None:
                f1_tensor = grid.idx2tensor[wine_id1.item()]
                f2_tensor = grid.idx2tensor[wine_id2.item()]
            else:
                # 如果没有grid，直接使用wine_id（假设是索引）
                f1_tensor = wine_id1
                f2_tensor = wine_id2
            
            # 处理embeddings
            if self.base_rnn.use_images:
                # 图像：使用CNN处理图像tensor
                if isinstance(f1_tensor, torch.Tensor) and len(f1_tensor.shape) >= 3:
                    # 已经是图像tensor，添加batch维度
                    f1_embed = self.base_rnn.face_embedding(f1_tensor.unsqueeze(0))  # [1, state_dim]
                    f2_embed = self.base_rnn.face_embedding(f2_tensor.unsqueeze(0))  # [1, state_dim]
                else:
                    # 需要从grid获取图像
                    f1_embed = self.base_rnn.face_embedding(f1_tensor.unsqueeze(0))
                    f2_embed = self.base_rnn.face_embedding(f2_tensor.unsqueeze(0))
            else:
                # 非图像：直接使用wine_id作为索引
                f1_embed = self.base_rnn.face_embedding(wine_id1)  # [batch, state_dim]
                f2_embed = self.base_rnn.face_embedding(wine_id2)  # [batch, state_dim]
            
            # 处理序列维度
            if self.base_rnn.use_images:
                f1_embed = f1_embed.unsqueeze(0)  # [1, batch, state_dim]
                f2_embed = f2_embed.unsqueeze(0)
            else:
                f1_embed = f1_embed.unsqueeze(0)  # [1, batch, state_dim]
                f2_embed = f2_embed.unsqueeze(0)
            
            rule_embed = rule_embed.unsqueeze(0)  # [1, batch, state_dim]
            
            # Scale rule embedding (类似ctx_scale)
            rule_embed = torch.tensor(self.base_rnn.ctx_scale, device=device) * rule_embed
            
            # 确定顺序（类似ctx_order）
            if self.base_rnn.ctx_order == 'first':
                x = torch.cat([rule_embed, f1_embed, f2_embed], dim=0)  # [3, batch, state_dim]
            else:
                x = torch.cat([f1_embed, f2_embed, rule_embed], dim=0)  # [3, batch, state_dim]
            
            # 通过LSTM处理（保持hidden state）
            lstm_out, (h_n, c_n) = self.base_rnn.lstm(x, (h_n, c_n))
            
            # 从最终hidden state获取输出
            output = self.base_rnn.out(h_n.squeeze(0))  # [batch, output_dim]
            outputs.append(output)
        
        return outputs, (h_n, c_n)


def meta_train_v2(model, args, n_meta_iterations=10000, n_tasks_per_batch=4):
    """
    Meta-Training V2: 使用规则向量和3类输出
    
    Process:
    1. 初始化RNN权重θ
    2. 对于每次meta-iteration:
       a. 采样一批新任务
       b. 对于每个任务:
          - In-context learning: 处理1D支持集（适应hidden state）
          - In-context testing: 处理2D查询集（评估2D泛化）
       c. 计算meta-loss（基于查询集）
       d. Meta-update: 根据meta-loss更新θ
    
    Args:
        model: RNN模型
        args: 参数对象
        n_meta_iterations: Meta-training迭代次数
        n_tasks_per_batch: 每批任务数
    """
    device = args.device
    model = model.to(device)
    model.train()
    
    # 修改模型输出为3类
    if model.output_dim != 3:
        model.output_dim = 3
        model.out = nn.Linear(model.hidden_dim, 3).to(device)
        print(f"修改模型输出维度为3类")
    
    # 确保模型有rule_embedding层（如果还没有）
    if not hasattr(model, 'rule_embedding'):
        model.rule_embedding = nn.Linear(4, model.state_dim).to(device)
        nn.init.xavier_normal_(model.rule_embedding.weight)
        nn.init.zeros_(model.rule_embedding.bias)
        print(f"添加规则向量embedding层到模型")
    
    # Wrap model for sequential processing
    seq_model = SequentialRNNV2(model)
    seq_model = seq_model.to(device)
    
    # 创建任务生成器
    n_support_per_rule = getattr(args, 'n_support_per_rule', 16)
    n_query = getattr(args, 'n_query', 32)
    task_generator = MetaTaskGeneratorV2(args, n_support_per_rule=n_support_per_rule, n_query=n_query)
    
    # Optimizer for meta-updates
    # 重要：需要包含model的所有参数（包括rule_embedding层）
    # 验证rule_embedding是否在model中
    if not hasattr(model, 'rule_embedding'):
        raise RuntimeError("rule_embedding should have been added to model!")
    
    # 检查参数数量（用于调试）
    param_count = sum(p.numel() for p in model.parameters())
    rule_embedding_params = sum(p.numel() for p in model.rule_embedding.parameters())
    print(f"  模型总参数数: {param_count}")
    print(f"  rule_embedding参数数: {rule_embedding_params} (应该是160: 4*32+32)")
    
    # 使用model.parameters()，这会自动包含rule_embedding（PyTorch会自动注册）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr if hasattr(args, 'meta_lr') else 0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"开始Meta-Training V2...")
    print(f"  总迭代次数: {n_meta_iterations}")
    print(f"  每批任务数: {n_tasks_per_batch}")
    print(f"  每个1D规则的Support样本数: {n_support_per_rule}")
    print(f"  Query Set大小: {n_query}")
    print(f"  输出类别数: 3 (Wine1胜, Wine2胜, 平局)\n")
    
    meta_losses = []
    
    # 创建进度条
    pbar = tqdm(range(n_meta_iterations), 
                desc="Meta-Training V2", 
                unit="iter",
                ncols=100 if HAS_TQDM else None)
    
    for meta_iter in pbar:
        # 采样一批任务
        tasks = [task_generator.generate_task() for _ in range(n_tasks_per_batch)]
        
        total_meta_loss = 0.0
        
        # 处理每个任务
        for task in tasks:
            # 准备支持集和查询集样本
            support_samples = task.support_set
            query_samples = task.query_set
            
            # 存储grid引用以便访问idx2tensor
            seq_model.base_rnn.grid = task.grid
            
            # In-context learning: 处理支持集
            support_outputs, adapted_hidden = seq_model.forward_sequence(support_samples)
            
            # In-context testing: 处理查询集（使用适应后的hidden state）
            query_outputs, _ = seq_model.forward_sequence(query_samples, adapted_hidden)
            
            # 计算查询集的loss（meta-loss）
            if len(query_outputs) > 0:
                query_preds = torch.cat(query_outputs, dim=0)
                query_labels = torch.tensor([label for _, _, _, label in query_samples]).to(device)
                
                task_loss = loss_fn(query_preds, query_labels)
                total_meta_loss += task_loss
            else:
                continue  # 跳过如果没有查询样本
        
        # 平均meta-loss
        avg_meta_loss = total_meta_loss / n_tasks_per_batch
        
        # Meta-update: 更新模型权重θ
        optimizer.zero_grad()
        avg_meta_loss.backward()
        optimizer.step()
        
        meta_losses.append(avg_meta_loss.item())
        
        # 更新进度条
        current_loss = avg_meta_loss.item()
        if len(meta_losses) > 1:
            window = min(10, len(meta_losses))
            avg_loss = sum(meta_losses[-window:]) / window
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Avg': f'{avg_loss:.4f}',
                'Min': f'{min(meta_losses):.4f}'
            })
        else:
            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # 每100次迭代打印详细信息
        if (meta_iter + 1) % 100 == 0:
            window = min(100, len(meta_losses))
            avg_loss = sum(meta_losses[-window:]) / window
            if HAS_TQDM:
                pbar.write(f"Meta-Iteration {meta_iter+1}/{n_meta_iterations}, "
                          f"Current Loss: {current_loss:.4f}, "
                          f"Avg Loss (last {window}): {avg_loss:.4f}")
            else:
                print(f"Meta-Iteration {meta_iter+1}/{n_meta_iterations}, "
                      f"Current Loss: {current_loss:.4f}, "
                      f"Avg Loss (last {window}): {avg_loss:.4f}")
    
    # 关闭进度条
    if HAS_TQDM:
        pbar.close()
    
    print(f"\nMeta-Training V2完成!")
    print(f"  最终Meta-Loss: {meta_losses[-1]:.4f}")
    
    return model, meta_losses


def meta_test_v2(model, args, n_test_tasks=10):
    """
    简化的Meta-Testing V2: 直接测试1D到2D的泛化能力
    
    Args:
        model: Meta-trained RNN模型
        args: 参数对象
        n_test_tasks: 测试任务数
    
    Returns:
        final_accuracy: 最终平均准确率
        all_accuracies: 每个任务的准确率列表
        rule_results: 按规则类型统计的结果
        rank_diff_results: 按rank difference统计的结果
    """
    device = args.device
    model = model.to(device)
    model.eval()  # 冻结权重
    
    # 确保模型有rule_embedding层（应该已经有了，从训练中继承）
    if not hasattr(model, 'rule_embedding'):
        print("⚠ 警告: 模型没有rule_embedding层，将创建新的（但权重未训练）")
        model.rule_embedding = nn.Linear(4, model.state_dim).to(device)
        nn.init.xavier_normal_(model.rule_embedding.weight)
        nn.init.zeros_(model.rule_embedding.bias)
    
    # Wrap model for sequential processing
    seq_model = SequentialRNNV2(model)
    seq_model = seq_model.to(device)
    
    print(f"\n开始Meta-Testing V2...")
    print(f"  模型权重已冻结")
    print(f"  测试任务数: {n_test_tasks}")
    print(f"  核心测试: 1D → 2D 泛化能力\n")
    
    # 使用args中的参数
    n_support_per_rule = getattr(args, 'n_support_per_rule', 16)
    n_query = getattr(args, 'n_query', 32)
    task_generator = MetaTaskGeneratorV2(args, n_support_per_rule=n_support_per_rule, n_query=n_query)
    
    all_accuracies = []
    # 详细结果：按规则类型统计
    rule_results = {}  # {rule_name: {'correct': int, 'total': int, 'accuracies': list}}
    # 详细结果：按rank difference统计
    rank_diff_results = {}  # {rank_diff: {'correct': int, 'total': int, 'accuracies': list}}
    
    # 规则名称映射
    def get_rule_name(rule_vector):
        # 转换为列表进行比较（如果是tensor）
        if isinstance(rule_vector, torch.Tensor):
            rule_vector = rule_vector.cpu().tolist()
        elif not isinstance(rule_vector, list):
            rule_vector = list(rule_vector)
        
        # 比较规则向量
        if rule_vector == RULE_SWEET:
            return "Sweet"
        elif rule_vector == RULE_DRY:
            return "Dry"
        elif rule_vector == RULE_LIGHT:
            return "Light"
        elif rule_vector == RULE_FULL:
            return "Full"
        elif rule_vector == RULE_SWEET_LIGHT:
            return "Sweet+Light"
        elif rule_vector == RULE_SWEET_FULL:
            return "Sweet+Full"
        elif rule_vector == RULE_DRY_LIGHT:
            return "Dry+Light"
        elif rule_vector == RULE_DRY_FULL:
            return "Dry+Full"
        else:
            return str(rule_vector)
    
    for task_idx in range(n_test_tasks):
        # 生成测试任务
        task = task_generator.generate_task()
        
        # 存储grid引用
        seq_model.base_rnn.grid = task.grid
        
        support_samples = task.support_set
        query_samples = task.query_set
        
        # 适应: 处理支持集
        with torch.no_grad():
            support_outputs, adapted_hidden = seq_model.forward_sequence(support_samples)
        
        # 测试: 处理查询集
        with torch.no_grad():
            query_outputs, _ = seq_model.forward_sequence(query_samples, adapted_hidden)
            
            if len(query_outputs) > 0:
                query_preds = torch.cat(query_outputs, dim=0)
                query_labels = torch.tensor([label for _, _, _, label in query_samples]).to(device)
                
                # 计算准确率
                preds = torch.argmax(query_preds, dim=1)
                correct = (preds == query_labels).float()
                accuracy = correct.mean().item()
                all_accuracies.append(accuracy)
                
                # 按规则类型和rank difference统计
                for i, (rule_vector, wine_id1, wine_id2, label) in enumerate(query_samples):
                    # 规则类型统计
                    rule_name = get_rule_name(rule_vector)
                    if rule_name not in rule_results:
                        rule_results[rule_name] = {'correct': 0, 'total': 0, 'accuracies': []}
                    
                    rule_results[rule_name]['total'] += 1
                    is_correct = (preds[i].item() == label)
                    if is_correct:
                        rule_results[rule_name]['correct'] += 1
                    rule_results[rule_name]['accuracies'].append(1.0 if is_correct else 0.0)
                    
                    # Rank difference统计
                    # 获取wine的位置
                    loc1 = task.grid.idx2loc[wine_id1]
                    loc2 = task.grid.idx2loc[wine_id2]
                    
                    # 计算rank difference（在规则对应的维度上）
                    grid_size = args.grid_size
                    value1 = get_wine_attribute_value(loc1, rule_vector, grid_size)
                    value2 = get_wine_attribute_value(loc2, rule_vector, grid_size)
                    rank_diff = abs(value1 - value2)
                    
                    if rank_diff not in rank_diff_results:
                        rank_diff_results[rank_diff] = {'correct': 0, 'total': 0, 'accuracies': []}
                    
                    rank_diff_results[rank_diff]['total'] += 1
                    if is_correct:
                        rank_diff_results[rank_diff]['correct'] += 1
                    rank_diff_results[rank_diff]['accuracies'].append(1.0 if is_correct else 0.0)
            else:
                continue
        
        if (task_idx + 1) % 5 == 0:
            avg_acc = sum(all_accuracies) / len(all_accuracies)
            print(f"  任务 {task_idx+1}/{n_test_tasks}: 当前平均准确率 = {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    
    final_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
    print(f"\n✓ Meta-Testing V2完成!")
    print(f"  最终平均准确率: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"  测试任务数: {len(all_accuracies)}")
    print(f"  随机猜测基线: 33.33% (3选1)")
    
    if final_accuracy > 0.50:
        print(f"  ✓ 成功！准确率超过随机基线")
    else:
        print(f"  ⚠ 需要改进，准确率接近随机水平")
    
    return final_accuracy, all_accuracies, rule_results, rank_diff_results

