"""
Meta-Learning Implementation for Country Task

This module implements In-Context Meta-Learning for training RNNs to be "fast learners"
that can generalize from 1D rules to 2D combinations, simulating human compositional generalization.

Key concepts:
- Meta-Training (Outer Loop): Train RNN to learn "how to learn"
- Meta-Testing (Inner Loop): Test RNN on real country tasks with zero-shot 2D generalization
"""

import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from itertools import cycle

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 如果没有tqdm，创建一个简单的替代
    def tqdm(iterable, **kwargs):
        return iterable

from data import GridDataGenerator
from models import get_model
from country_task import CountryTaskDataGenerator


class MetaTask:
    """
    Represents a single meta-learning task/episode.
    Each task contains:
    - A new 4x4 wine space (grid)
    - Support set: 1D rule samples (for adaptation)
    - Query set: 2D rule samples (for evaluation)
    """
    def __init__(self, grid, support_set, query_set):
        self.grid = grid
        self.support_set = support_set  # List of (ctx, f1, f2, y) tuples, only 1D rules
        self.query_set = query_set      # List of (ctx, f1, f2, y) tuples, 2D rules


class MetaTaskGenerator:
    """
    Generates meta-learning tasks for training.
    Each task is a new random 4x4 wine space with 1D support and 2D query sets.
    """
    def __init__(self, args, n_support=16, n_query=32):
        self.args = args
        self.n_support = n_support  # Number of 1D samples in support set
        self.n_query = n_query      # Number of 2D samples in query set
        
    def generate_task(self):
        """
        Generate a new meta-learning task:
        1. Create a new random 4x4 wine space
        2. Generate support set (1D rules only)
        3. Generate query set (2D rules only)
        """
        # Create a new random grid
        grid = GridDataGenerator(
            training_regime=self.args.training_regime,
            size=self.args.grid_size,
            use_images=self.args.use_images,
            image_dir=self.args.image_dir,
            inner_4x4=self.args.inner_4x4
        )
        
        # Support set: Only 1D rules (context 0 or context 1, but not both)
        support_set = []
        # Sample from training data, but only keep 1D samples
        # 1D samples: samples where one dimension is the same (cong=0) or both point same direction
        for sample in grid.train:
            ctx, loc1, loc2, y, info = sample
            # Only include samples where the rule is effectively 1D
            # (i.e., one dimension dominates, or both dimensions agree)
            cong = info.get('cong', 0)
            if cong != -1:  # Exclude incongruent samples (they require 2D reasoning)
                support_set.append(sample)
        
        # Randomly sample n_support samples
        if len(support_set) > self.n_support:
            support_set = random.sample(support_set, self.n_support)
        
        # Query set: 2D rules (both dimensions matter)
        query_set = []
        # Generate 2D samples: samples where both dimensions differ
        # We'll create samples that require 2D reasoning
        for sample in grid.test:
            ctx, loc1, loc2, y, info = sample
            cong = info.get('cong', -1)  # Incongruent samples require 2D reasoning
            if cong == -1:  # Only include incongruent samples
                query_set.append(sample)
        
        # Randomly sample n_query samples
        if len(query_set) > self.n_query:
            query_set = random.sample(query_set, self.n_query)
        
        return MetaTask(grid, support_set, query_set)


class SequentialRNN(nn.Module):
    """
    Wrapper around RNN to handle sequential input for meta-learning.
    This allows the RNN to process a sequence of samples while maintaining hidden state.
    """
    def __init__(self, base_rnn):
        super(SequentialRNN, self).__init__()
        self.base_rnn = base_rnn
        self.hidden_dim = base_rnn.hidden_dim
        
    def forward_sequence(self, samples, hidden_state=None):
        """
        Process a sequence of samples, maintaining hidden state across samples.
        
        Args:
            samples: List of (ctx, f1, f2, y) tuples
            hidden_state: Initial hidden state (h_n, c_n) or None
        
        Returns:
            outputs: List of model outputs for each sample
            final_hidden: Final hidden state after processing all samples
        """
        outputs = []
        device = next(self.base_rnn.parameters()).device
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h_n = torch.zeros(1, 1, self.hidden_dim).to(device)
            c_n = torch.zeros(1, 1, self.hidden_dim).to(device)
        else:
            h_n, c_n = hidden_state
        
        # Process each sample in sequence
        for sample in samples:
            ctx, f1, f2, y = sample
            
            # Handle batch dimension
            if ctx.dim() == 0:
                ctx = ctx.unsqueeze(0)
            if f1.dim() == 0:
                f1 = f1.unsqueeze(0)
            if f2.dim() == 0:
                f2 = f2.unsqueeze(0)
            
            batch_size = ctx.shape[0]
            
            # Get embeddings
            ctx_embed = self.base_rnn.ctx_embedding(ctx)  # [batch, state_dim]
            f1_embed = self.base_rnn.face_embedding(f1)   # [batch, state_dim] or [batch, ...]
            f2_embed = self.base_rnn.face_embedding(f2)   # [batch, state_dim] or [batch, ...]
            
            # Handle image embeddings (CNN output)
            if self.base_rnn.use_images:
                # CNN outputs [batch, state_dim], need to unsqueeze for sequence
                f1_embed = f1_embed.unsqueeze(0)  # [1, batch, state_dim]
                f2_embed = f2_embed.unsqueeze(0)  # [1, batch, state_dim]
            else:
                f1_embed = f1_embed.unsqueeze(0)  # [1, batch, state_dim]
                f2_embed = f2_embed.unsqueeze(0)  # [1, batch, state_dim]
            
            ctx_embed = ctx_embed.unsqueeze(0)  # [1, batch, state_dim]
            
            # Scale context
            ctx_embed = torch.tensor(self.base_rnn.ctx_scale, device=device) * ctx_embed
            
            # Determine order
            if self.base_rnn.ctx_order == 'first':
                x = torch.cat([ctx_embed, f1_embed, f2_embed], dim=0)  # [3, batch, state_dim]
            else:
                x = torch.cat([f1_embed, f2_embed, ctx_embed], dim=0)  # [3, batch, state_dim]
            
            # Process through LSTM (maintaining hidden state)
            # h_n, c_n: [1, batch, hidden_dim]
            lstm_out, (h_n, c_n) = self.base_rnn.lstm(x, (h_n, c_n))
            
            # Get output from final hidden state
            # h_n: [1, batch, hidden_dim] -> [batch, hidden_dim]
            output = self.base_rnn.out(h_n.squeeze(0))  # [batch, output_dim]
            outputs.append(output)
        
        return outputs, (h_n, c_n)


def meta_train(model, args, n_meta_iterations=10000, n_tasks_per_batch=4):
    """
    Meta-Training (Outer Loop): Train RNN to be a "fast learner"
    
    Process:
    1. Initialize RNN weights θ
    2. For each meta-iteration:
       a. Sample a batch of new tasks
       b. For each task:
          - In-context learning: Process 1D support set (adapt hidden state)
          - In-context testing: Process 2D query set (evaluate on 2D)
       c. Calculate meta-loss across all tasks
       d. Meta-update: Update θ based on meta-loss
    
    Args:
        model: RNN model
        args: Arguments object
        n_meta_iterations: Number of meta-training iterations
        n_tasks_per_batch: Number of tasks per meta-batch
    """
    device = args.device
    model = model.to(device)
    model.train()
    
    # Wrap model for sequential processing
    seq_model = SequentialRNN(model)
    seq_model = seq_model.to(device)
    
    # Create task generator
    task_generator = MetaTaskGenerator(args)
    
    # Optimizer for meta-updates
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr if hasattr(args, 'meta_lr') else 0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"开始Meta-Training...")
    print(f"  总迭代次数: {n_meta_iterations}")
    print(f"  每批任务数: {n_tasks_per_batch}")
    print(f"  Support set大小: {task_generator.n_support}")
    print(f"  Query set大小: {task_generator.n_query}\n")
    
    meta_losses = []
    
    # 创建进度条
    pbar = tqdm(range(n_meta_iterations), 
                desc="Meta-Training", 
                unit="iter",
                ncols=100 if HAS_TQDM else None)
    
    for meta_iter in pbar:
        # Sample a batch of tasks
        tasks = [task_generator.generate_task() for _ in range(n_tasks_per_batch)]
        
        total_meta_loss = 0.0
        
        # Process each task
        for task in tasks:
            # Convert samples to tensors
            def prepare_samples(samples):
                ctx_list = []
                f1_list = []
                f2_list = []
                y_list = []
                
                for ctx, loc1, loc2, y, info in samples:
                    ctx_list.append(ctx)
                    f1_idx = task.grid.loc2idx[loc1]
                    f2_idx = task.grid.loc2idx[loc2]
                    f1_tensor = task.grid.idx2tensor[f1_idx]
                    f2_tensor = task.grid.idx2tensor[f2_idx]
                    
                    # Handle different tensor types
                    if args.use_images:
                        # Images are already tensors
                        f1_list.append(f1_tensor)
                        f2_list.append(f2_tensor)
                    else:
                        # Indices need to be converted
                        if isinstance(f1_tensor, torch.Tensor):
                            f1_list.append(f1_tensor)
                            f2_list.append(f2_tensor)
                        else:
                            f1_list.append(torch.tensor(f1_tensor))
                            f2_list.append(torch.tensor(f2_tensor))
                    y_list.append(y)
                
                ctx_tensor = torch.tensor(ctx_list).to(device)
                if args.use_images:
                    f1_tensor = torch.stack(f1_list).to(device)
                    f2_tensor = torch.stack(f2_list).to(device)
                else:
                    # For indices, stack as is
                    f1_tensor = torch.stack(f1_list).to(device)
                    f2_tensor = torch.stack(f2_list).to(device)
                y_tensor = torch.tensor(y_list).to(device)
                
                return list(zip(ctx_tensor, f1_tensor, f2_tensor, y_tensor))
            
            support_samples = prepare_samples(task.support_set)
            query_samples = prepare_samples(task.query_set)
            
            # In-context learning (adaptation): Process support set
            # Hidden state evolves but weights θ remain unchanged
            support_outputs, adapted_hidden = seq_model.forward_sequence(support_samples)
            
            # In-context testing (evaluation): Process query set with adapted hidden state
            query_outputs, _ = seq_model.forward_sequence(query_samples, adapted_hidden)
            
            # Calculate loss on query set (this is the meta-loss for this task)
            if len(query_outputs) > 0:
                query_preds = torch.cat(query_outputs, dim=0)
                query_labels = torch.cat([y.unsqueeze(0) if y.dim() == 0 else y for _, _, _, y in query_samples], dim=0)
            else:
                continue  # Skip if no query samples
            
            task_loss = loss_fn(query_preds, query_labels)
            total_meta_loss += task_loss
        
        # Average meta-loss across tasks
        avg_meta_loss = total_meta_loss / n_tasks_per_batch
        
        # Meta-update: Update model weights θ
        optimizer.zero_grad()
        avg_meta_loss.backward()
        optimizer.step()
        
        meta_losses.append(avg_meta_loss.item())
        
        # 更新进度条显示
        current_loss = avg_meta_loss.item()
        if len(meta_losses) > 1:
            # 计算移动平均（最近10次）
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
    
    print(f"\nMeta-Training完成!")
    print(f"  最终Meta-Loss: {meta_losses[-1]:.4f}")
    
    return model, meta_losses


def meta_test(model, args, grid_data_gen, country_data_gen, country_loader):
    """
    Meta-Testing: Evaluate meta-trained RNN on real country tasks
    
    Process:
    1. Freeze model weights θ_final
    2. For each country (real task):
       a. Adaptation: Process 1D support set (simulate human learning)
       b. Evaluation: Process 2D query set (simulate human testing)
    3. Compare RNN accuracy with human baseline
    
    Args:
        model: Meta-trained RNN model
        args: Arguments object
        grid_data_gen: GridDataGenerator for the specific 4x4 space
        country_data_gen: CountryTaskDataGenerator
        country_loader: DataLoader for country task
    """
    device = args.device
    model = model.to(device)
    model.eval()  # Freeze weights
    
    # Wrap model for sequential processing
    seq_model = SequentialRNN(model)
    seq_model = seq_model.to(device)
    
    print(f"\n开始Meta-Testing...")
    print(f"  模型权重已冻结")
    print(f"  测试国家数量: {country_data_gen.n_countries}\n")
    
    # For each country, we need to:
    # 1. Create a 1D support set (simulating human learning phase)
    # 2. Test on 2D query set (simulating human testing phase)
    
    country_results = {}
    
    for country_idx in range(country_data_gen.n_countries):
        country_info = country_data_gen.countries[country_idx]
        country_name = country_info[0]
        pref_type = country_info[1]
        
        print(f"测试国家: {country_name} ({pref_type})")
        
        # Generate 1D support set for this country
        # For 1D countries: use samples that match their 1D preference
        # For 2D countries: use samples that help infer the 2D preference
        support_set = []
        
        # Get country attributes and direction (needed for both single and dual)
        attrs = country_info[2]
        direction = country_info[3] if len(country_info) > 3 else (1 if pref_type == 'single' else (1, 1))
        
        if pref_type == 'single':
            # For 1D countries, create support samples using their 1D rule
            ctx_idx = attrs[0]
            
            # Sample from grid_data_gen that match this 1D rule
            # Important: We need to match both context AND direction
            for sample in grid_data_gen.train:
                ctx, loc1, loc2, y, info = sample
                if ctx == ctx_idx:  # Match the context
                    # Check if the sample's direction matches the country's direction
                    # In grid.train, samples are generated with direction=1 (higher is better)
                    # So y=0 means wine1 is better (wine1 has higher rank in ctx dimension)
                    rank1 = loc1[ctx_idx]
                    rank2 = loc2[ctx_idx]
                    
                    # Determine the sample's implicit direction
                    if rank1 > rank2:
                        # Wine1 has higher rank, so if y=0, direction is 1 (higher is better)
                        sample_direction = 1
                    elif rank1 < rank2:
                        # Wine2 has higher rank, so if y=1, direction is 1 (higher is better)
                        sample_direction = 1
                    else:
                        # Equal ranks, direction doesn't matter
                        sample_direction = 0
                    
                    # For direction=-1 countries, we need to flip the labels in support set
                    # so the model learns the correct preference
                    if direction == -1:
                        # Flip the label: if y=0 (wine1 better with direction=1), 
                        # then with direction=-1, wine2 is better (y=1)
                        flipped_sample = (ctx, loc1, loc2, 1 - y, info)
                        support_set.append(flipped_sample)
                    else:
                        support_set.append(sample)
                    if len(support_set) >= 16:
                        break
        else:
            # For 2D countries, create support samples that help infer 2D preference
            # Use samples from both contexts
            ctx0_samples = [s for s in grid_data_gen.train[:16] if s[0] == 0]
            ctx1_samples = [s for s in grid_data_gen.train[16:32] if s[0] == 1]
            support_set = ctx0_samples + ctx1_samples
        
        # Prepare support samples
        def prepare_samples(samples):
            ctx_list = []
            f1_list = []
            f2_list = []
            y_list = []
            
            for ctx, loc1, loc2, y, info in samples:
                ctx_list.append(ctx)
                f1_idx = grid_data_gen.loc2idx[loc1]
                f2_idx = grid_data_gen.loc2idx[loc2]
                f1_list.append(grid_data_gen.idx2tensor[f1_idx])
                f2_list.append(grid_data_gen.idx2tensor[f2_idx])
                y_list.append(y)
            
            ctx_tensor = torch.tensor(ctx_list).to(device)
            f1_tensor = torch.stack(f1_list).to(device) if not args.use_images else torch.stack(f1_list).to(device)
            f2_tensor = torch.stack(f2_list).to(device) if not args.use_images else torch.stack(f2_list).to(device)
            y_tensor = torch.tensor(y_list).to(device)
            
            return list(zip(ctx_tensor, f1_tensor, f2_tensor, y_tensor))
        
        support_samples = prepare_samples(support_set)
        
        # Adaptation: Process support set (hidden state adapts)
        with torch.no_grad():
            support_outputs, adapted_hidden = seq_model.forward_sequence(support_samples)
        
        # Evaluation: Test on country task samples
        # Get all samples for this country
        country_samples = country_data_gen.get_country_samples(country_idx)
        
        # Convert to query format
        query_samples = []
        correct_predictions = 0
        total_predictions = 0
        
        # Get country attributes and direction (needed for both single and dual)
        attrs = country_info[2]
        direction = country_info[3] if len(country_info) > 3 else (1 if pref_type == 'single' else (1, 1))
        
        for country_idx_q, wine1_idx, wine2_idx, correct_ans, info in country_samples[:32]:  # Test on first 32
            # Prepare input based on country preference
            f1_tensor = grid_data_gen.idx2tensor[wine1_idx].unsqueeze(0).to(device)
            f2_tensor = grid_data_gen.idx2tensor[wine2_idx].unsqueeze(0).to(device)
            
            # Determine context and prediction based on country preference
            if pref_type == 'single':
                # For 1D countries: use the context they care about
                ctx_idx = attrs[0]
                ctx_tensor = torch.tensor([ctx_idx]).to(device)
                
                query_sample = [(ctx_tensor, f1_tensor, f2_tensor, torch.tensor([correct_ans]).to(device))]
                
                with torch.no_grad():
                    query_outputs, _ = seq_model.forward_sequence(query_sample, adapted_hidden)
                    model_pred = torch.argmax(query_outputs[0], dim=1).item()
                    
                    # Apply direction: if direction=-1, flip the prediction
                    if direction == 1:
                        pred = model_pred  # Higher is better
                    else:  # direction == -1
                        pred = 1 - model_pred  # Lower is better, flip
            else:
                # For 2D countries: use model to predict for both contexts, then combine
                ctx_idx1, ctx_idx2 = attrs[0], attrs[1]
                dir1, dir2 = direction[0], direction[1]
                
                # Predict using context 1
                ctx1_tensor = torch.tensor([ctx_idx1]).to(device)
                query_sample_ctx1 = [(ctx1_tensor, f1_tensor, f2_tensor, torch.tensor([correct_ans]).to(device))]
                
                with torch.no_grad():
                    query_outputs_ctx1, _ = seq_model.forward_sequence(query_sample_ctx1, adapted_hidden)
                    model_pred_ctx1 = torch.argmax(query_outputs_ctx1[0], dim=1).item()
                    # Apply direction for ctx1
                    if dir1 == 1:
                        pred_ctx1 = model_pred_ctx1  # Higher is better
                    else:
                        pred_ctx1 = 1 - model_pred_ctx1  # Lower is better, flip
                    # Convert to score: 0 means wine1 better, 1 means wine2 better
                    score_ctx1 = 1 if pred_ctx1 == 1 else -1  # wine1 better = -1, wine2 better = +1
                
                # Predict using context 2
                ctx2_tensor = torch.tensor([ctx_idx2]).to(device)
                query_sample_ctx2 = [(ctx2_tensor, f1_tensor, f2_tensor, torch.tensor([correct_ans]).to(device))]
                
                with torch.no_grad():
                    query_outputs_ctx2, _ = seq_model.forward_sequence(query_sample_ctx2, adapted_hidden)
                    model_pred_ctx2 = torch.argmax(query_outputs_ctx2[0], dim=1).item()
                    # Apply direction for ctx2
                    if dir2 == 1:
                        pred_ctx2 = model_pred_ctx2  # Higher is better
                    else:
                        pred_ctx2 = 1 - model_pred_ctx2  # Lower is better, flip
                    # Convert to score
                    score_ctx2 = 1 if pred_ctx2 == 1 else -1
                
                # Combine scores from both contexts
                # If both contexts agree, use that; if they disagree, use the stronger signal
                combined_score = score_ctx1 + score_ctx2
                
                if combined_score < 0:
                    pred = 0  # wine1 better (both contexts prefer wine1, or one strongly prefers wine1)
                elif combined_score > 0:
                    pred = 1  # wine2 better (both contexts prefer wine2, or one strongly prefers wine2)
                else:
                    # Equal case: randomly choose
                    pred = random.choice([0, 1])
            
            if pred == correct_ans:
                correct_predictions += 1
            total_predictions += 1
        
        acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        country_results[country_name] = acc
        print(f"  {country_name} 准确率: {acc:.4f} ({acc*100:.2f}%)\n")
    
    return country_results


def create_meta_learning_args(base_args):
    """
    Create arguments object with meta-learning specific parameters.
    Note: This function preserves values from base_args if they exist.
    """
    class MetaArgs:
        def __init__(self, base_args):
            # Copy all attributes from base_args first
            for attr in dir(base_args):
                if not attr.startswith('_'):
                    setattr(self, attr, getattr(base_args, attr))
            
            # Meta-learning specific parameters (only set defaults if not already in base_args)
            # Since we copied all attributes above, we only need to set defaults for missing ones
            if not hasattr(self, 'meta_lr'):
                self.meta_lr = 0.001  # Meta-learning rate
            if not hasattr(self, 'n_support'):
                self.n_support = 16   # Number of support samples per task
            if not hasattr(self, 'n_query'):
                self.n_query = 32     # Number of query samples per task
            # n_meta_iterations: 使用base_args中的值（如果已设置），否则使用默认值10000
            if not hasattr(self, 'n_meta_iterations'):
                self.n_meta_iterations = 10000  # Number of meta-training iterations (default)
            # 注意：如果base_args中已经设置了n_meta_iterations，上面的复制已经包含了它，这里不会覆盖
            if not hasattr(self, 'n_tasks_per_batch'):
                self.n_tasks_per_batch = 4      # Number of tasks per meta-batch
    
    return MetaArgs(base_args)

