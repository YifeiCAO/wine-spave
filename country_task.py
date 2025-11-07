import torch
import torch.nn as nn
import random
from itertools import permutations
from torch.utils.data import Dataset, DataLoader

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from utils import get_congruency


class CountryTaskDataGenerator:
    """
    Generate data for the countries task.
    
    Countries have different preferences:
    - Some countries prefer only one attribute (single attribute)
    - Some countries prefer two attributes equally (dual attribute, need to combine)
    """
    def __init__(self, size=4, use_images=False, image_dir=None, idx2tensor=None, loc2idx=None):
        self.size = size
        self.use_images = use_images
        self.image_dir = image_dir
        self.idx2tensor = idx2tensor  # From existing data generator
        self.loc2idx = loc2idx
        
        # Generate locations
        self.locs = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.idxs = [idx for idx in range(len(self.locs))]
        if self.loc2idx is None:
            self.loc2idx = {loc: idx for loc, idx in zip(self.locs, self.idxs)}
        if self.idx2tensor is None:
            # Create default idx2tensor if not provided
            self.idx2tensor = {idx: torch.tensor(idx).type(torch.long) for idx in self.idxs}
        
        # Define 8 countries and their preferences
        # Format: (country_name, preference_type, attributes, direction)
        # preference_type: 'single' or 'dual'
        # attributes: for 'single' it's (ctx_idx,), for 'dual' it's (ctx_idx1, ctx_idx2)
        # direction: for 'single' it's the direction in that dimension, for 'dual' it's (dir1, dir2)
        # direction: 1 means higher is better, -1 means lower is better
        
        # 4个1D国家：只关心一个维度，且关心的方向互不相同
        # - France: 关心上下文0，方向=1 (高taste更好)
        # - Italy: 关心上下文0，方向=-1 (低taste更好，即高dry更好)
        # - China: 关心上下文1，方向=1 (高body更好，即full更好)
        # - Australia: 关心上下文1，方向=-1 (低body更好，即light更好)
        
        # 4个2D国家：关心两个维度，且方向组合互不相同
        # - Russia: 关心上下文0和1，方向=(1, 1) (都高更好)
        # - Brazil: 关心上下文0和1，方向=(1, -1) (上下文0高，上下文1低)
        # - India: 关心上下文0和1，方向=(-1, 1) (上下文0低，上下文1高)
        # - Mexico: 关心上下文0和1，方向=(-1, -1) (都低更好)
        
        self.countries = [
            # 1D countries - 4个，方向互不相同
            ('France', 'single', (0,), 1),      # Context 0, higher is better
            ('Italy', 'single', (0,), -1),      # Context 0, lower is better
            ('China', 'single', (1,), 1),       # Context 1, higher is better
            ('Australia', 'single', (1,), -1),  # Context 1, lower is better
            
            # 2D countries - 4个，方向组合互不相同
            ('Russia', 'dual', (0, 1), (1, 1)),    # Both high
            ('Brazil', 'dual', (0, 1), (1, -1)),  # Ctx0 high, Ctx1 low
            ('India', 'dual', (0, 1), (-1, 1)),   # Ctx0 low, Ctx1 high
            ('Mexico', 'dual', (0, 1), (-1, -1)), # Both low
        ]
        self.n_countries = len(self.countries)
        
        # Generate all possible wine pairs
        self.all_pairs = list(permutations(self.idxs, 2))
        
        # Generate samples for each country
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        """
        Generate samples for all countries.
        Each sample: (country_idx, wine1_idx, wine2_idx, correct_answer, info)
        correct_answer: 0=wine1 better, 1=wine2 better, 2=equal
        """
        samples = []
        
        for country_idx, country_info in enumerate(self.countries):
            country_name = country_info[0]
            pref_type = country_info[1]
            attrs = country_info[2]
            direction = country_info[3] if len(country_info) > 3 else (1 if pref_type == 'single' else (1, 1))
            
            for wine1_idx, wine2_idx in self.all_pairs:
                loc1 = self.locs[wine1_idx]
                loc2 = self.locs[wine2_idx]
                
                # Determine correct answer based on country preference and direction
                if pref_type == 'single':
                    # Single attribute: compare based on one dimension with direction
                    ctx_idx = attrs[0]
                    rank1 = loc1[ctx_idx]
                    rank2 = loc2[ctx_idx]
                    
                    # Apply direction: 1 means higher is better, -1 means lower is better
                    if direction == 1:
                        # Higher is better
                        if rank1 > rank2:
                            correct = 0  # wine1 is better
                        elif rank1 < rank2:
                            correct = 1  # wine2 is better
                        else:
                            correct = 2  # equal
                    else:  # direction == -1
                        # Lower is better
                        if rank1 < rank2:
                            correct = 0  # wine1 is better
                        elif rank1 > rank2:
                            correct = 1  # wine2 is better
                        else:
                            correct = 2  # equal
                
                elif pref_type == 'dual':
                    # Dual attribute: combine both dimensions with their respective directions
                    ctx_idx1, ctx_idx2 = attrs[0], attrs[1]
                    dir1, dir2 = direction[0], direction[1]
                    
                    rank1_ctx1 = loc1[ctx_idx1]
                    rank2_ctx1 = loc2[ctx_idx1]
                    rank1_ctx2 = loc1[ctx_idx2]
                    rank2_ctx2 = loc2[ctx_idx2]
                    
                    # Calculate value considering direction for each dimension
                    # For direction=1: higher is better, so use rank directly
                    # For direction=-1: lower is better, so use (max_rank - rank)
                    max_rank = self.size - 1
                    value1 = (rank1_ctx1 if dir1 == 1 else (max_rank - rank1_ctx1)) + \
                            (rank1_ctx2 if dir2 == 1 else (max_rank - rank1_ctx2))
                    value2 = (rank2_ctx1 if dir1 == 1 else (max_rank - rank2_ctx1)) + \
                            (rank2_ctx2 if dir2 == 1 else (max_rank - rank2_ctx2))
                    
                    if value1 > value2:
                        correct = 0  # wine1 is better
                    elif value1 < value2:
                        correct = 1  # wine2 is better
                    else:
                        correct = 2  # equal
                
                # Additional info
                info = {
                    'country_name': country_name,
                    'pref_type': pref_type,
                    'attrs': attrs,
                    'direction': direction,
                    'loc1': loc1,
                    'loc2': loc2,
                    'rank_diff_ctx0': loc1[0] - loc2[0],
                    'rank_diff_ctx1': loc1[1] - loc2[1],
                }
                
                samples.append((country_idx, wine1_idx, wine2_idx, correct, info))
        
        return samples
    
    def get_country_samples(self, country_idx):
        """Get all samples for a specific country"""
        return [s for s in self.samples if s[0] == country_idx]
    
    def create_blocks(self, n_runs=2, trials_per_block=16):
        """
        Create blocks for countries task.
        Structure: n_runs, each run has n_countries blocks, each block has trials_per_block trials
        """
        blocks = []
        
        for run_idx in range(n_runs):
            # Shuffle country order for each run
            country_order = list(range(self.n_countries))
            random.shuffle(country_order)
            
            for country_idx in country_order:
                # Get all samples for this country
                country_samples = self.get_country_samples(country_idx)
                
                # Randomly sample trials_per_block trials
                if len(country_samples) >= trials_per_block:
                    block_samples = random.sample(country_samples, trials_per_block)
                else:
                    # If not enough samples, repeat with shuffling
                    block_samples = random.choices(country_samples, k=trials_per_block)
                
                blocks.append({
                    'run_idx': run_idx,
                    'country_idx': country_idx,
                    'country_name': self.countries[country_idx][0],
                    'samples': block_samples
                })
        
        return blocks


class CountryTaskDataset(Dataset):
    """
    Dataset for countries task.
    Each sample: (country_idx, wine1, wine2, correct_answer, info)
    """
    def __init__(self, blocks, idx2tensor, loc2idx):
        self.blocks = blocks
        self.idx2tensor = idx2tensor
        self.loc2idx = loc2idx
        
        # Flatten blocks into samples
        self.samples = []
        for block in blocks:
            for sample in block['samples']:
                country_idx, wine1_idx, wine2_idx, correct, info = sample
                # Add block info to sample
                sample_with_block = (country_idx, wine1_idx, wine2_idx, correct, 
                                   {**info, 'run_idx': block['run_idx'], 
                                    'block_idx': block['country_idx']})
                self.samples.append(sample_with_block)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        country_idx, wine1_idx, wine2_idx, correct, info = self.samples[i]
        
        country_idx_tensor = torch.tensor(country_idx).type(torch.long)
        wine1_tensor = self.idx2tensor[wine1_idx]
        wine2_tensor = self.idx2tensor[wine2_idx]
        correct_tensor = torch.tensor(correct).type(torch.long)
        
        return country_idx_tensor, wine1_tensor, wine2_tensor, correct_tensor, info


def country_task_collate(samples):
    """Collate function for countries task"""
    country_batch = torch.cat([s[0].unsqueeze(0) for s in samples], dim=0)
    wine1_batch = torch.cat([s[1].unsqueeze(0) if len(s[1].shape) == 0 else s[1] for s in samples], dim=0)
    wine2_batch = torch.cat([s[2].unsqueeze(0) if len(s[2].shape) == 0 else s[2] for s in samples], dim=0)
    correct_batch = torch.cat([s[3].unsqueeze(0) for s in samples], dim=0)
    
    info_batch = {k: [] for k in samples[0][4].keys()}
    for s in samples:
        info = s[4]
        for k, v in info.items():
            info_batch[k].append(v)
    
    return country_batch, wine1_batch, wine2_batch, correct_batch, info_batch


def test_country_task(model, loader, args, country_data_gen):
    """
    Test model on countries task.
    
    The model needs to output 3 classes:
    - 0: wine1 is better for the country
    - 1: wine2 is better for the country
    - 2: wines are equal for the country
    """
    model.eval()
    
    # For countries task, we need 3-class output
    # If model only has 2-class output, we'll need to handle it differently
    # For now, assume model can be adapted or we use a wrapper
    
    with torch.no_grad():
        correct = []
        country_correct = {i: [] for i in range(country_data_gen.n_countries)}
        pref_type_correct = {'single': [], 'dual': []}
        rank_diff_0_correct = []  # For trials where rank difference in ctx0 is 0
        rank_diff_1_correct = []  # For trials where rank difference in ctx1 is 0
        rank_diff_2d_correct = []  # For 2D trials (both rank differences are 0)
        
        for batch in loader:
            country_idx, wine1, wine2, correct_ans, info = batch
            country_idx = country_idx.to(args.device)
            wine1 = wine1.to(args.device)
            wine2 = wine2.to(args.device)
            correct_ans = correct_ans.to(args.device)
            
            # For countries task, we need to determine which context to use
            # based on the country's preference
            # Since the model expects (ctx, f1, f2), we need to adapt
            
            # Get country preferences
            batch_size = country_idx.shape[0]
            predictions = []
            
            for i in range(batch_size):
                country_i = country_idx[i].item()
                country_info = country_data_gen.countries[country_i]
                country_name = country_info[0]
                pref_type = country_info[1]
                attrs = country_info[2]
                direction = country_info[3] if len(country_info) > 3 else (1 if pref_type == 'single' else (1, 1))
                
                wine1_i = wine1[i:i+1]
                wine2_i = wine2[i:i+1]
                
                if pref_type == 'single':
                    # Single attribute: use that context with direction
                    ctx_idx = attrs[0]
                    ctx_tensor = torch.tensor([ctx_idx]).to(args.device)
                    
                    # Get actual ranks to check for equality first
                    loc1 = info['loc1'][i]
                    loc2 = info['loc2'][i]
                    rank1 = loc1[ctx_idx]
                    rank2 = loc2[ctx_idx]
                    
                    if rank1 == rank2:
                        pred = 2  # equal
                    else:
                        # Run model with this context
                        y_hat, _ = model(ctx_tensor, wine1_i, wine2_i)
                        # y_hat: [1, 2] - 0 means wine1 better, 1 means wine2 better
                        model_pred = torch.argmax(y_hat, dim=1).item()
                        
                        # Apply direction: if direction=-1, we need to flip the prediction
                        # because model was trained with direction=1 (higher is better)
                        if direction == 1:
                            pred = model_pred  # Higher is better, use model prediction directly
                        else:  # direction == -1
                            # Lower is better, flip the prediction
                            pred = 1 - model_pred if model_pred != 2 else 2
                
                elif pref_type == 'dual':
                    # Dual attribute: combine both dimensions with their directions
                    ctx_idx1, ctx_idx2 = attrs[0], attrs[1]
                    dir1, dir2 = direction[0], direction[1]
                    
                    # Get actual ranks
                    loc1 = info['loc1'][i]
                    loc2 = info['loc2'][i]
                    rank1_ctx1, rank2_ctx1 = loc1[ctx_idx1], loc2[ctx_idx1]
                    rank1_ctx2, rank2_ctx2 = loc1[ctx_idx2], loc2[ctx_idx2]
                    
                    # Calculate value considering direction for each dimension
                    max_rank = country_data_gen.size - 1
                    value1 = (rank1_ctx1 if dir1 == 1 else (max_rank - rank1_ctx1)) + \
                            (rank1_ctx2 if dir2 == 1 else (max_rank - rank1_ctx2))
                    value2 = (rank2_ctx1 if dir1 == 1 else (max_rank - rank2_ctx1)) + \
                            (rank2_ctx2 if dir2 == 1 else (max_rank - rank2_ctx2))
                    
                    if value1 > value2:
                        pred = 0  # wine1 better
                    elif value1 < value2:
                        pred = 1  # wine2 better
                    else:
                        pred = 2  # equal
                
                predictions.append(pred)
            
            predictions = torch.tensor(predictions).to(args.device)
            
            # Compute correctness
            c = (predictions == correct_ans).cpu().tolist()  # Use tolist() for compatibility
            c = [bool(c_i) for c_i in c]
            correct += c
            
            # Separate by country
            for i, (c_i, country_i) in enumerate(zip(c, country_idx.cpu().tolist())):
                country_correct[country_i].append(c_i)
                
                # Separate by preference type
                _, pref_type, _ = country_data_gen.countries[country_i]
                pref_type_correct[pref_type].append(c_i)
                
                # Check for 2D trials (rank difference 0 in both dimensions)
                rank_diff_ctx0 = info['rank_diff_ctx0'][i]
                rank_diff_ctx1 = info['rank_diff_ctx1'][i]
                
                if rank_diff_ctx0 == 0:
                    rank_diff_0_correct.append(c_i)
                if rank_diff_ctx1 == 0:
                    rank_diff_1_correct.append(c_i)
                if rank_diff_ctx0 == 0 and rank_diff_ctx1 == 0:
                    rank_diff_2d_correct.append(c_i)
        
        # Compute accuracies (using Python built-in functions for compatibility)
        def safe_mean(lst):
            return sum(lst) / len(lst) if len(lst) > 0 else 0.0
        
        results = {
            'overall_acc': safe_mean(correct),
            'country_acc': {i: safe_mean(country_correct[i]) 
                           for i in range(country_data_gen.n_countries)},
            'single_pref_acc': safe_mean(pref_type_correct['single']),
            'dual_pref_acc': safe_mean(pref_type_correct['dual']),
            'rank_diff_0_acc': safe_mean(rank_diff_0_correct),
            'rank_diff_1_acc': safe_mean(rank_diff_1_correct),
            'rank_diff_2d_acc': safe_mean(rank_diff_2d_correct),
        }
        
        # Add country names to results
        results['country_names'] = [country_data_gen.countries[i][0] for i in range(country_data_gen.n_countries)]
        
        model.train()
        return results


def get_country_task_loaders(args, grid_data_gen):
    """
    Create data loaders for countries task.
    
    Args:
        args: arguments object
        grid_data_gen: GridDataGenerator instance (for idx2tensor and loc2idx)
    """
    # Create country task data generator
    country_data_gen = CountryTaskDataGenerator(
        size=args.grid_size,
        use_images=args.use_images,
        image_dir=args.image_dir,
        idx2tensor=grid_data_gen.idx2tensor,
        loc2idx=grid_data_gen.loc2idx
    )
    
    # Create blocks: 2 runs, 8 countries per run, 16 trials per block
    n_runs = getattr(args, 'country_n_runs', 2)
    trials_per_block = getattr(args, 'country_trials_per_block', 16)
    blocks = country_data_gen.create_blocks(n_runs=n_runs, trials_per_block=trials_per_block)
    
    # Create dataset
    dataset = CountryTaskDataset(blocks, grid_data_gen.idx2tensor, grid_data_gen.loc2idx)
    
    # Create data loader
    batch_size = getattr(args, 'country_bs', args.bs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       collate_fn=country_task_collate)
    
    return loader, country_data_gen

