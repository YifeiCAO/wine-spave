"""
调试脚本：检查meta_learning_v2的实现是否正确
"""
import torch
import torch.nn as nn

# 模拟检查
class TestRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_dim = 32
        self.hidden_dim = 128
        self.lstm = nn.LSTM(32, 128)
        self.out = nn.Linear(128, 2)

# 创建模型
model = TestRNN()
print(f"初始参数数: {sum(p.numel() for p in model.parameters())}")

# 动态添加rule_embedding（模拟meta_train_v2）
model.rule_embedding = nn.Linear(4, model.state_dim)
print(f"添加rule_embedding后参数数: {sum(p.numel() for p in model.parameters())}")

# 检查是否包含rule_embedding的参数
has_rule_embedding = any('rule_embedding' in name for name, _ in model.named_parameters())
print(f"包含rule_embedding参数: {has_rule_embedding}")

# 检查参数数量
rule_embedding_params = sum(p.numel() for p in model.rule_embedding.parameters())
print(f"rule_embedding参数数: {rule_embedding_params} (应该是160: 4*32+32)")

# 测试optimizer
optimizer = torch.optim.Adam(model.parameters())
print(f"Optimizer参数组数: {len(optimizer.param_groups)}")
print(f"Optimizer参数总数: {sum(len(group['params']) for group in optimizer.param_groups)}")

print("\n✅ 如果所有检查都通过，说明实现是正确的")

