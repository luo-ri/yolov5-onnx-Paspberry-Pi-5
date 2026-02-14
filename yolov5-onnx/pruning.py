from utils.general import LOGGER
import torch
from utils.torch_utils import prune, select_device

# 设置设备
device = select_device('0')

# 加载模型
weights_path = 'weights/road_best.pt'
ckpt = torch.load(weights_path, map_location=device)

# 提取模型
model = ckpt['model']
model.to(device)

# 剪枝前参数量
params_before = sum(p.numel() for p in model.parameters())
nonzero_before = sum((p != 0).sum().item() for p in model.parameters())

print('='*50)
print('剪枝前统计:')
print(f'总参数量: {params_before:,}')
print(f'非零参数: {nonzero_before:,}')
print(f'零值参数: {params_before - nonzero_before:,}')
print(f'稀疏度: {(params_before - nonzero_before) / params_before * 100:.2f}%')
print('='*50)

# 执行剪枝
print('\n正在剪枝...')
prune(model, 0.3)
pruned_model =model
# 剪枝后参数量
params_after = sum(p.numel() for p in model.parameters())
nonzero_after = sum((p != 0).sum().item() for p in model.parameters())

print('='*50)
print('剪枝后统计:')
print(f'总参数量: {params_after:,}')
print(f'非零参数: {nonzero_after:,}')
print(f'零值参数: {params_after - nonzero_after:,}')
print(f'稀疏度: {(params_after - nonzero_after) / params_after * 100:.2f}%')
print('='*50)

# 对比
print('\n对比结果:')
print(f'参数减少: {params_before - params_after:,} ({(params_before - params_after) / params_before * 100:.2f}%)')
print(f'实际有效参数: {nonzero_after:,} -> {nonzero_after / params_before * 100:.2f}% 的原始模型')

# 保存剪枝后的模型
output_path = r'weights\roda_best_pruned.pt'  # 输出文件路径
torch.save({'model': pruned_model}, output_path)  # 保存为字典格式
LOGGER.info(f'\n剪枝后的模型已保存至: {output_path}')


