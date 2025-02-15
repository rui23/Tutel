import subprocess
import time
import os

env = os.environ.copy()

env['SKIP_EXPERT'] = '0'

# 定义参数范围
batch_size_range = [32] # [8, 16, 32]
num_tokens_range = [1024] # [512, 1024]
a2a_ffn_overlap_degree_range = [3] # [2, 3, 4]
model_dim_range = [1024, 2048, 4096, 8192]
hidden_size_range = [8192] # [1024, 2048, 4096, 8192, 16384, 32768]
# batch_size_range = [2]

# 基础命令
base_cmd = ['python3', '-m', 'torch.distributed.run', '--nproc_per_node=8', '--nnodes=4', '--node_rank=3', '--master_addr=10.0.0.11', '-m', 
            'tutel.examples.helloworld', '--top=2', '--num_local_experts=1']

# 构建所有可能的命令
commands = []
for batch_size in batch_size_range:
    for num_tokens in num_tokens_range:
        for model_dim in model_dim_range:
            for hidden_size in hidden_size_range:
                for  a2a_ffn_overlap_degree in a2a_ffn_overlap_degree_range:
                    cmd = base_cmd + [
                        '--batch_size', str(batch_size),
                        '--num_tokens', str(num_tokens),
                        '--a2a_ffn_overlap_degree',str(a2a_ffn_overlap_degree),
                        '--model_dim', str(model_dim),
                        '--hidden_size', str(hidden_size)
                    ]
                    commands.append(cmd)

# 执行命令并记录时间
for i, cmd in enumerate(commands):
    print(f"Running command {i + 1} out of {len(commands)}: \n{' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("Command finished")
    print(f"Output: {result.stdout}")
    print(f"Errors (if any): {result.stderr}")
    print("-" * 80)