import file_utils


# CSV文件名
filename = 'new_example.csv'

# 表头
header = ['world_size', 'model_dim', 'hidden_size', 'num_local_experts', 'batch_size','iter', 'time']

# 创建CSV文件并写入表头
file_utils.create_csv(filename, header)


# # 新的一行数据，除了time以外
# new_data = [32, 4096, 1024, 1, 512, 0, 0.0]

# # 添加新行到CSV文件
# file_utils.add_record(filename, new_data)

# # 更新CSV文件
file_utils.update_last_record(filename, 'iter', 20)
file_utils.update_time(filename, 2.1)