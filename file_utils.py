import csv
import os
import pandas as pd

def create_csv(filename, headers):
    """
    创建一个新的CSV文件，并写入表头。
    
    参数:
        filename (str): CSV文件的路径和名称。
        headers (list): 表头列表。
    """
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=headers)
        df.to_csv(filename, index=False)


def add_record(filename, record):
    """
    在CSV文件中添加一行新的记录。
    
    参数:
        filename (str): CSV文件的路径和名称。
        record (dict): 包含列名和对应值的字典。
    """
    with open(filename,'a') as f:
        writer = csv.writer(f)
        writer.writerow(record)



def update_last_record(filename, column_name, new_value):
    """
    更新CSV文件中最后一行的指定列的值。
    
    参数:
        filename (str): CSV文件的路径和名称。
        column_name (str): 要更新的列名。
        new_value: 新的值。
    """
    # 读取CSV文件
    df = pd.read_csv(filename)
    
    # 获取最后一行索引
    last_index = df.index[-1]
    
    # 更新最后一行的指定列
    df.at[last_index, column_name] = new_value
    
    # 写回CSV文件
    df.to_csv(filename, index=False)


def update_time(filename, time):
    df = pd.read_csv(filename)
    last_index = df.index[-1]
    cur_iter = df.at[last_index, 'iter']
    if cur_iter >= 20:
        df.at[last_index,'time'] += time
        df.to_csv(filename, index=False)

def update_pipe_gemm_time(filename, idx, time):
    df = pd.read_csv(filename)
    last_index = df.index[-1]
    cur_iter = df.at[last_index, 'iter']
    if cur_iter >= 20:
        df.at[last_index,'time'+str(idx)] += time
        df.to_csv(filename, index=False)