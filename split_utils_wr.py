NBYTES_PER_ELEMENT = 4

# # 32 GPUs
ALPHA_GEMM = 0.0007418184974416292
BETA_GEMM = 2.6878112330809866e-14
# ALPHA_COMM = 0.005187680139563881
# BETA_COMM = 9.813817794986829e-10

# 16 GPUs
# ALPHA_GEMM = -0.006414464795235193
# BETA_GEMM = 2.892511360391272e-14
# ALPHA_COMM = 0.014294450539516544
# BETA_COMM = 6.860893488916374e-10

# 8 GPUs
ALPHA_COMM = 0.002591612937800384
BETA_COMM = 1.1260154433799957e-10

def predict_gemm_time(m: int, n: int, k: int, alpha: float, beta: float) -> float:
    x = m * n * k
    t = alpha + beta * x * NBYTES_PER_ELEMENT
    return t


def predict_a2a_time(x: int, alpha: float, beta: float) -> float:
    t = alpha + beta * x * NBYTES_PER_ELEMENT
    return t


def predict_moe_baseline_time(
    B: int,
    L: int,
    M: int,
    H: int,
    r: int,
    experts_per_token: int,
    capacity_factor: float,
    gemm: tuple[float, float] = (ALPHA_GEMM, BETA_GEMM,),
    comm: tuple[float, float] = (ALPHA_COMM, BETA_COMM),
) -> float:
    num_tokens_pre_gpu = experts_per_token * int(capacity_factor * B * L / r + 0.5)
    len_of_alltoall = num_tokens_pre_gpu * M
    len_of_expert_gemm = num_tokens_pre_gpu * M * H

    n_e = len_of_expert_gemm * NBYTES_PER_ELEMENT
    n_d = len_of_alltoall * NBYTES_PER_ELEMENT

    alpha_e = 2 * gemm[0]
    beta_e = 2 * gemm[1]

    t_e = alpha_e + beta_e * n_e
    t_d = comm[0] + comm[1] * n_d
    # print(f'\nbase t_e = {t_e}, t_d = {t_d}') 

    timestamp_d_lst = [0]
    timestamp_e_lst = [t_d] # 第r个expert的开始时间
    for i in range(1, r):
        timestamp_d_lst.append(timestamp_d_lst[-1]+t_d)
        timestamp_e_lst.append(max(timestamp_e_lst[-1] + t_e, t_d * (i + 1)))

    timestamp_c_lst = [max(t_d * r, timestamp_e_lst[0] + t_e)] # 第r个AlltoAll combine的开始时间
    for i in range(1, r):
        timestamp_c_lst.append(max(timestamp_c_lst[-1] + t_d, timestamp_e_lst[i] + t_e))
    
    timestamp_gemm_start = []
    timestamp_gemm_end = []
    timestamp_a2a_lst = timestamp_d_lst + timestamp_c_lst
    for i in range(len(timestamp_e_lst)):
        timestamp_gemm_start.append(timestamp_e_lst[i])
        timestamp_gemm_start.append(timestamp_e_lst[i]+float(t_e/2))
        timestamp_gemm_end.append(timestamp_e_lst[i]+float(t_e/2))
        timestamp_gemm_end.append(timestamp_e_lst[i]+t_e)
    timestamp_splits = list(0.0 for _ in range(len(timestamp_gemm_start)))
    cur_gemm_idx = 0
    for comb in timestamp_a2a_lst:
        while cur_gemm_idx<len(timestamp_gemm_start) and timestamp_gemm_end[cur_gemm_idx]<comb:
            cur_gemm_idx += 1  
        if cur_gemm_idx == len(timestamp_gemm_start):
                break
        if timestamp_gemm_start[cur_gemm_idx]<comb and timestamp_gemm_end[cur_gemm_idx]>comb:
            timestamp_splits[cur_gemm_idx] = (comb-timestamp_gemm_start[cur_gemm_idx])/(timestamp_gemm_end[cur_gemm_idx]-timestamp_gemm_start[cur_gemm_idx])
            cur_gemm_idx += 1
            if cur_gemm_idx == len(timestamp_gemm_start):
                break
    # print(f'\ngemm = {t_e/2}\ntimestamp_a2a_lst = {timestamp_a2a_lst} \ntimestamp_gemm_start = {timestamp_gemm_start}, \ntimestamp_gemm_end = {timestamp_gemm_end}\nsplits = {timestamp_splits}')
                
    return timestamp_splits

def main():
    for B in [8, 16, 32, 64]:
        for L in [512, 1024, 2048, 4096]: 
            for M in [1024, 2048, 4096, 8192]:
                for H in [4096]:
                    for r in [3, 4]:
                        split_list = predict_moe_baseline_time(
                            B,
                            L,
                            M,
                            H,
                            r,
                            2,
                            1,
                            (
                                ALPHA_GEMM,
                                BETA_GEMM,
                            ),
                            (
                                ALPHA_COMM,
                                BETA_COMM,
                            ),
                        )
                        # print(B, L, M, H, r, split_list)
                        print(split_list)

if __name__ == "__main__":
    main()