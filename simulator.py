NBYTES_PER_ELEMENT = 4

# # 32 GPUs
ALPHA_GEMM = 0.0007418184974416292
BETA_GEMM = 2.6878112330809866e-14
# ALPHA_COMM = 0.005187680139563881
# BETA_COMM = 9.813817794986829e-10

# 16 GPUs
# ALPHA_GEMM = -0.006414464795235193
# BETA_GEMM = 2.892511360391272e-14
# ALPHA_COMM = 0.0020978959973651816 # 0.014294450539516544
# BETA_COMM = 4.349552216335826e-10 # 6.860893488916374e-10

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
    gemm: tuple[float, float],
    comm: tuple[float, float],
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
    return timestamp_c_lst[-1] + t_d


def predict_moe_our_time(
    B: int,
    L: int,
    M: int,
    H: int,
    r: int,
    experts_per_token: int,
    capacity_factor: float,
    gemm: tuple[float, float],
    comm: tuple[float, float],
) -> float:
    num_tokens_pre_gpu = experts_per_token * int(capacity_factor * B * L / r + 0.5)
    len_of_alltoall = num_tokens_pre_gpu * M
    len_of_expert_gemm = num_tokens_pre_gpu * M * H

    n_e = len_of_expert_gemm * NBYTES_PER_ELEMENT
    n_d = len_of_alltoall * NBYTES_PER_ELEMENT

    t_gemm = gemm[0] + gemm[1] * n_e
    t_d = comm[0] + comm[1] * n_d
    print(t_d)
    # print(f't_gemm={t_gemm}, t_d={t_d}')

    timestamp_d_lst = [0]
    timestamp_e_lst = [t_d]
    for i in range(1, r):
        min_computaion_finish = timestamp_e_lst[-1] + 2 * t_gemm
        for e in timestamp_e_lst:
            for k in [0, 1, 2]:
                if e + k * t_gemm >= timestamp_d_lst[-1] + t_d:
                    min_computaion_finish = min(min_computaion_finish, e + k * t_gemm)
        timestamp_d_lst.append(min_computaion_finish)

        # print(timestamp_e_lst[-1] + 2 * t_gemm, timestamp_d_lst[-1] + t_d)
        timestamp_e_lst.append(
            max(timestamp_e_lst[-1] + 2 * t_gemm, timestamp_d_lst[-1] + t_d)
        )

    min_computaion_finish = timestamp_e_lst[-1] + 2 * t_gemm
    for e in timestamp_e_lst[1:]:
        for k in [0, 1, 2]:
            if e + k * t_gemm >= timestamp_d_lst[-1] + t_d:
                min_computaion_finish = min(min_computaion_finish, e + k * t_gemm)
    timestamp_c_lst = [
        min_computaion_finish,
    ]

    for i in range(1, r):
        min_computaion_finish = max(
            timestamp_e_lst[-1] + 2 * t_gemm, timestamp_c_lst[-1] + t_d
        )
        for e in timestamp_e_lst[i + 1 :]:
            for k in [0, 1, 2]:
                if e + k * t_gemm >= timestamp_c_lst[-1] + t_d:
                    min_computaion_finish = min(min_computaion_finish, e + k * t_gemm)
        timestamp_c_lst.append(min_computaion_finish)

    return timestamp_c_lst[-1] + t_d


def main():
    for B in [8, 16, 32, 64]:
        for L in [512, 1024, 2048, 4096]: 
            for M in [1024, 2048, 4096, 8192]:
                for H in [4096]:
                    for r in [3, 4]:
                        # if B!=64 or L!=512 or M!=2048 or H!=4096 or r!=3:
                            # continue
                        base = predict_moe_baseline_time(
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
                        our = predict_moe_our_time(
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
                        # if our != base:
                            # print(B, L, M, H, r, our, base)
                        # print(B, L, M, H, r, our)
                        # return 0


if __name__ == "__main__":
    main()
