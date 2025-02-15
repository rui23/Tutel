NBYTES_PER_ELEMENT = 4

ALPHA_GEMM = 0
BETA_GEMM = 0.0001
ALPHA_COMM = 0
BETA_COMM = 1.0


def predict_gemm_time(m: int, n: int, k: int, alpha: float, beta: float) -> float:
    x = m * n * k
    t = alpha + beta * x * NBYTES_PER_ELEMENT
    return t


def predict_a2a_time(x: int, alpha: float, beta: float) -> float:
    t = alpha + beta * x * NBYTES_PER_ELEMENT
    return t


import math


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
    num_tokens_pre_gpu = experts_per_token * math.ceil(capacity_factor * B * L / r)
    len_of_alltoall = num_tokens_pre_gpu * M
    len_of_expert_gemm = num_tokens_pre_gpu * M * H

    n_e = len_of_expert_gemm * NBYTES_PER_ELEMENT
    n_d = len_of_alltoall * NBYTES_PER_ELEMENT

    alpha_e = 2 * gemm[0]
    beta_e = 2 * gemm[1]

    t_gemm = gemm[0] + gemm[1] * n_e

    t_e = alpha_e + beta_e * n_e
    t_d = comm[0] + comm[1] * n_d

    if t_d >= t_gemm:
        return []
    if 2 * math.ceil(t_d / t_gemm) * (r - 1) <= 2 * r:
        return []
    ps = 1
    while (
        math.ceil(ps * t_d / t_e) + (2 * r - 2 - ps) * math.ceil(t_d / t_gemm) > 2 * r
    ):
        ps = ps + 1

    lst = []

    for i in range(ps):
        lst.append(2 * r - (2 * r - 2 - ps) * math.ceil(t_d / t_gemm) * i / ps)
    return lst


def main():
    for B in [2, 4, 8]:
        for L in [512, 1024, 2048]:
            for M in [1024, 2048, 4096]:
                for H in [1024, 2048, 4096]:
                    for r in [2, 3, 4]:
                        split_lst = predict_moe_baseline_time(
                            B,
                            L,
                            M,
                            H,
                            r,
                            2,
                            1.0,
                            (
                                ALPHA_GEMM,
                                BETA_GEMM,
                            ),
                            (
                                ALPHA_COMM,
                                BETA_COMM,
                            ),
                        )
                        print(split_lst)


if __name__ == "__main__":
    # main()
    split_lst = predict_moe_baseline_time(
                            2,
                            2048,
                            4096,
                            16384,
                            3,
                            2,
                            1.0,
                            (
                                ALPHA_GEMM,
                                BETA_GEMM,
                            ),
                            (
                                ALPHA_COMM,
                                BETA_COMM,
                            ),
                        )
    print(split_lst)