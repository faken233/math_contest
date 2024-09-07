import numpy as np
from itertools import product


def generate_matrices(n, m):
    matrices = []

    # 生成前 n-1 行的所有可能
    for zeros_positions in product([None] + list(range(m)), repeat=(n - 1)):
        matrix = np.ones((n, m), dtype=int)

        for i, pos in enumerate(zeros_positions):
            if pos is not None:
                matrix[i, pos] = 0

        # 生成最后一行
        if m == 1:
            matrix[-1, :] = 0
            matrices.append(matrix.copy())
        else:
            matrix[-1, :-1] = 1
            for last_row_last_element in [0, 1]:
                matrix[-1, -1] = last_row_last_element
                matrices.append(matrix.copy())

    return matrices


print()

# 示例：生成 3 行 1 列的矩阵
n = 4
m = 1
matrices = generate_matrices(n, m)

# 打印所有生成的矩阵
for matrix in matrices:
    print(matrix)
    print()