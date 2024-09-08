import numpy as np
import itertools
from itertools import product

def generate_matrix_q2(n):
    # 生成所有可能的第一行和第二行
    rows_1_2 = []
    for i in range(n + 1):  # 0到n个0的情况
        row = [1] * n
        if i > 0:
            row[i - 1] = 0
        rows_1_2.append(row)

    # 生成所有可能的第三行
    rows_3 = list(itertools.product([0, 1], repeat=n))

    # 组合生成所有矩阵
    matrices = []
    for row1 in rows_1_2:
        for row2 in rows_1_2:
            for row3 in rows_3:
                matrix = np.array([row1, row2, row3])
                matrices.append(matrix)
    return matrices
