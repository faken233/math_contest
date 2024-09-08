import itertools

import numpy as np


def generate_matrix_q3_1(n, m):
    matrices = []

    # 生成前 n-1 行的所有可能
    for zeros_positions in itertools.product([None] + list(range(m)), repeat=(n - 1)):
        matrix = np.ones((n, m), dtype=int)

        # 设置前 n-1 行的零元素
        for i, pos in enumerate(zeros_positions):
            if pos is not None:
                matrix[i, pos] = 0

        # 生成倒数第一行
        if m == 1:
            matrix[-1, :] = 0
            matrices.append(matrix.copy())
        else:
            matrix[-1, :-1] = 1
            for last_row_last_element in [0, 1]:
                matrix[-1, -1] = last_row_last_element
                matrices.append(matrix.copy())

    return matrices

def generate_matrix_q3_2(n):
    # 生成所有可能的第一到第三
    rows_1_2_3 = []
    for i in range(n + 1):  # 0到n个0的情况
        row = [1] * n
        if i > 0:
            row[i - 1] = 0
        rows_1_2_3.append(row)

    # 生成所有可能的第三行
    rows_4 = list(itertools.product([0, 1], repeat=n))

    # 组合生成所有矩阵
    matrices = []
    for row1 in rows_1_2_3:
        for row2 in rows_1_2_3:
            for row3 in rows_4:
                for row4 in rows_4:
                    matrix = np.array([row1, row2, row3, row4])
                    matrices.append(matrix)
    return matrices