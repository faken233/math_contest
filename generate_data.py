import numpy as np

# 1. 生成一个矩阵，矩阵的元素均为0或1
def generate_matrix(rows, cols):
    matrix = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        # 计算每一行中1的数量，大约为10%，浮动不超过5%
        num_ones = int(cols * 0.1) + np.random.randint(-int(cols * 0.05), int(cols * 0.05))
        # 确保num_ones在合理范围内
        num_ones = max(0, min(cols, num_ones))
        # 随机选择num_ones个位置设置为1
        ones_indices = np.random.choice(cols, num_ones, replace=False)
        matrix[i, ones_indices] = 1
    return matrix

# 2. 指定每一行若1占行所有元素的百分之十以内为这一行对应零件可以被接受
def is_row_acceptable(row):
    return np.sum(row) <= len(row) * 0.1

# 3. 生成一个一维数组，表示每一行是否可以被接受
def generate_acceptance_array(matrix):
    return np.array([is_row_acceptable(row) for row in matrix], dtype=int)

# 示例使用
rows = 1000
cols = 10000
matrix = generate_matrix(rows, cols)
acceptance_array = generate_acceptance_array(matrix)

print("生成的矩阵:")
print(matrix)
print("\n每一行是否可以被接受:")
print(acceptance_array)