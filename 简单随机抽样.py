import random
import numpy as np
from tqdm import tqdm
import generate_data as gd
from skopt import gp_minimize


# 简单随机抽样
def simple_choice(sample_sizes, matrix, acceptance_array, rows, P):
    accuracy_results = {}
    for sample_size in tqdm(sample_sizes):
        correct_count = 0
        for i in range(rows):
            data_set = matrix[i, :].tolist()
            sample = random.sample(data_set, sample_size)
            count = sum(sample)
            percentage = count / sample_size
            correct = acceptance_array[i]
            if (percentage < P) == correct:
                correct_count += 1
        accuracy = correct_count / rows
        accuracy_results[sample_size] = accuracy
    return accuracy_results


# topsis评价方法
def topsis(accuracy_results, weights):
    # 构建决策矩阵
    decision_matrix = np.array([[sample_size, accuracy] for sample_size, accuracy in accuracy_results.items()])

    # 标准化决策矩阵
    normalized_matrix = decision_matrix / np.linalg.norm(decision_matrix, axis=0)

    # 计算加权标准化决策矩阵
    weighted_normalized_matrix = normalized_matrix * weights

    # 确定正理想解和负理想解
    ideal_best = np.max(weighted_normalized_matrix, axis=0)
    ideal_worst = np.min(weighted_normalized_matrix, axis=0)

    # 计算每个方案到正理想解和负理想解的距离
    distance_to_ideal_best = np.linalg.norm(weighted_normalized_matrix - ideal_best, axis=1)
    distance_to_ideal_worst = np.linalg.norm(weighted_normalized_matrix - ideal_worst, axis=1)

    # 计算每个方案的相对接近度
    relative_closeness = distance_to_ideal_worst / (distance_to_ideal_worst + distance_to_ideal_best)

    # 找到最佳方案
    best_index = np.argmax(relative_closeness)
    best_sample_size = sample_sizes[best_index]
    best_accuracy = accuracy_results[best_sample_size]

    return best_sample_size, best_accuracy


def objective_function(sample_size, matrix, acceptance_array, P):
    rows = matrix.shape[0]
    correct_count = 0
    for i in range(rows):
        data_set = matrix[i, :].tolist()
        sample = random.sample(data_set, int(sample_size[0]))
        count = sum(sample)
        percentage = count / int(sample_size[0])
        correct = acceptance_array[i]
        if (percentage < P) == correct:
            correct_count += 1
    return -correct_count / rows  # 负号是因为gp_minimize默认最小化目标函数


def bayesian_optimization(matrix, acceptance_array, P, sample_sizes):
    res = gp_minimize(lambda x: objective_function(x, matrix, acceptance_array, P),
                      [(10, 100)],  # 搜索范围
                      n_calls=50,  # 调用次数
                      random_state=1)
    best_sample_size = int(res.x[0])
    best_accuracy = -res.fun
    return best_sample_size, best_accuracy


if __name__ == "__main__":
    # 设置数据矩阵
    P = 0.1
    rows = 1000
    cols = 10000
    E = 0  # 置信度
    weights = np.array([0.9, 0.1])
    matrix = gd.generate_matrix(rows, cols)
    acceptance_array = gd.generate_acceptance_array(matrix, E)

    # 定义要测试的sample_size范围
    sample_sizes = range(200, 1001)  # 从200到1000

    # 计算每个 sample_size 的 accuracy
    accuracy_results = simple_choice(sample_sizes, matrix, acceptance_array, rows, P)

    # 使用 TOPSIS 方法找到最佳 sample_size
    best_sample_size, best_accuracy = topsis(accuracy_results, weights)

    print(f"最佳sample_size: {best_sample_size}, 正确率: {best_accuracy:.2%}")
