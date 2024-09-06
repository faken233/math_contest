import random
import numpy as np
from tqdm import trange
import generate_data as gd
from skopt import gp_minimize


def simple_choice(sample_sizes, matrix, acceptance_array, rows, P):
    # 存储每个sample_size的正确率
    accuracy_results = {}
    for sample_size in sample_sizes:
        correct_count = 0
        for i in range(rows):
            # 获取数组
            data_set = matrix[i, :]
            data_set = data_set.tolist()
            # 使用 random.sample() 函数进行简单随机抽样
            sample = random.sample(data_set, sample_size)
            count = sum(sample)
            percentage = count / sample_size
            correct = acceptance_array[i]
            if (percentage < P) == correct:
                correct_count += 1

        # 计算正确率
        accuracy = correct_count / rows
        accuracy_results[sample_size] = accuracy

    # 找出正确率最高的sample_size
    best_sample_size = max(accuracy_results, key=accuracy_results.get)
    best_accuracy = accuracy_results[best_sample_size]

    print(f"简单随机抽样最佳sample_size: {best_sample_size}, 正确率: {best_accuracy:.2%}")


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
                      [(10, 1000)],  # 搜索范围
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
    matrix = gd.generate_matrix(rows, cols)
    acceptance_array = gd.generate_acceptance_array(matrix, E)

    # 定义要测试的sample_size范围
    sample_sizes = trange(10, 1001)  # 从10到1000

    simple_choice(sample_sizes, matrix, acceptance_array, rows, P)
