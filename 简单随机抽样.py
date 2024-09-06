import random
import numpy as np
from tqdm import tqdm
import generate_data as gd
from skopt import gp_minimize


'''通过引入蒙特卡洛方法，我们可以在 simple_choice 函数中进行多次迭代，从而获得更稳定的准确率估计。这种方法特别适用于需要多次随机采样的情况，能够有效减少随机性带来的影响。'''
# 蒙特卡洛随机简单抽样
def monte_carlo_simple_choice(sample_sizes, matrix, acceptance_array, rows, P, num_iterations=1):
    accuracy_results = {}

    for sample_size in tqdm(sample_sizes):
        correct_counts = []
        for _ in range(num_iterations):
            correct_count = 0
            for i in range(rows):
                data_set = matrix[i, :].tolist()
                sample = random.sample(data_set, sample_size)
                count = sum(sample)
                percentage = count / sample_size
                correct = acceptance_array[i]
                if (percentage < P) == correct:
                    correct_count += 1
            correct_counts.append(correct_count)

        # 计算平均准确率
        average_correct_count = np.mean(correct_counts)
        accuracy = average_correct_count / rows
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
        sample = random.sample(data_set, int(sample_size))
        count = sum(sample)
        percentage = count / int(sample_size)
        correct = acceptance_array[i]
        if (percentage < P) == correct:
            correct_count += 1
    accuracy = correct_count / rows
    return sample_size * (1 - accuracy)  # 优化目标是最小化样本量，同时保持准确率


def bayesian_optimization(matrix, acceptance_array, P, sample_sizes):
    # 定义样本量的搜索范围
    bounds = [(min(sample_sizes), max(sample_sizes))]  # 使用给定的样本量范围
    res = gp_minimize(lambda x: objective_function(int(x[0]), matrix, acceptance_array, P),
                      bounds,
                      n_calls=50,  # 调用次数
                      random_state=1)
    best_sample_size = int(res.x[0])
    best_accuracy = 1 - res.fun / best_sample_size  # 计算最佳准确率
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

    best_sample_size, best_accuracy = bayesian_optimization(matrix, acceptance_array, P, [200, 1001])
    print("Best Sample Size:", best_sample_size)
    print("Best Accuracy:", best_accuracy)

    # 计算每个 sample_size 的 accuracy
    accuracy_results = monte_carlo_simple_choice(sample_sizes, matrix, acceptance_array, rows, P)
    # 使用 TOPSIS 方法找到最佳 sample_size
    best_sample_size, best_accuracy = topsis(accuracy_results, weights)

    print(f"最佳sample_size: {best_sample_size}, 正确率: {best_accuracy:.2%}")
