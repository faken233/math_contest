import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import generate_data_q1 as gd
from skopt import gp_minimize


# 抽样方法：序贯抽样
def Sequential_Sampling_With_CI(data_set, initial_sample_size, max_sample_size, defect_rate_threshold, confidence_level):
    sample_size = initial_sample_size
    sample = random.sample(data_set, sample_size)
    defect_rate = sum(sample) / sample_size

    while sample_size < max_sample_size:
        # 计算当前样本的置信区间
        lower_bound, upper_bound = get_confidence_interval(sample, defect_rate, confidence_level)

        # 根据置信区间判断是否满足条件
        if (confidence_level == 0.95 and upper_bound < defect_rate_threshold) or \
           (confidence_level == 0.90 and lower_bound <= defect_rate_threshold):
            break

        # 增加样本量
        additional_sample_size = min(10, max_sample_size - sample_size)
        additional_sample = random.sample(data_set, additional_sample_size)
        sample.extend(additional_sample)
        sample_size += additional_sample_size
        defect_rate = sum(sample) / sample_size

    # 计算最终样本的缺陷率及其置信区间
    final_defect_rate, lower_bound, upper_bound = extract_defect_rate_and_ci(sample)

    return sample, final_defect_rate, lower_bound, upper_bound


def get_confidence_interval(sample, defect_rate, confidence_level=0.95):
    sample_size = len(sample)
    margin_of_error = norm.ppf((1 + confidence_level) / 2) * np.sqrt((defect_rate * (1 - defect_rate)) / sample_size)
    lower_bound = defect_rate - margin_of_error
    upper_bound = defect_rate + margin_of_error
    return lower_bound, upper_bound


def extract_defect_rate_and_ci(sample):
    sample_size = len(sample)
    defect_count = sum(sample)
    defect_rate = defect_count / sample_size
    lower_bound, upper_bound = get_confidence_interval(sample, defect_rate)
    return defect_rate, lower_bound, upper_bound


# 目标函数
def objective_function(sample_ratio, matrix, acceptance_array, P, initial_sample_size, max_sample_size, defect_rate_threshold, alpha, confidence_level):
    correct_count = 0
    rows = matrix.shape[0]
    total_sample_size = 0

    for i in range(rows):
        data_set = matrix[i, :].tolist()
        sample_size = int(sample_ratio * len(data_set))

        try:
            sample, final_defect_rate, lower_bound, upper_bound = Sequential_Sampling_With_CI(
                data_set, initial_sample_size, max_sample_size, defect_rate_threshold, confidence_level
            )
            sample_size = len(sample)  # 更新样本量
        except Exception as e:
            print(f"Error in sampling method Sequential_Sampling_With_CI: {e}")
            continue

        count = sum(sample)
        current_sample_size = len(sample)
        total_sample_size += current_sample_size

        percentage = count / current_sample_size
        correct = acceptance_array[i]

        # 判断是否与接受率匹配
        if (confidence_level == 0.95 and upper_bound < defect_rate_threshold) or \
           (confidence_level == 0.90 and lower_bound <= defect_rate_threshold):
            correct = True
        else:
            correct = False

        if (percentage < P * E) == correct:
            correct_count += 1

    avg_sample_size = total_sample_size / rows
    accuracy = correct_count / rows

    # 目标函数：样本量与准确率的权衡
    return alpha * avg_sample_size + (1 - alpha) * (1 - accuracy)


# 贝叶斯优化
def bayesian_optimization(matrix, acceptance_array, P, sample_ratios, initial_sample_size, max_sample_size, defect_rate_threshold, alpha, confidence_level):
    if sample_ratios.size == 0:
        raise ValueError("样本比例范围不能为空")

    min_sample_ratio, max_sample_ratio = min(sample_ratios), max(sample_ratios)

    # 如果最小样本比例等于最大样本比例，直接返回该比例
    if min_sample_ratio == max_sample_ratio:
        return min_sample_ratio, 1.0, [], []

    bounds = [(min_sample_ratio, max_sample_ratio)]

    res = gp_minimize(
        lambda x: objective_function(x[0], matrix, acceptance_array, P, initial_sample_size, max_sample_size, defect_rate_threshold, alpha, confidence_level),
        bounds, n_calls=50, random_state=1
    )

    best_sample_ratio = res.x[0]
    best_accuracy = 1 - res.fun / (best_sample_ratio * len(matrix[0]))
    return best_sample_ratio, best_accuracy, res.func_vals, res.x_iters


# 主程序
if __name__ == "__main__":
    P = 0.1
    rows, cols = 1000, 10000
    E = 1

    matrix = gd.generate_matrix(rows, cols)
    acceptance_array = gd.generate_acceptance_array(matrix)

    sample_ratios = np.linspace(0.01, 0.1, 10)  # 样本比例范围，从0.01到0.1

    # Sequential_Sampling的参数
    initial_sample_size = 10
    max_sample_size = 500
    defect_rate_threshold = 0.1
    alpha = 0.1

    results = {}

    for confidence_level in [0.95, 0.90]:
        try:
            best_sample_ratio, best_accuracy, func_vals, x_iters = bayesian_optimization(
                matrix, acceptance_array, P, sample_ratios, initial_sample_size, max_sample_size, defect_rate_threshold, alpha, confidence_level
            )

            # 处理结果并记录
            results[confidence_level] = (best_sample_ratio, best_accuracy)

        except Exception as e:
            print(f"Error during optimization for Confidence Level {confidence_level}: {e}")

    print("最终结果:")
    for confidence_level, (ratio, accuracy) in results.items():
        print(f"信度 {confidence_level:.2f}: 最佳样本比例 = {ratio:.4f}, 正确率 = {accuracy:.2%}")

"""
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(12, 8))
plt.bar([32, 64], [0.9261, 0.9976], align='center', width=20)
plt.xticks([0, 32, 64, 96])
plt.ylim(0.8, 1.01)

# 在 y=0.95 和 y=0.90 处画横向虚线
plt.axhline(y=0.95, color='blue', linestyle='--', label='α = 0.95')
plt.axhline(y=0.90, color='red', linestyle='--', label='α = 0.90')

# 添加图例
plt.legend(fontsize='large')

plt.show()
"""
