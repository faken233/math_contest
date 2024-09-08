import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import generate_data_q1 as gd

# 配置 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示符号


# 抽样方法：简单随机抽样
def Simple_Random_Sampling(data_set, sample_size):
    if sample_size > len(data_set):
        raise ValueError("样本量大于数据集总量")
    return random.sample(data_set, sample_size)


# 抽样方法：系统抽样
def Systematic_Sampling(data_set, sample_size):
    nc = len(data_set)
    if sample_size > nc:
        raise ValueError("样本量大于数据集总量")

    step = nc // sample_size
    start = random.randint(0, step - 1)
    sample = [data_set[i] for i in range(start, nc, step)]

    while len(sample) < sample_size:
        sample.append(data_set[len(sample) % nc])  # 补足样本

    return sample


"""
# 抽样方法：两阶段动态抽样
def Dynamic_Two_Stage_Sampling(data_set, sample_size, first_stage_size_range, second_stage_size_range):
    first_stage_size = random.choice(first_stage_size_range)
    if first_stage_size > len(data_set):
        raise ValueError("第一阶段样本量大于数据集总量")

    first_stage_sample = random.sample(data_set, first_stage_size)
    first_stage_defect_rate = sum(first_stage_sample) / first_stage_size

    if first_stage_defect_rate > 0.1:
        second_stage_size = random.choice(second_stage_size_range)
        second_stage_sample = random.sample(data_set, second_stage_size)
        final_sample = first_stage_sample + second_stage_sample
    else:
        final_sample = first_stage_sample

    return final_sample
"""


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


# 抽样方法：序贯抽样
def Sequential_Sampling_With_CI(data_set, initial_sample_size, max_sample_size, defect_rate_threshold):
    sample_size = initial_sample_size
    sample = random.sample(data_set, sample_size)
    defect_rate = sum(sample) / sample_size

    while sample_size < max_sample_size:
        # 计算当前样本的置信区间
        lower_bound, upper_bound = get_confidence_interval(sample, defect_rate)

        # 如果置信区间的上界小于缺陷率阈值，则可以停止抽样
        if upper_bound < defect_rate_threshold:
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


# 评估抽样方法，只计算准确率
def evaluate_sampling_method(sampling_method, data_set, sample_sizes, defect_rate_threshold, n_runs=100):
    results = []

    for sample_size in sample_sizes:
        accuracies = []
        rows = len(data_set)

        for _ in range(n_runs):
            correct_count = 0
            total_sample_size = 0

            for i in range(rows):
                data = data_set[i]
                if not isinstance(data, list):
                    data = list(data)  # 确保数据是列表类型

                try:
                    # if sampling_method == Dynamic_Two_Stage_Sampling:
                    #    sample = sampling_method(data, sample_size, first_stage_size_range, second_stage_size_range)
                    if sampling_method == Sequential_Sampling_With_CI:
                        sample, final_defect_rate, lower_bound, upper_bound = sampling_method(
                            data, initial_sample_size, max_sample_size, defect_rate_threshold
                        )
                        sample_size = len(sample)  # 更新样本量
                    else:
                        sample = sampling_method(data, sample_size)
                except Exception as e:
                    print(f"Error in sampling method {sampling_method.__name__}: {e}")
                    continue

                count = sum(sample)
                current_sample_size = len(sample)
                total_sample_size += current_sample_size

                percentage = count / current_sample_size
                correct = is_row_acceptable(data)

                # 判断是否与接受率匹配
                if (percentage < P * E) == correct:
                    correct_count += 1

            avg_sample_size = total_sample_size / rows
            accuracy = correct_count / rows
            accuracies.append(accuracy)

        # 计算平均准确率
        avg_accuracy = np.mean(accuracies)
        results.append((sample_size, avg_accuracy))

    return results


# 判断数据行是否符合接受率
def is_row_acceptable(data):
    return sum(data) / len(data) < P * E


# 结果可视化，去掉时间曲线
def plot_comparison(results_dict):
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 绘制每个抽样方法的准确率结果
    for (method_name, results), color in zip(results_dict.items(), colors):
        sample_sizes_plot = [result[0] for result in results]
        accuracies_plot = [result[1] for result in results]

        plt.plot(sample_sizes_plot, accuracies_plot, marker='o', linestyle='-', label=f"{method_name} - 准确率",
                 color=color)

    plt.title('不同抽样方法的比较（准确率）')
    plt.xlabel('样本量')
    plt.ylabel('准确率')
    plt.grid(True)
    plt.legend()
    plt.savefig('comparison_plot.png')
    plt.show()


# 主程序
if __name__ == "__main__":
    P = 0.1
    rows, cols = 1000, 10000
    E = 1

    matrix = gd.generate_matrix(rows, cols)
    acceptance_array = gd.generate_acceptance_array(matrix)

    sample_sizes = [10, 50, 100, 200, 300]  # 样本量范围
    n_runs = 10  # 运行次数

    choice_functions = {
        'Simple_Random_Sampling': Simple_Random_Sampling,
        'Systematic_Sampling': Systematic_Sampling,
        # 'Dynamic_Two_Stage_Sampling': Dynamic_Two_Stage_Sampling,
        'Sequential_Sampling_With_CI': Sequential_Sampling_With_CI
    }

    # Dynamic_Two_Stage_Sampling的参数
    first_stage_size_range = range(50, 201)
    second_stage_size_range = range(50, 201)

    # Sequential_Sampling的参数
    initial_sample_size = 10
    max_sample_size = 500
    defect_rate_threshold = 0.1

    results_dict = {}

    for method_name, sampling_method in choice_functions.items():
        if method_name == 'Dynamic_Two_Stage_Sampling' or method_name == 'Sequential_Sampling_With_CI':
            results = evaluate_sampling_method(
                sampling_method, matrix, sample_sizes, defect_rate_threshold, n_runs
            )
        else:
            results = evaluate_sampling_method(
                sampling_method, matrix, sample_sizes, None, n_runs
            )

        results_dict[method_name] = results

    # 绘制比较图
    plot_comparison(results_dict)
