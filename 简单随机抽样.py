import random
import numpy as np
from tqdm import trange

# 假设有一个数据集
# 定义数组长度和1的比例
n = 1000  # 数组长度
P = 0.1  # 1的比例
E = 0.05  # 误差范围

# 定义要测试的sample_size范围
sample_sizes = range(50, 500)  # 从10到100，步长为10

# 定义抽样次数
num_samples = 1000

# 存储每个sample_size的正确率
accuracy_results = {}

for sample_size in sample_sizes:
    correct_count = 0
    for _ in trange(num_samples):
        # 生成数组
        data_set = np.random.binomial(1, P, n)
        data_set = data_set.tolist()
        # 使用 random.sample() 函数进行简单随机抽样
        sample = random.sample(data_set, sample_size)
        count = sum(sample)
        percentage = count / sample_size

        # 判断是否在误差范围内
        if abs(percentage - P) <= E:
            correct_count += 1

    # 计算正确率
    accuracy = correct_count / num_samples
    accuracy_results[sample_size] = accuracy

# 找出正确率最高的sample_size
best_sample_size = max(accuracy_results, key=accuracy_results.get)
best_accuracy = accuracy_results[best_sample_size]

print(f"最佳sample_size: {best_sample_size}, 正确率: {best_accuracy:.2%}")
