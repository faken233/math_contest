import matplotlib.pyplot as plt
import numpy as np

import generate_data_q2 as gd

# 配置 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示符号

# 变量
p1_list = [0.07, 0.12, 0.09, 0.11, 0.06, 0.14]  # 部件一的次品率, 随机抽样得到
p2_list = [0.08, 0.13, 0.10, 0.05, 0.15, 0.09] # 部件二的次品率, 随机抽样得到
p3_list = [0.1, 0.2, 0.1, 0.2, 0.1, 0.05]  # 成品在部件一和部件二都是正品时的次品率, 为固定值
n = m = 10000
price_1 = [4, 4, 4, 4, 4, 4]  # 部件一的成本
price_2 = [18, 18, 18, 18, 18, 18]  # 部件二的成本
check_1_list = [2, 2, 2, 1, 8, 2]  # 部件一的检测成本
check_2_list = [3, 3, 3, 1, 1, 3]  # 部件二的检测成本
check_3_list = [3, 3, 3, 2, 2, 3]  # 成品的检测成本
price_assemble_list = [6, 6, 6, 6, 6, 6]  # 装配成本
purchases = [56, 56, 56, 56, 56, 56]  # 成品售价
punish_list = [6, 6, 30, 30, 10, 10]  # 惩罚
dismantle_list = [5, 5, 5, 5, 5, 40]  # 拆解成本
W = 0  # 总收益
COST = price_1 * n + price_2 * m  # 总成本

b_matrix = None


def func(p1, p2, p3, n, m, b1, b2, b3, b4, check_1, check_2, check_3, price_assemble, purchase, punish, dismantle):
    COST = price_1[0] * n + price_2[0] * m
    W = 0

    if b1 == 0:
        # 对部件一进行检测, 数量减少, 同时有检测成本
        COST += check_1 * n
        n = n * (1.0 - p1)
    if b2 == 0:
        # 对部件二进行检测, 数量减少, 同时有检测成本
        COST += check_2 * m
        m = m * (1.0 - p2)

    # 获取最大成品数
    _p1 = p1 * b1
    _p2 = p2 * b2

    if n >= m:
        c3 = m
        overflow_1 = n - m
        overflow_2 = 0
    else:
        c3 = n
        overflow_1 = 0
        overflow_2 = m - n

    # 获取成品装配后次品率
    _p3 = (1 - (1 - _p2) * (1 - _p1)) + (1 - _p2) * (1 - _p1) * p3
    COST += c3 * price_assemble

    # 获取合格产品数
    qualified_product_count = c3 * (1.0 - _p3)
    if b3 == 1:
        COST += check_3 * c3
        W += qualified_product_count * purchase
    else:
        W += qualified_product_count * purchase
        COST += c3 * _p3 * punish

    # 不合格产品处理
    if b4 >= 1:
        unqualified_product_count = c3 - qualified_product_count
        if _p1 != 0.0:
            _p1 = (c3 * _p1 + overflow_1 * p1) / (unqualified_product_count + overflow_1)
        if _p2 != 0.0:
            _p2 = (c3 * _p2 + overflow_2 * p2) / (unqualified_product_count + overflow_2)

        COST += unqualified_product_count * dismantle

        # 递归模拟回炉
        return W - COST + func(_p1, _p2, p3, unqualified_product_count + overflow_1,
                               unqualified_product_count + overflow_2, b_matrix[0, -b4], b_matrix[1, -b4],
                               b_matrix[2, -b4], b4 - 1,
                               check_1, check_2, check_3, price_assemble, purchase, punish, dismantle)
    else:
        return W - COST


if __name__ == '__main__':
    b4 = 3  # 回炉次数为

    # 遍历 p1, p2, p3 的每种组合情况
    for i in range(len(p1_list)):
        p1 = p1_list[i]
        p2 = p2_list[i]
        p3 = p3_list[i]
        check_1 = check_1_list[i]
        check_2 = check_2_list[i]
        check_3 = check_3_list[i]
        price_assemble = price_assemble_list[i]
        purchase = purchases[i]
        punish = punish_list[i]
        dismantle = dismantle_list[i]

        print(f"Scenario {i + 1}: p1 = {p1}, p2 = {p2}, p3 = {p3}")
        best_profit = -np.inf
        best_matrix = None
        best_b4 = None
        results = []

        for j in range(b4):
            print(f"  Trying with {j} rework steps...")
            b_matrices = gd.generate_matrix_q2(j + 1)

            # 遍历每种 b_matrix
            for matrix in b_matrices:
                b_matrix = matrix
                profit = func(p1, p2, p3, n, m, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], j, check_1,
                              check_2, check_3, price_assemble, purchase, punish, dismantle)
                results.append(profit)
                if profit > best_profit + 0.1:
                    best_profit = profit
                    best_matrix = matrix
                    best_b4 = j

        # 输出当前情况的最优解
        sort = np.sort(results)
        print(f"Best profit: {best_profit / m}")
        print(f"Best strategy matrix:\n {best_matrix}")
        print("\n")

