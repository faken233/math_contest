import numpy as np
from scipy.interpolate import CubicSpline
import generate_data_q2_q3 as gd
import matplotlib.pyplot as plt

# 配置 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示符号

# 变量
p1_list = [0.1, 0.2, 0.1, 0.2, 0.1, 0.05]  # 部件一的次品率
p2_list = [0.1, 0.2, 0.1, 0.2, 0.2, 0.05]  # 部件二的次品率
p3_list = [0.1, 0.2, 0.1, 0.2, 0.1, 0.05]  # 成品在部件一和部件二都是正品时的次品率
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
# b1 部件一不被检测的flag值, 0表示被检查, 1表示不检查, 且列表中如果出现过0, 使得利润最大化的操作则是将0之后的数据改为1. b2以及b3同.
# b2 部件二~~
# b3 是否对成品进行检查
# b4 是否对不合格成品进行拆解
"""
    即: 要么在一开始就检查, 然后全取1, 要么先不检查, 如果回炉则必须检查, 然后后面元素全取0
    [0,1,1,1...]
    [1,0,1,1,1...]
"""

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

    # c3 = min(n, m)
    if n >= m:
        c3 = m
        overflow_1 = n - m
        overflow_2 = 0
    else:
        c3 = n
        overflow_1 = 0
        overflow_2 = m - n

    # 获取成品装配后次品率
    # _p3 = _p1 + (1 - _p1) * _p2 + (1 - _p2) * (1 - _p1) * p3
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
    b4 = 10  # 假设回炉次数为10

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
                # print(matrix)
                profit = func(p1, p2, p3, n, m, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], j, check_1,
                              check_2, check_3, price_assemble, purchase, punish, dismantle)
                results.append(profit)
                if profit > best_profit + 0.1:
                    best_profit = profit
                    best_matrix = matrix
                    best_b4 = j

        # 输出当前情况的最优解
        sort = np.sort(results)
        # print(f"All results: {results}")
        print(f"Best profit: {best_profit / m}")
        print(f"Best strategy matrix:\n {best_matrix}")
        print("\n")

"""
        x = np.arange(len(results))  # 原始 x 值 (点的索引)
        y = np.array(results)  # 原始 y 值 (利润)

        # 使用 Cubic Spline 插值
        cs = CubicSpline(x, y)

        # 生成更多的点来使曲线平滑
        x_smooth = np.linspace(x.min(), x.max(), 1000)  # 生成500个平滑的点
        y_smooth = cs(x_smooth)

        # 绘制平滑曲线
        plt.figure(figsize=(20, 10))
        plt.plot(x_smooth / 1000, y_smooth / m, linestyle='-', color='steelblue')
        plt.xlabel('Size')
        plt.ylabel('Profit')
        plt.title(f'情况{i + 1}的利润曲线')
        plt.grid(True)
        plt.show()
"""
