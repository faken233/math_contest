import numpy as np
from scipy.optimize import linprog

import generate_data_q2_q3 as gd
from deap import base, creator, tools, algorithms
import random

# 部件个数
n = 100
# 部件购买成本
price_1 = 2
price_2 = 8
price_3 = 12
price_4 = 2
price_5 = 8
price_6 = 12
price_7 = 8
price_8 = 12

# 部件检验成本
check_1 = 1
check_2 = 1
check_3 = 2
check_4 = 1
check_5 = 1
check_6 = 2
check_7 = 1
check_8 = 2

p_part  = 0.1 # 部件的次品率
dismantle_semi  = 6   # 半成品拆解成本
check_semi      = 6   # 半成品检测成本
p_semi          = 0.1 # 半成品次品率
dismantle_final = 10  # 成品拆解成本
check_final     = 6   # 成品检测成本
p_final         = 0.1 # 成品次品率
price_assemble  = 8   # 半成品/成品的装配成本

purchase = 200 # 成品售价
punish   = 40  # 调换费用

COST = n * (price_1 + price_2 + price_3 + price_4 + price_5 + price_6 + price_7 + price_8) # 总成本基础值
"""
    p1-p3: 对应部件的次品率
    n1-n3: 对应部件数
    b1-b3: 是否对相应部件检查
    b4:    是否对半成品检查
    b5:    对检查出的不合格产品是否拆解
"""
b_matrix = None


def func_1(p1, p2, p3, n1, n2, n3, b1, b2, b3, b4, b5):
    global COST, check_1, check_2, check_3, price_assemble, b_matrix

    if b1 == 0:
        # 对部件一进行检测, 数量减少, 同时有检测成本
        COST += check_1 * n1
        n1 = n1 * (1.0 - p1)
    if b2 == 0:
        # 对部件二进行检测,
        COST += check_2 * n2
        n2 = n2 * (1.0 - p2)
    if b3 == 0:
        COST += check_3 * n3
        n3 = n3 * (1.0 - p3)

    # 获取最大成品数
    _p1 = p1 * b1
    _p2 = p2 * b2
    _p3 = p3 * b3
    c_semi = min(n1, n2, n3)

    # 获取成品装配后次品率
    _p_semi = (1 - (1 - _p1) * (1 - _p2) * (1 - _p3)) + (1 - _p1) * (1 - _p2) * (1 - _p3) * p_semi

    # 获取装配所需成本
    COST += c_semi * price_assemble

    # 获取合格/不合格产品数
    qualified_product_count = c_semi * (1.0 - _p_semi)
    unqualified_product_count = c_semi - qualified_product_count
    if b4 == 1: # 对于半成品进行检查
        COST += check_semi * c_semi
        c_next_step = qualified_product_count
        if b5 >= 1:
            # 注意回炉重造时的次品率变化
            if _p1 != 0.0:
                _p1 = c_semi * _p1 / unqualified_product_count
            if _p2 != 0.0:
                _p2 = c_semi * _p2 / unqualified_product_count
            if _p3 != 0.0:
                _p3 = c_semi * _p3 / unqualified_product_count

            # 拆解成本
            COST += unqualified_product_count * dismantle_semi

            # 递归模拟回炉
            a, b = func_1(_p1, _p2, _p3,
                        unqualified_product_count,
                        unqualified_product_count,
                        unqualified_product_count,
                        b_matrix[0, -b5], b_matrix[1, -b5], b_matrix[2, -b5], b_matrix[3, -b5], b5 - 1)
            # 累加, 每次回炉都会有新的半成品伴随进入装配成品工序
            c_next_step += a
            # 赋值, 每次回炉都会将已有的不合格产品丢进回炉工序, 每次回炉都会对已有的不合格产品做操作, 此处使用赋值
            unqualified_product_count = b
            return c_next_step, unqualified_product_count
        else:
            # 不回炉, 但是已经检查, 进入下一步工序的半成品为合格品, 没有不合格品混入其中
            return c_next_step, 0
    else:
        # 不回炉也不检查, 所有装配好的半成品进入下一轮工序, 包括次品
        c_next_step = c_semi
        return c_next_step, unqualified_product_count



if __name__ == '__main__':
    b5 = [1]
    for reverse_time in b5:
        b_matrices = gd.generate_matrix_q3_1(4, reverse_time + 1)
        length = len(b_matrices)
        costs = np.array([])
        produce = np.array([])
        defective = np.array([])
        yield_rate = np.array([])
        print(f"-----------reverse_time: {reverse_time}-------------")
        for matrix in b_matrices:
            b_matrix = matrix
            a, b = func_1(p_part, p_part, p_part, n, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], b_matrix[3, 0], reverse_time)
            print(f"produce all: {a:.2f}, unqual: {b:.2f}, COST = {COST:.2f}")
            costs = np.append(costs, COST)
            produce = np.append(produce, a)
            defective = np.append(defective, b)
            yield_rate = np.append(yield_rate, (a - b) / n)
            COST = n * (price_1 + price_2 + price_3)

        cost_max = np.max(costs)
        cost_min = np.min(costs)
        yield_rate_max = np.max(yield_rate)
        yield_rate_min = np.min(yield_rate)

        scores = {}
        w1 = 0.5
        w2 = 0.5
        for i in range(costs.size):
            costs_normalized = (costs[i] - cost_min) / (cost_max - cost_min)
            yields_normalized = (yield_rate[i] - yield_rate_min) / (yield_rate_max - yield_rate_min)
            score = w1 * -costs_normalized + w2 * yields_normalized
            print(f"cost_normalize = {costs_normalized}, yields_normalized = {yields_normalized}, score = {score}")