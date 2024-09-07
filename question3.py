import numpy as np

import generate_data_q2_q3 as gd

n = 100
price_1 = 2   # 部件一的成本
price_2 = 8   # 部件二的成本
price_3 = 12  # 部件三的成本
check_1 = 2   # 部件一的检测成本
check_2 = 3   # 部件二的检测成本
check_3 = 3   # 部件三的检测成本
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

COST = n * (price_1 + price_2 + price_3) # 总成本基础值
"""
    p1-p3: 对应部件的次品率
    n1-n3: 对应部件数
    b1-b3: 是否对相应部件检查
    b4:    是否对半成品检查
    b5:    对检查出的不合格产品是否拆解
"""
b_matrix = None


def func(p1, p2, p3, n1, n2, n3, b1, b2, b3, b4, b5):
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
            a, b = func(_p1, _p2, _p3,
                        unqualified_product_count,
                        unqualified_product_count,
                        unqualified_product_count,
                        b_matrix[0, -b5], b_matrix[1, -b5], b_matrix[2, -b5], b_matrix[3, -b5], b5 - 1)
            c_next_step += a
            unqualified_product_count = b
            return c_next_step, unqualified_product_count
        else:
            return c_next_step, 0
    else:
        c_next_step = c_semi
        return c_next_step, unqualified_product_count
    # 不合格产品处理


if __name__ == '__main__':
    # 情况1
    b5 = 3

    b_matrices = gd.generate_matrix_q3_1(4, b5 + 1)
    res = np.array([])
    for matrix in b_matrices:
        b_matrix = matrix
        a, b = func(p_part, p_part, p_part, n, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], b_matrix[3, 0], b5)
        print(f'c_next_step = {a}, unqualified_product_count = {b}, Cost = {COST}')
        COST = n * (price_1 + price_2 + price_3)

