import numpy as np
import generate_data_q2_q3 as gd

# 变量
p_intermediate_1 = p_intermediate_2 = 0.009 # 半成品一和二的所有方案次品率
p_intermediate_3 = 0.009 # 半成品三的所有方案次品率

cost_per_intermediate_1 = cost_per_intermediate_2 = 44.29 # 一和二的每件半成品成本
cost_per_intermediate_3 = 40.96
p_product      = 0.1 # 成品在半成品都是正品时的次品率
n = 10000 # 半成品量

check_product = 6  # 成品的检测成本
check_intermediate_product = 4  # 半成品的检测成本


price_assemble = 8  # 装配成本
purchase = 200  # 成品售价
punish = 40  # 惩罚
dismantle = 10  # 拆解成本
W = 0    # 总收益
COST = 0 # 总成本

def func(p1, p2, p3, n1, n2, n3, b1, b2, b3, b4, reverse_time):
    global COST, W, check_1, check_2, check_3, price_assemble, purchase, punish, b_matrix

    if b1 == 0:
        # 对部件一进行检测, 数量减少, 同时有检测成本
        COST += check_intermediate_product * n1
        n1 = n1 * (1.0 - p1)
    if b2 == 0:
        # 对部件二进行检测, 数量减少, 同时有检测成本
        COST += check_intermediate_product * n2
        n2 = n2 * (1.0 - p2)
    if b3 == 0:
        COST += check_intermediate_product * n3
        n3 = n3 * (1.0 - p3)

    # 获取最大成品数
    _p1 = p1 * b1
    _p2 = p2 * b2
    _p3 = p3 * b3

    c_product = min(n1, n2, n3)
    overflow_1 = 0
    overflow_2 = 0
    overflow_3 = 0

    if c_product == n1:
        overflow_1 = 0
        overflow_2 = n2 - n1
        overflow_3 = n3 - n1
    elif c_product == n2:
        overflow_2 = 0
        overflow_3 = n3 - n2
        overflow_1 = n1 - n2
    elif c_product == n3:
        overflow_3 = 0
        overflow_1 = n1 - n2
        overflow_2 = n2 - n3

    # 获取成品装配后次品率
    _p_product = (1 - (1 - _p3) * (1 - _p2) * (1 - _p1)) + (1 - _p3) * (1 - _p2) * (1 - _p1) * p_product
    COST += c_product * price_assemble

    # 获取合格产品数
    qualified_product_count = c_product * (1.0 - _p_product)
    if b4 == 1:  # 对成品进行检查
        # 合格产品直接收益
        COST += check_product * c_product
        W += qualified_product_count * purchase
    else:
        # 不检查可以从所有产品处获得收益, 但是有调换损失
        W += qualified_product_count * purchase
        COST += c_product * _p_product * punish

    # 不合格产品处理
    if reverse_time >= 1:
        unqualified_product_count = c_product - qualified_product_count
        # 注意回炉重造时的次品率变化
        if _p1 != 0.0:
            _p1 = (c_product * _p1 + overflow_1 * p1) / (unqualified_product_count + overflow_1)
        if _p2 != 0.0:
            _p2 = (c_product * _p2 + overflow_2 * p2) / (unqualified_product_count + overflow_2)
        if _p3 != 0.0:
            _p3 = (c_product * _p3 + overflow_3 * p3) / (unqualified_product_count + overflow_3)

        # 拆解成本
        COST += unqualified_product_count * dismantle

        # 递归模拟回炉
        func(_p1, _p2, _p3, unqualified_product_count + overflow_1, unqualified_product_count + overflow_2, unqualified_product_count + overflow_3,b_matrix[0, -reverse_time], b_matrix[1, -reverse_time], b_matrix[2, -reverse_time], b_matrix[3, -reverse_time], reverse_time - 1)
    else:
        return


if __name__ == '__main__':
    '''
        5次 515732
        4从 515691
        3次 515280
        2次 511174
        1次 492843
        0次 326803
    '''
    b4 = 5  # 回炉次数

    print(f"p1 = {p_intermediate_1}, p2 = {p_intermediate_2}, p3 = {p_intermediate_3}")
    b_matrices = gd.generate_matrix_q3_2(b4 + 1)
    res = np.array([])
    map = {}

    # 遍历每种 b_matrix
    for matrix in b_matrices:
        b_matrix = matrix
        func(p_intermediate_1, p_intermediate_2, p_intermediate_3, n, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], b_matrix[3, 0], reverse_time=b4)
        profit = W - COST - n * (cost_per_intermediate_1 * 2 + cost_per_intermediate_3)
        res = np.append(res, profit)
        map[str(profit)] = b_matrix
        W = 0
        COST = 0

    # 输出当前情况的最优解
    sort = np.sort(res)
    best_profit = sort[-1]
    best_matrix = map[str(best_profit)]
    print(f"All results: {sort}")
    print(f"Best profit: {best_profit:.0f}")
    print(f"Best strategy matrix:\n {best_matrix}")
    print("\n")
