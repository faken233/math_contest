import numpy as np
import generate_data_q2_q3 as gd

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

p_part = 0.1  # 部件的次品率
dismantle_semi = 6  # 半成品拆解成本
check_semi = 6  # 半成品检测成本
p_semi = 0.1  # 半成品次品率
dismantle_final = 10  # 成品拆解成本
check_final = 6  # 成品检测成本
p_final = 0.1  # 成品次品率
price_assemble = 8  # 半成品/成品的装配成本

purchase = 200  # 成品售价
punish = 40  # 调换费用

COST1 = COST2 = n * (price_1 + price_2 + price_3)  # 半成品一 二的总成本基础值
COST3 = n * (price_7 * price_8)
"""
    p1-p3: 对应部件的次品率
    n1-n3: 对应部件数
    b1-b3: 是否对相应部件检查
    b4:    是否对半成品检查
    b5:    对检查出的不合格产品是否拆解
"""


def func_1(p1, p2, p3, n1, n2, n3, b1, b2, b3, b4, b5):
    global COST1, check_1, check_2, check_3, price_assemble, b_matrix

    if b1 == 0:
        # 对部件一进行检测, 数量减少, 同时有检测成本
        COST1 += check_1 * n1
        n1 = n1 * (1.0 - p1)
    if b2 == 0:
        # 对部件二进行检测,
        COST1 += check_2 * n2
        n2 = n2 * (1.0 - p2)
    if b3 == 0:
        COST1 += check_3 * n3
        n3 = n3 * (1.0 - p3)

    # 获取最大成品数
    _p1 = p1 * b1
    _p2 = p2 * b2
    _p3 = p3 * b3
    c_semi = min(n1, n2, n3)

    # 获取成品装配后次品率
    _p_semi = (1 - (1 - _p1) * (1 - _p2) * (1 - _p3)) + (1 - _p1) * (1 - _p2) * (1 - _p3) * p_semi

    # 获取装配所需成本
    COST1 += c_semi * price_assemble

    # 获取合格/不合格产品数
    qualified_product_count = c_semi * (1.0 - _p_semi)
    unqualified_product_count = c_semi - qualified_product_count
    if b4 == 1:  # 对于半成品进行检查
        COST1 += check_semi * c_semi
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
            COST1 += unqualified_product_count * dismantle_semi

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


def func_2(p1, p2, n1, n2, b1, b2, b3, b4):
    global COST3, check_7, check_8, price_assemble, b_matrix

    if b1 == 0:
        # 对部件七进行检测, 数量减少, 同时有检测成本
        COST3 += check_7 * n1
        n1 = n1 * (1.0 - p1)
    if b2 == 0:
        # 对部件八进行检测,
        COST3 += check_8 * n2
        n2 = n2 * (1.0 - p2)

    # 获取最大成品数
    _p1 = p1 * b1
    _p2 = p2 * b2
    c_semi = min(n1, n2)

    _p_semi = (1 - (1 - _p1) * (1 - _p2)) + (1 - _p1) * (1 - _p2) * p_semi

    COST3 += c_semi * price_assemble

    # 获取合格/不合格产品数
    qualified_product_count = c_semi * (1.0 - _p_semi)
    unqualified_product_count = c_semi - qualified_product_count
    if b3 == 1:  # 对于半成品进行检查
        COST3 += check_semi * c_semi
        c_next_step = qualified_product_count
        if b4 >= 1:
            # 注意回炉重造时的次品率变化
            if _p1 != 0.0:
                _p1 = c_semi * _p1 / unqualified_product_count
            if _p2 != 0.0:
                _p2 = c_semi * _p2 / unqualified_product_count

            # 拆解成本
            COST3 += unqualified_product_count * dismantle_semi

            # 递归模拟回炉
            a, b = func_2(_p1, _p2,
                          unqualified_product_count,
                          unqualified_product_count,
                          b_matrix[0, -b4], b_matrix[1, -b4], b_matrix[2, -b4], b4 - 1)
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


def calculate_weights(comparison_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    weights = eigenvectors[:, max_eigenvalue_index].real
    weights /= weights.sum()
    return weights


def calculate_score(decision_matrix, weights):
    return np.dot(decision_matrix, weights)


if __name__ == '__main__':
    # global COST1, b_matrix, COST3  # 如果零件为三合一用COST1，其次用COST3
    b5 = [0, 1, 2]
    costs = np.array([])
    produce = np.array([])
    defective = np.array([])
    yield_rate = np.array([])
    mat = []
    for reverse_time in b5:
        b_matrices = gd.generate_matrix_q3_1(3, reverse_time + 1)
        length = len(b_matrices)

        for matrix in b_matrices:
            b_matrix = matrix

            # a, b = func_1(p_part, p_part, p_part, n, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], b_matrix[3, 0], reverse_time)
            a, b = func_2(p_part, p_part, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], reverse_time)

            # costs = np.append(costs, COST1)
            costs = np.append(costs, COST3)

            produce = np.append(produce, a)
            defective = np.append(defective, b)
            yield_rate = np.append(yield_rate, (a - b) / n)
            mat.append(b_matrix)

            # COST1 = n * (price_1 + price_2 + price_3)
            COST3 = n * (price_7 + price_8)

    # 标准化处理
    max_cost = np.max(costs)
    min_yield = np.min(yield_rate)
    max_defective = np.max(defective)

    normalized_costs = -costs / max_cost
    normalized_yield_rate = yield_rate / min_yield
    normalized_defective = -defective / max_defective

    # 构建标准化后的决策矩阵
    normalized_decision_matrix = np.array([normalized_costs, normalized_yield_rate, normalized_defective]).T


    # 比较矩阵
    comparison_matrix = np.array([
        [5,     4    , 1],  # 成本5 vs 良率4, 成本5 vs 次品率1
        [1 / 4, 1    , 4],  # 良率 vs 成本, 良率 vs 次品率
        [1    , 1 / 4, 1/5]  # 次品率 vs 成本, 次品率 vs 良率
    ])

    weights = calculate_weights(comparison_matrix)
    print("Weights:", weights)

    scores = calculate_score(normalized_decision_matrix, weights)
    sorted_indices = np.argsort(scores)[::-1]
    print(sorted_indices)

    # 输出结果
    for i, index in enumerate(sorted_indices):
        cost = costs[index]
        yield_rate_value = yield_rate[index]
        defective_value = defective[index]
        produce_value = produce[index]

        print(f"Rank {i+1}: Matrix {index}, Score: {scores[index]:.4f}")
        print(f"Cost: {cost / produce_value:.2f} per one, Yield Rate: {yield_rate_value * 100.0:.3f}%, Defective: {defective_value:.3f}%, Produce: {produce_value:.2f}")
        print(mat[index])
        print("=====================================")


def main():
    global COST1, b_matrix, COST3  # 如果零件为三合一用COST1，其次用COST3
    b5 = [0, 1, 2]
    costs = np.array([])
    produce = np.array([])
    defective = np.array([])
    yield_rate = np.array([])
    mat = []
    for reverse_time in b5:
        b_matrices = gd.generate_matrix_q3_1(3, reverse_time + 1)
        length = len(b_matrices)

        for matrix in b_matrices:
            b_matrix = matrix

            # a, b = func_1(p_part, p_part, p_part, n, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], b_matrix[3, 0], reverse_time)
            a, b = func_2(p_part, p_part, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], reverse_time)

            # costs = np.append(costs, COST1)
            costs = np.append(costs, COST3)

            produce = np.append(produce, a)
            defective = np.append(defective, b)
            yield_rate = np.append(yield_rate, (a - b) / n)
            mat.append(b_matrix)

            # COST1 = n * (price_1 + price_2 + price_3)
            COST3 = n * (price_7 + price_8)

    # 标准化处理
    max_cost = np.max(costs)
    min_yield = np.min(yield_rate)
    max_defective = np.max(defective)

    normalized_costs = -costs / max_cost
    normalized_yield_rate = yield_rate / min_yield
    normalized_defective = -defective / max_defective

    # 构建标准化后的决策矩阵
    normalized_decision_matrix = np.array([normalized_costs, normalized_yield_rate, normalized_defective]).T


    # 比较矩阵
    comparison_matrix = np.array([
        [5,     4    , 1],  # 成本5 vs 良率4, 成本5 vs 次品率1
        [1 / 4, 1    , 4],  # 良率 vs 成本, 良率 vs 次品率
        [1    , 1 / 4, 1/5]  # 次品率 vs 成本, 次品率 vs 良率
    ])

    weights = calculate_weights(comparison_matrix)
    print("Weights:", weights)

    scores = calculate_score(normalized_decision_matrix, weights)
    sorted_indices = np.argsort(scores)[::-1]

    return sorted_indices
