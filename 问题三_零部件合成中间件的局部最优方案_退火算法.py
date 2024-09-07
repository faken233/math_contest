import math
import random

import generate_data_q2_q3 as gd


# 定义退火算法的参数
initial_temperature = 1000  # 初始温度
cooling_rate = 0.95  # 每次迭代降温的比例
min_temperature = 1e-3  # 最低温度，达到此温度后停止迭代
max_iterations = 1000  # 最大迭代次数


# 部件个数
n = m = 100
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
check_semi      = 4   # 半成品检测成本
p_semi          = 0.1 # 半成品次品率
dismantle_final = 10  # 成品拆解成本
check_final     = 6   # 成品检测成本
p_final         = 0.1 # 成品次品率
price_assemble  = 8   # 半成品/成品的装配成本

purchase = 200 # 成品售价
punish   = 40  # 调换费用

COST1 = COST2 = n * (price_1 + price_2 + price_3) # 半成品一 二的总成本基础值
COST3 = n * (price_7 + price_8)
"""
    p1-p3: 对应部件的次品率
    n1-n3: 对应部件数
    b1-b3: 是否对相应部件检查
    b4:    是否对半成品检查
    b5:    对检查出的不合格产品是否拆解
"""


# 退火算法的核心函数
def evaluate_solution1(n, p_part, reverse_time):
    global COST1, b_matrix
    a, b = func_1(p_part, p_part, p_part, n, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], b_matrix[3, 0],
                  reverse_time)
    current_cost = COST1
    current_yield = (a - b) / n
    current_defective = b / a
    COST1 = n * (price_1 + price_2 + price_3)  # 重置成本
    return current_cost, current_yield, current_defective


def evaluate_solution2(n, p_part, reverse_time):
    global COST3, b_matrix
    a, b = func_2(p_part, p_part, n, n, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0],
                  reverse_time)
    current_cost = COST3
    current_yield = (a - b) / n
    current_defective = b / a
    COST3 = n * (price_1 + price_2 + price_3)  # 重置成本
    return current_cost, current_yield, current_defective


# 评估当前解的成本和良率
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
    if b3 == 1: # 对于半成品进行检查
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


def simulated_annealing1(b_matrices, n, p_part, reverse_time, max_iterations, initial_temperature, cooling_rate,
                        min_temperature):
    global b_matrix

    # 初始化解：随机选择一个策略矩阵
    current_solution = random.choice(b_matrices)
    b_matrix = current_solution
    current_cost, current_yield, current_defective = evaluate_solution1(n, p_part, reverse_time)

    best_solution = current_solution
    best_cost = current_cost
    best_yield = current_yield
    best_defective = current_defective

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # 降温
        temperature = temperature * cooling_rate
        if temperature < min_temperature:
            break

        # 随机选择一个邻近解（即新的策略矩阵）
        new_solution = random.choice(b_matrices)
        b_matrix = new_solution
        new_cost, new_yield, new_defective = evaluate_solution1(n, p_part, reverse_time)

        # 计算成本差异
        cost_diff = new_cost - current_cost
        yield_diff = new_yield - current_yield
        defective_diff = new_defective - current_defective

        # 可以通过调整权重 α, β, γ 来控制各个目标对接受概率的影响
        alpha, beta, gamma = 0.45, 0.45, 0.1  # 权重值可以根据实际需要调整
        objective_diff = alpha * cost_diff - beta * yield_diff + gamma * defective_diff

        # 如果新解更优，接受它；如果更差，按概率接受（考虑到 objective_diff 和温度的关系）
        if objective_diff < 0 or math.exp(-objective_diff / temperature) > random.random():
            current_solution = new_solution
            current_cost = new_cost
            current_yield = new_yield
            current_defective = new_defective

        # 更新最优解
        if new_cost < best_cost and new_yield > best_yield and new_defective < best_defective:
            best_solution = new_solution
            best_cost = new_cost
            best_yield = new_yield
            best_defective = new_defective

        # 打印当前迭代的信息
        # print(
        #     f"Iteration {iteration + 1}: Temperature = {temperature:.4f}, Current Cost = {current_cost:.2f}, Best Cost = {best_cost:.2f}, Yield Rate = {best_yield:.2f}")

    return best_solution, best_cost, best_yield, best_defective


def simulated_annealing2(b_matrices, n, p_part, reverse_time, max_iterations, initial_temperature, cooling_rate,
                        min_temperature):
    global b_matrix

    # 初始化解：随机选择一个策略矩阵
    current_solution = random.choice(b_matrices)
    b_matrix = current_solution
    current_cost, current_yield, current_defective = evaluate_solution2(n, p_part, reverse_time)

    best_solution = current_solution
    best_cost = current_cost
    best_yield = current_yield
    best_defective = current_defective

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # 降温
        temperature = temperature * cooling_rate
        if temperature < min_temperature:
            break

        # 随机选择一个邻近解（即新的策略矩阵）
        new_solution = random.choice(b_matrices)
        b_matrix = new_solution
        new_cost, new_yield, new_defective = evaluate_solution2(n, p_part, reverse_time)

        # 计算成本差异
        cost_diff = new_cost - current_cost
        yield_diff = new_yield - current_yield
        defective_diff = new_defective - current_defective

        # 可以通过调整权重 α, β, γ 来控制各个目标对接受概率的影响
        alpha, beta, gamma = 0.5, 0.3, 0.2  # 权重值可以根据实际需要调整
        objective_diff = alpha * cost_diff - beta * yield_diff + gamma * defective_diff

        # 如果新解更优，接受它；如果更差，按概率接受（考虑到 objective_diff 和温度的关系）
        if objective_diff < 0 or math.exp(-objective_diff / temperature) > random.random():
            current_solution = new_solution
            current_cost = new_cost
            current_yield = new_yield
            current_defective = new_defective

        # 更新最优解
        if new_cost < best_cost and new_yield > best_yield and new_defective < best_defective:
            best_solution = new_solution
            best_cost = new_cost
            best_yield = new_yield
            best_defective = new_defective

        # 打印当前迭代的信息
        # print(
        #     f"Iteration {iteration + 1}: Temperature = {temperature:.4f}, Current Cost = {current_cost:.2f}, Best Cost = {best_cost:.2f}, Yield Rate = {best_yield:.2f}")

    return best_solution, best_cost, best_yield, best_defective


if __name__ == '__main__':
    reversed_times = [0, 1, 2]
    for reverse_time in reversed_times:
        # 生成策略矩阵
        b_matrices = gd.generate_matrix_q3_1(4, reverse_time + 1)

        # 使用退火算法进行优化
        best_matrix, best_cost, best_yield, best_defective = simulated_annealing1(b_matrices, n, p_part, reverse_time, max_iterations,
                                                                 initial_temperature, cooling_rate, min_temperature)

        # 输出最佳结果
        print(f"for situation 1, reversed for {reverse_time} iterations:")
        print(f"Best solution found:\n {best_matrix}")
        print(f"Lowest cost per intermediate_product: {best_cost/(n * best_yield):.2f}")
        print(f"Highest yield rate: {best_yield:.3f}")
        print(f"Lowest defective: {best_defective:.3f}")
        print("=====================================")

    for reverse_time in reversed_times:
        b_matrices = gd.generate_matrix_q3_1(3, reverse_time + 1)
        best_matrix, best_cost, best_yield, best_defective = simulated_annealing2(b_matrices, m, p_part, reverse_time, max_iterations,
                                                                  initial_temperature, cooling_rate, min_temperature)

        print(f"for situation 2, reversed for {reverse_time} iterations:")
        print(f"Best solution found:\n {best_matrix}")
        print(f"Lowest cost per intermediate_product: {best_cost/(m * best_yield):.2f}")
        print(f"Highest yield rate: {best_yield:.3f}")
        print(f"Lowest defective: {best_defective:.3f}")
        print("=====================================")
