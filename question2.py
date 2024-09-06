import numpy as np
import generate_data_q2 as gd2

# 变量
p1 = 0.1 # 部件一的次品率
p2 = 0.1 # 部件二的次品率
p3 = 0.1 # 成品在部件一和部件二都是正品时的次品率
n = m = 100
price_1 = 4 # 部件一的成本
price_2 = 18 # 部件二的成本
check_1 = 2 # 部件一的检测成本
check_2 = 3 # 部件二的检测成本
check_3 = 3 # 成品的检测成本
price_assemble = 6 # 装配成本
purchase = 56 # 成品售价
punish = 6 # 惩罚
dismantle = 5 # 拆解成本
W = 0    # 总收益
COST = price_1 * n + price_2 * m # 总成本
# b1 部件一不被检测的flag值, 0表示被检查, 1表示不检查, 且列表中如果出现过0, 使得利润最大化的操作则是将0之后的数据改为1. b2以及b3同.
# b2 部件二~~
# b3 是否对成品进行检查
# b4 是否对不合格成品进行拆解
"""
    即: 要么在一开始就检查, 然后全取1, 要么先不检查, 如果回炉则必须检查, 然后后期全取0
    [0,1,1,1...]
    [1,0,1,1,1...]
"""

b_matrices = None
b_matrix = None

def func(p1, p2, p3, n, m, b1, b2, b3, b4):
    global COST, W, check_1, check_2, check_3, price_assemble, purchase, punish, b_matrix

    if b1 == 0:
        # 对部件一进行检测, 数量减少, 同时有检测成本
        COST += check_1 * n
        n = n * (1.0 - p1)
    if b2 == 0:
        # 对部件二进行检测,
        COST += check_2 * m
        m = m * (1.0 - p2)

    # 获取最大成品数
    _p1 = p1 * b1
    _p2 = p2 * b2
    c3 = min(n, m)

    # 获取成品装配后次品率
    _p3 = _p1 + (1 - _p1) * _p2 + (1 - _p2) * (1 - _p1) * p3

    # 获取装配所需成本
    COST += c3 * price_assemble

    # 获取合格产品数
    qualified_product_count = c3 * (1.0 - _p3)
    if b3 == 1: # 对成品进行检查
        # 合格产品直接收益
        COST += check_3 * c3
        W += qualified_product_count * purchase
    else:
        # 不检查可以从所有产品处获得收益, 但是有调换损失
        W += qualified_product_count * purchase
        COST += c3 * _p3 * punish

    # 不合格产品处理
    if b4 >= 1:
        unqualified_product_count = c3 - qualified_product_count
        # 注意回炉重造时的次品率变化
        if _p1 != 0.0:
            _p1 = c3 * p1 / unqualified_product_count
        if _p2 != 0.0:
            _p2 = c3 * p2 / unqualified_product_count

        # 拆解成本
        COST += unqualified_product_count * dismantle

        # 递归模拟回炉
        func(_p1, _p2, p3, unqualified_product_count, unqualified_product_count, b_matrix[0, -b4], b_matrix[1, -b4], b_matrix[2, -b4], b4 - 1)
    else:
        return
        
if __name__ == '__main__':
    # 情况1
    b4 = 1

    b_matrices = gd2.generate_matrix_q2(b4 + 1)
    res = np.array([])
    map = {}
    for matrix in b_matrices:
        b_matrix = matrix
        func(p1, p2, p3, n, m, b_matrix[0, 0], b_matrix[1, 0], b_matrix[2, 0], b4)
        single_profit = round((W - COST) / n, 3)
        res = np.append(res, single_profit)
        map[str(single_profit)] = b_matrix
        W = 0
        COST = price_1 * n + price_2 * m
    sort = np.sort(res)
    print(sort)
    print(map[str(sort[-1])])
