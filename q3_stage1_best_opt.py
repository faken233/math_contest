import matplotlib.pyplot as plt
import numpy as np

import q3_stage1_AHP as system1
import q3_stage1_TOPSIS as system2
import q3_stage1_weight_score as system3

# 配置 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示符号

b5 = [0, 1, 2]
ranks = []

for i in range(3):
    if i == 0:
        rank, num = system2.main()

        # 根据 num 进行排序
        sorted_indices = np.argsort(num)

        # 对 rank 和 num 进行排序
        sorted_rank = np.array(rank)[sorted_indices]
        sorted_num = np.array(num)[sorted_indices]

        ranks.append(sorted_rank)

        # 输出排序后的结果
        for i in range(len(sorted_rank)):
            print(f"Rank: {sorted_rank[i]}, Num: {sorted_num[i]}")

    if i == 1:
        num = system1.main()

        rank = np.zeros_like(num)

        # 遍历 sorted_indices 并为每个元素赋予对应的 rank
        for j, index in enumerate(num):
            rank[index] = j + 1

        # 根据 num 进行排序
        sorted_indices = np.argsort(num)

        # 对 rank 和 num 进行排序
        sorted_rank = np.array(rank)[sorted_indices]
        sorted_num = np.array(num)[sorted_indices]

        ranks.append(sorted_rank)

        # 输出排序后的结果
        for i in range(len(sorted_rank)):
            print(f"Rank: {sorted_rank[i]}, Num: {sorted_num[i]}")

    if i == 2:
        score = system3.main()

        rank = np.zeros(len(score))
        num = np.zeros(len(score))

        for j, (score, index) in enumerate(score):
            rank[index] = j + 1
            num[index] = index

        # 根据 num 进行排序
        sorted_indices = np.argsort(num)

        # 对 rank 和 num 进行排序
        sorted_rank = np.array(rank)[sorted_indices]
        sorted_num = np.array(num)[sorted_indices]

        ranks.append(sorted_rank)

        # 输出排序后的结果
        for i in range(len(sorted_rank)):
            print(f"Rank: {sorted_rank[i]}, Num: {sorted_num[i]}")

x = np.array(ranks[0])
y = np.array(ranks[1])
z = np.array(ranks[2])

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
scatter = ax.scatter(x, y, z, edgecolors='blue', marker='o')

# 添加颜色条
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

# 设置坐标轴标签
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# 调整观察角度和方位角
ax.view_init(30, 35)

plt.show()


# 最短距离
distances = []

for i in range(len(ranks[0])):
    distances.append(np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2))
sorted_indices = np.argsort(distances)
min_index = sorted_indices[0]  # 最短距离的索引
min_distance = distances[min_index]  # 最短距离

print(f"最短距离: {min_distance}")
print(f"最短距离对应的索引: {min_index}")
