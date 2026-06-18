import csv
import matplotlib.pyplot as plt

# 读取 CSV 数据
iters = []
xs = []
ys = []
costs = []

with open('gradient_data.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # 跳过表头
    for row in reader:
        iters.append(int(row[0]))
        xs.append(float(row[1]))
        ys.append(float(row[2]))
        costs.append(float(row[3]))

# ====== 图1：迭代轨迹 ======
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(xs, ys, 'ro-', markersize=3, linewidth=1, label='path')
plt.plot(xs[0], ys[0], 'go', markersize=8, label='starting')
plt.plot(xs[-1], ys[-1], 'bo', markersize=8, label='ending')
plt.plot(3, 3, 'r*', markersize=12, label='sign(3,3)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('gradient path(η=0.1)')
plt.legend()
plt.grid(True)
plt.axis('equal')

# ====== 图2：代价函数收敛曲线 ======
plt.subplot(1, 2, 2)
plt.semilogy(iters, costs, 'b-', linewidth=2)
plt.xlabel('iter')
plt.ylabel('value')
plt.title('Convergence Curve (log site)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('gradient_descent_q1.png', dpi=150)
plt.show()

print("图片已保存为 gradient_descent_q1.png")#不想搞得太复杂，于是我每个学习率对应的图片就重命名了
