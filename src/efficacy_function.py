import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 标称次品率
p0 = 0.10

def K_n_p(n, p):
    """
    计算效度函数K(n,p)
    n: 样本量
    p: 实际次品率
    """
    # 使用正态分布近似计算
    # 对于二项分布B(n,p)，当n较大时，可近似为正态分布N(np, np(1-p))
    
    # 计算t分布的临界值（95%置信度）
    t_critical = stats.t.ppf(0.95, n-1)
    
    # 计算标准差（对于二项分布，标准差为sqrt(np(1-p)))
    std_dev = np.sqrt(p * (1 - p) / n)
    
    # 计算拒绝域的临界值
    critical_value = p0 + t_critical * std_dev
    
    # 如果临界值超过1，则概率为0
    if critical_value >= 1:
        return 0
    
    # 计算X_bar > critical_value的概率，这是犯第二类错误的概率
    # 使用正态分布近似
    z_score = (critical_value - p) / std_dev
    probability = 1 - stats.norm.cdf(z_score)
    
    return probability

def K_n(n):
    """
    计算平均效度K(n)
    n: 样本量
    """
    # 假设p(p0)是均匀分布，即所有p0的概率密度相同
    def integrand(p):
        return K_n_p(n, p)
    
    # 使用数值积分计算从0.10到1的积分
    result, _ = integrate.quad(integrand, 0.10, 1.0)
    
    # 由于积分区间长度为0.9，且假设均匀分布，需要除以0.9
    return result / 0.9

if __name__ == "__main__":
    # 计算不同样本量下的效度函数值
    n_values = np.arange(10, 101, 10)  # 样本量从10到100，步长为10
    p_values = np.linspace(0.1, 0.9, 9)  # 实际次品率从0.1到0.9

    # 计算K(n,p)
    K_matrix = np.zeros((len(n_values), len(p_values)))
    for i, n in enumerate(n_values):
        for j, p in enumerate(p_values):
            K_matrix[i, j] = K_n_p(n, p)

    # 计算K(n)
    K_n_values = [K_n(n) for n in n_values]

    # 可视化K(n,p)
    plt.figure(figsize=(10, 6))
    for j, p in enumerate(p_values):
        plt.plot(n_values, K_matrix[:, j], marker='o', label=f'p={p:.2f}')

    plt.xlabel('样本量 (n)')
    plt.ylabel('效度函数 K(n,p)')
    plt.title('不同实际次品率下的效度函数随样本量的变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('K_n_p.png')

    # 可视化K(n)
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, K_n_values, marker='o', color='red')
    plt.xlabel('样本量 (n)')
    plt.ylabel('平均效度 K(n)')
    plt.title('平均效度随样本量的变化')
    plt.grid(True)
    plt.savefig('K_n.png')

    # 输出结果
    print("效度函数K(n,p)的值：")
    for i, n in enumerate(n_values):
        print(f"n = {n}:")
        for j, p in enumerate(p_values):
            print(f"  p = {p:.2f}: K(n,p) = {K_matrix[i, j]:.4f}")

    print("\n平均效度K(n)的值：")
    for i, n in enumerate(n_values):
        print(f"n = {n}: K(n) = {K_n_values[i]:.4f}")
        
    # 可视化 K(n,p)，以p为横坐标，K(n,p)为纵坐标
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(n_values):
        plt.plot(p_values, K_matrix[i], marker='o', label=f'n={n}')
    plt.xlabel('实际次品率 (p)')
    plt.ylabel('效度函数 K(n,p)')
    plt.title('不同样本量下的效度函数随实际次品率的变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('K_n_p_p.png')


    plt.show() 