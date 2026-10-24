""" 
GA太不稳定了，在0.6~0.96范围内效果不如optimize1_3.py的实现
不过optimize1_3.py不收敛的地方倒是可以接受
"""
'''
这段代码实现了使用遗传算法优化双重抽样方案参数的功能。主要内容包括：
设定真实次品率p_0为固定值(0.15)作为示例
定义了计算第一类错误概率和第二类错误概率的函数
计算期望抽样次数的函数
设计了目标函数，将期望抽样次数和约束条件(信度95%和目标效度)结合起来
使用遗传算法(GA)进行优化
输出优化结果和性能指标
绘制不同次品率下的效度曲线以及遗传算法的进化过程
代码中的数学模型完全基于所提供的公式，使用正态分布近似二项分布来计算概率。优化结果给出了一组参数(n1, c1, r1, n2, r2)，这组参数在满足效度和信度要求的情况下，使期望抽样次数最小。
如果需要考虑不同的p_0分布情况，可以修改objective_function函数，将其改为对p_0的积分形式，这需要使用数值积分方法来实现。
'''
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, basinhopping, differential_evolution
import matplotlib.pyplot as plt
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import sys
import os
import multiprocessing as mp
# 导入效度函数
from scipy import stats, integrate
from efficacy_function import K_n, K_n_p
# 导入遗传算法库
from sko.GA import GA

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

P_0 = 0.15  # default

# 定义参数边界
def parameter_bounds(n1, c1, r1, n2, r2):
    valid = True
    valid = valid and (n1 > 0)
    valid = valid and (n2 >= n1)
    valid = valid and (0 < c1 < r1 < 1)
    valid = valid and (0 < r2 < 1)
    return valid

# 计算拒绝H0的概率 (第一类错误)。在不传入p时能够返回放缩后的极值
def probability_reject_H0(n1, c1, r1, n2, r2, p=0.1):
    # 在第一轮直接拒绝的概率
    z1 = (r1 - p) / np.sqrt(p * (1 - p) / n1)
    prob_direct_reject = 1 - norm.cdf(z1)
    
    # 在第二轮拒绝的概率
    z2 = (r2 - p) / np.sqrt(p * (1 - p) / n2)
    prob_reject_2nd_round = (norm.cdf((r1 - p) / np.sqrt(p * (1 - p) / n1)) - 
                            norm.cdf((c1 - p) / np.sqrt(p * (1 - p) / n1))) * (1 - norm.cdf(z2))
    
    return prob_direct_reject + prob_reject_2nd_round

# 计算接受H1的概率 (第二类错误)
def probability_accept_H1(n1, c1, r1, n2, r2, p=P_0):
    # 在第一轮直接接受的概率
    z1 = (c1 - p) / np.sqrt(p * (1 - p) / n1)
    prob_direct_accept = norm.cdf(z1)
    
    # 在第二轮接受的概率
    z2 = (r2 - p) / np.sqrt(p * (1 - p) / n2)
    prob_accept_2nd_round = (norm.cdf((r1 - p) / np.sqrt(p * (1 - p) / n1)) - 
                           norm.cdf((c1 - p) / np.sqrt(p * (1 - p) / n1))) * norm.cdf(z2)
    
    return prob_direct_accept + prob_accept_2nd_round

def calc_K_n(n1, c1, r1, n2, r2):
    # 假设f(p0)是均匀分布，即所有p0的概率密度相同
    def integrand(p):
        return 1 - probability_accept_H1(n1, c1, r1, n2, r2, p)
    
    # 使用数值积分计算从0.10到1的积分
    # result, _ = integrate.quad(integrand, 0.15, 1.0)
    result = 0
    for p in [0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]:
        result += integrand(p) / 9
    # 计算当前方案的效度
    current_K = result
    # current_K = 1 - probability_accept_H1(n1, c1, r1, n2, r2)
    return current_K

# 计算期望抽样次数
def expected_sample_size_single(n1, c1, r1, n2, r2, p):
    # 直接接受或拒绝的概率
    prob_direct_decision = norm.cdf((c1 - p) / np.sqrt(p * (1 - p) / n1)) + (1 - norm.cdf((r1 - p) / np.sqrt(p * (1 - p) / n1)))
    
    # 需要第二轮抽样的概率
    prob_second_round = 1 - prob_direct_decision
    
    # 期望抽样次数
    E_n = n1 + n2 * prob_second_round
    
    return E_n

def expected_sample_size(n1, c1, r1, n2, r2,p=-1):
    if p == -1:
        p = np.linspace(0.15, 0.9, 1000)
        E_n = expected_sample_size_single(n1, c1, r1, n2, r2, p)
        return E_n.mean()
    else:
        return expected_sample_size_single(n1, c1, r1, n2, r2, p)

def expected_sample_size_by_params(params):
    n1, c1, r1, n2, r2 = params
    return expected_sample_size(n1, c1, r1, n2, r2, p_0)

# 定义目标函数（要最小化的函数）
def objective_function(params, target_K=0.95):
    n1, c1, r1, n2, r2 = params
    
    # 如果参数不在合理范围内，返回一个很大的值
    if not parameter_bounds(n1, c1, r1, n2, r2):
        return 1e14
    
    # 计算第一类错误概率
    type_I_error = probability_reject_H0(n1, c1, r1, n2, r2)
    
    current_K = calc_K_n(n1,c1,r1,n2,r2)
    
    # 添加约束条件，确保第一类错误概率≤0.05且效度接近目标值
    constraint_penalty = 0
    if type_I_error > 0.05:
        constraint_penalty += 1e8 * (type_I_error - 0.05)**2
    
    constraint_penalty += 1e8 * (current_K - target_K)**2
    
    # 返回期望抽样次数加上约束惩罚
    return expected_sample_size(n1, c1, r1, n2, r2) + constraint_penalty

def CalcMinNByTargetK(target_K=0.95, output=False):
    """
    使用遗传算法找到满足目标效度要求的最优双重抽样参数
    
    参数:
    target_K: 目标效度值
    output: 是否输出详细信息
    
    返回:
    最优参数组合 [n1, c1, r1, n2, r2]
    """
    # 定义适应度函数（就是目标函数）
    def fitness_func(p):
        return objective_function(p, target_K)
    
    # 定义参数边界
    lb = [10, 0.01, 0.1, 10, 0.1]     # 下界
    ub = [1000, 0.3, 0.5, 5000, 0.5]    # 上界
    
    # 初始化遗传算法
    ga = GA(
        func=fitness_func,         # 适应度函数
        n_dim=5,                   # 维度：n1, c1, r1, n2, r2
        size_pop=100,               # 种群大小
        max_iter=100,              # 最大迭代次数
        prob_mut=0.01,             # 变异概率
        lb=lb,                     # 参数下界
        ub=ub,                     # 参数上界
        precision=[1, 1e-4, 1e-4, 1, 1e-4]  # 参数精度
    )
    
    # 运行遗传算法
    best_params, best_fitness = ga.run()
    
    # 如果 best_fitness 是数组，取第一个元素
    if isinstance(best_fitness, np.ndarray):
        best_fitness = best_fitness[0]

    # 输出结果
    if output:
        print("\n优化结果:")
        print(f"n1 = {best_params[0]:.2f}")
        print(f"c1 = {best_params[1]:.4f}")
        print(f"r1 = {best_params[2]:.4f}")
        print(f"n2 = {best_params[3]:.2f}")
        print(f"r2 = {best_params[4]:.4f}")
        print(f"适应度函数值 = {best_fitness:.4f}")
        
    # 计算最优方案的性能指标
    n1, c1, r1, n2, r2 = best_params
    type_I_error = probability_reject_H0(n1, c1, r1, n2, r2)
    effectiveness = calc_K_n(n1, c1, r1, n2, r2)
    exp_sample_size = expected_sample_size(n1, c1, r1, n2, r2)

    if output:
        print("\n性能指标:")
        print(f"第一类错误概率 = {type_I_error:.4f}")
        print(f"效度 = {effectiveness:.4f}")
        print(f"期望抽样次数 = {exp_sample_size:.2f}")
        
    if output:
        # 可视化进化过程
        import pandas as pd
        import matplotlib.pyplot as plt
        
        Y_history = pd.DataFrame(ga.all_history_Y)
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        ax[0].set_title('每代种群的适应度值分布')
        
        best_Y = Y_history.min(axis=1)
        ax[1].plot(best_Y.index, best_Y, label='每代最优适应度')
        ax[1].plot(best_Y.index, best_Y.cummin(), label='全局最优适应度')
        ax[1].set_title('适应度进化曲线')
        ax[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'GA_evolution_K{target_K}.png')
        print(f"进化过程已保存到GA_evolution_K{target_K}.png")
    
    return best_params

# CalcMinNByTargetK(0.88,0.1,output=True)
# # for target_K in [0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50]:
# #     res = CalcMinNByTargetK(target_K,output=True)
# #     print(f"目标效度 = {target_K:.2f}")
# #     print(f"最小样本量 = {res[0]:.2f}")

# 将worker函数移到全局作用域
def optimization_worker(task):
    k_target = task
    result = CalcMinNByTargetK(k_target, output=False)
    
    # 计算性能指标 - 结果现在直接是参数数组
    n1, c1, r1, n2, r2 = result
    type_I_error = probability_reject_H0(n1, c1, r1, n2, r2)
    effectiveness = calc_K_n(n1, c1, r1, n2, r2)
    exp_sample_size = expected_sample_size(n1, c1, r1, n2, r2)
    
    return {
        'K_Target': k_target,
        'p_0': -1,
        'n1': n1,
        'c1': c1,
        'r1': r1,
        'n2': n2,
        'r2': r2,
        '第一类错误概率': type_I_error,
        '效度': effectiveness,
        '期望抽样次数': exp_sample_size
    }

# 修改多线程优化函数
def multi_thread_optimization(max_thread = 16):
    # 定义K_Target的范围
    k_targets = np.arange(0.6, 0.99, 0.01)
    
    # 创建任务列表
    tasks = []
    for k in k_targets:
        tasks.append(k)
    
    print(f"总共需要计算 {len(tasks)} 个K_Target")
    
    # 存储结果的数据框
    results = []
    
    # 获取CPU核心数量，但最多使用16个
    num_processes = min(max_thread, mp.cpu_count(), len(tasks))
    print(f"使用 {num_processes} 个进程进行并行计算")
    
    # 创建进程池并执行任务
    if __name__ == "__main__":  # 确保多进程代码在主模块中运行
        with mp.Pool(processes=num_processes) as pool:
            # 使用imap处理任务并结合tqdm显示进度
            for result in tqdm(pool.imap(optimization_worker, tasks), 
                              total=len(tasks), desc="优化计算进度"):
                results.append(result)
    else:
        # 如果不是在主模块运行，则使用单进程执行
        for task in tqdm(tasks, desc="优化计算进度（单进程）"):
            results.append(optimization_worker(task))
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存结果到CSV文件
    results_df.to_csv('optimization_results.csv', index=False)
    
    # 打印部分结果
    print("\n优化结果摘要:")
    print(results_df.sort_values(by=['K_Target']).head(10))
    
    return results_df

# 修改独立绘图函数
def plot_optimization_results(data_source=None):
    """
    绘制优化结果的图像
    
    参数:
    data_source: 可以是DataFrame或CSV文件路径字符串
    """
    # 处理数据源
    if data_source is None:
        # 默认尝试读取CSV文件
        csv_file = 'optimization_results.csv'
        if not os.path.exists(csv_file):
            print(f"错误：找不到文件 {csv_file}")
            return
        print(f"正在读取文件：{csv_file}")
        results_df = pd.read_csv(csv_file)
    elif isinstance(data_source, pd.DataFrame):
        # 直接使用传入的DataFrame
        results_df = data_source
        print("使用传入的DataFrame数据")
    elif isinstance(data_source, str):
        # 从指定路径读取CSV文件
        if not os.path.exists(data_source):
            print(f"错误：找不到文件 {data_source}")
            return
        print(f"正在读取文件：{data_source}")
        results_df = pd.read_csv(data_source)
    else:
        print("错误：无效的数据源类型，应为DataFrame或文件路径字符串")
        return
    
    print(f"数据包含 {len(results_df)} 条记录")
    print("数据摘要:")
    print(results_df.describe())
    
    # 1. 创建n1与K_Target和p_0的关系热图
    print("绘制第一阶段样本量(n1)热图...")
    plt.figure(figsize=(12, 8))
    pivot_df = results_df.pivot_table(values='n1', index='p_0', columns='K_Target')
    plt.imshow(pivot_df, cmap='hot', interpolation='nearest')
    plt.colorbar(label='n1 (第一阶段样本量)')
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.xticks(range(len(pivot_df.columns)), [f"{x:.2f}" for x in pivot_df.columns])
    plt.xlabel('目标效度 (K_Target)')
    plt.ylabel('次品率 (p_0)')
    plt.title('不同目标效度和次品率下的最优第一阶段样本量(n1)')
    plt.tight_layout()
    plt.savefig(f'optimization_heatmap_n1.png')
    
    # 2. 创建期望抽样次数的热图
    print("绘制期望抽样次数热图...")
    plt.figure(figsize=(12, 8))
    pivot_df = results_df.pivot_table(values='期望抽样次数', index='p_0', columns='K_Target')
    plt.imshow(pivot_df, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='期望抽样次数')
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.xticks(range(len(pivot_df.columns)), [f"{x:.2f}" for x in pivot_df.columns])
    plt.xlabel('目标效度 (K_Target)')
    plt.ylabel('次品率 (p_0)')
    plt.title('不同目标效度和次品率下的期望抽样次数')
    plt.tight_layout()
    plt.savefig(f'optimization_heatmap_en.png')
    
    # 3. 创建期望抽样次数与目标效度的关系曲线图
    print("绘制期望抽样次数与目标效度关系曲线（en_vs_ktarget）...")
    plt.figure(figsize=(15, 10))
    
    # 挑选几个不同的p0值绘制曲线
    p0_values = sorted(results_df['p_0'].unique())[::]
    # p0_values = sorted(results_df['p_0'].unique())[::1]  # 每隔一个取一个值
    
    for p in p0_values:
        subset = results_df[results_df['p_0'] == p]
        subset = subset.sort_values('K_Target')
        plt.plot(subset['K_Target'], subset['期望抽样次数'], 'o-', label=f'双重抽样')
    
    # 添加单次抽样的K(n)函数曲线作为对比
    # 计算一系列样本量对应的K(n)值
    n_values = np.arange(10, 1000, 5)
    k_values = [K_n(n) for n in n_values]
    
    # 添加单次抽样曲线（虚线）
    plt.plot(k_values, n_values, 'r--', linewidth=2, label='单次抽样')
    
    plt.xlabel('效度')
    plt.ylabel('期望抽样次数')
    plt.title('不同目标效度下期望抽样次数与效度的关系')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'en_vs_ktarget.png')
    
    
    # 4. 创建效度与实际次品率的关系曲线图
    print("绘制效度与实际次品率关系曲线...")
    plt.figure(figsize=(15, 10))
    
    # 挑选几个不同的K_Target值绘制曲线
    target_k_values = sorted(results_df['K_Target'].unique())[::5]  # 每隔5个取一个值
    
    # 创建更细致的p_0范围用于绘制平滑曲线
    smooth_p0_values = np.linspace(0.15, 0.9, 100)
    
    for k in target_k_values:
        subset = results_df[results_df['K_Target'] == k]
        # 获取该K_Target对应的最优参数
        if len(subset) > 0:
            # 取第一个匹配的参数组合
            params = subset.iloc[0][['n1', 'c1', 'r1', 'n2', 'r2']].values
            n1, c1, r1, n2, r2 = params
            
            # 计算不同p_0下的效度
            effectiveness_values = []
            for p in smooth_p0_values:
                eff = 1 - probability_accept_H1(n1, c1, r1, n2, r2, p)
                effectiveness_values.append(eff)
            
            plt.plot(smooth_p0_values, effectiveness_values, '-', label=f'K_Target={k:.2f}')
            
            # 也可以标记原始数据点
            original_p0s = subset['p_0'].values
            original_effs = [1 - probability_accept_H1(n1, c1, r1, n2, r2, p) for p in original_p0s]
            plt.scatter(original_p0s, original_effs, marker='o')

    plt.xlabel('次品率 (p_0)')
    plt.ylabel('实际效度')
    plt.title('不同目标效度下实际效度与次品率的关系')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'effectiveness_vs_p0.png')
    
    # 5. 创建样本量与效度的关系曲线图
    print("绘制样本量与效度关系曲线...")
    plt.figure(figsize=(15, 10))
    
    # 挑选几个不同的p_0值绘制曲线
    p0_values = sorted(results_df['p_0'].unique())[::2]  # 每隔一个取一个值
    
    for p in p0_values:
        subset = results_df[results_df['p_0'] == p]
        subset = subset.sort_values('K_Target')
        plt.plot(subset['K_Target'], subset['n1'], 'o-', label=f'p_0={p:.2f}')
    
    plt.xlabel('目标效度 (K_Target)')
    plt.ylabel('第一阶段样本量 (n1)')
    plt.title('不同次品率下第一阶段样本量与目标效度的关系')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'n1_vs_ktarget.png')
    
    # 6. 创建期望抽样次数与实际效度的关系曲线图
    print("绘制期望抽样次数与实际效度关系曲线（en_vs_effectiveness）...")
    plt.figure(figsize=(15, 10))
    
    # 挑选几个不同的p0值绘制曲线
    p0_values = sorted(results_df['p_0'].unique())[::]
    # p0_values = sorted(results_df['p_0'].unique())[::1]  # 每隔一个取一个值
    
    for p in p0_values:
        subset = results_df[results_df['p_0'] == p]
        subset = subset.sort_values('K_Target')
        plt.plot(subset['效度'], subset['期望抽样次数'], 'o-', label=f'双重抽样')
    
    # 添加单次抽样的K(n)函数曲线作为对比
    # 计算一系列样本量对应的K(n)值
    n_values = np.arange(10, 1000, 5)
    k_values = [K_n(n) for n in n_values]
    
    # 添加单次抽样曲线（虚线）
    plt.plot(k_values, n_values, 'r--', linewidth=2, label='单次抽样')
    
    plt.xlabel('效度')
    plt.ylabel('期望抽样次数')
    plt.title('不同目标效度下期望抽样次数与效度的关系')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'en_vs_k.png')
    
    print(f"所有图像已保存")
    
    return results_df


# 主函数
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == "full":
        print("开始多线程优化计算...")
        results = multi_thread_optimization(max_thread = int(args[1]) if len(args) >= 2 else 16)
        print("计算完成，开始绘制图像...")
        plot_optimization_results(results)
    elif len(args) == 1 and args[0] == "draw":
        # command: draw
        # 读取csv，重新绘制图像
        print(f"将使用文件 optimization_results.csv 重新绘制图像")
        plot_optimization_results("optimization_results.csv")
    elif len(args) == 1 and args[0] == "test":
        # 测试命令
        for target_K in [0.99,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50]:
            res= CalcMinNByTargetK(target_K,0.1)
            print(f"目标效度 = {target_K:.2f}")
            print(f"期望抽样次数 = {expected_sample_size_by_params(res):.2f}")
    else:
        # 测试命令
        res = CalcMinNByTargetK(1,output=True)


# 注释掉不需要的代码，保留以备将来使用
'''
# 绘制不同p_0下的效度曲线
import matplotlib.pyplot as plt

p_values = np.linspace(0.1, 0.3, 50)
effectiveness_values = []

for p in p_values:
    eff = 1 - probability_accept_H1(optimal_n1, optimal_c1, optimal_r1, optimal_n2, optimal_r2, p)
    effectiveness_values.append(eff)

plt.figure(figsize=(10, 6))
plt.plot(p_values, effectiveness_values, 'b-')
plt.axhline(y=0.95, color='r', linestyle='--', label='目标效度(0.95)')
plt.axvline(x=0.1, color='g', linestyle='--', label='标称次品率(0.1)')
plt.xlabel('实际次品率(p_0)')
plt.ylabel('效度K(n,p_0)')
plt.title('双重抽样方案效度曲线')
plt.grid(True)
plt.legend()
plt.savefig('double_sampling_effectiveness.png')
plt.show()
'''
