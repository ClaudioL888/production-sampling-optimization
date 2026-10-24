import torch
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置 GPU 设备（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 全局参数

# 零配件1参数
C1b = torch.tensor(4.0, device=device)    # 购买单价
C1d = torch.tensor(2.0, device=device)    # 检测成本
P1_mean = 0.10                            # 次品率均值

# 零配件2参数
C2b = torch.tensor(18.0, device=device)   # 购买单价
C2d = torch.tensor(3.0, device=device)    # 检测成本
P2_mean = 0.10                            # 次品率均值

# 成品参数
Pf = 0.10              # 成品次品率（基础，不考虑其他因素）
Cf_a = torch.tensor(6.0, device=device)   # 装配成本
Cf_d = torch.tensor(3.0, device=device)   # 成品检测成本
S = torch.tensor(56.0, device=device)       # 市场售价
L = torch.tensor(6.0, device=device)        # 替换损失（退换成本）

# 拆解参数
Cfr = torch.tensor(5.0, device=device)    # 拆解费用
# 回收率参数
R1_full = 0.8    # 拆解后零配件1检测时回收率
R1_half = 0.4    # 拆解后零配件1不检测时部分回收
R2_full = 0.8
R2_half = 0.4

# 蒙特卡罗模拟参数
num_samples = 10000  
P_std = 0.02         # 次品率采样标准差

# 随机种子设置
torch.manual_seed(42)
random.seed(42)

# 参数范围定义（用于敏感性分析）

P_std_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
pop_size_values = [20, 50, 100, 200]
num_generations_values = [10, 20, 30, 50]
mutation_rate_values = [0.05, 0.1, 0.15, 0.2]
num_parents_values = [10, 20, 30, 40]


scenarios = [
    # 情境1
    {
      "情境": 1,
      "零配件1_次品率": 0.10, "零配件1_购买单价": 4.0, "零配件1_检测成本": 2.0,
      "零配件2_次品率": 0.10, "零配件2_购买单价": 18.0, "零配件2_检测成本": 3.0,
      "成品_次品率": 0.10, "成品_装配成本": 6.0, "成品_检测成本": 3.0, "成品_市场售价": 56.0,
      "不合格品_调换损失": 6.0, "不合格品_拆解费用": 5.0
    },
    # 情境2
    {
      "情境": 2,
      "零配件1_次品率": 0.20, "零配件1_购买单价": 4.0, "零配件1_检测成本": 2.0,
      "零配件2_次品率": 0.20, "零配件2_购买单价": 18.0, "零配件2_检测成本": 3.0,
      "成品_次品率": 0.20, "成品_装配成本": 6.0, "成品_检测成本": 3.0, "成品_市场售价": 56.0,
      "不合格品_调换损失": 6.0, "不合格品_拆解费用": 5.0
    },
    # 情境3
    {
      "情境": 3,
      "零配件1_次品率": 0.10, "零配件1_购买单价": 4.0, "零配件1_检测成本": 2.0,
      "零配件2_次品率": 0.10, "零配件2_购买单价": 18.0, "零配件2_检测成本": 3.0,
      "成品_次品率": 0.10, "成品_装配成本": 6.0, "成品_检测成本": 3.0, "成品_市场售价": 56.0,
      "不合格品_调换损失": 30.0, "不合格品_拆解费用": 5.0
    },
    # 情境4
    {
      "情境": 4,
      "零配件1_次品率": 0.20, "零配件1_购买单价": 4.0, "零配件1_检测成本": 1.0,
      "零配件2_次品率": 0.20, "零配件2_购买单价": 18.0, "零配件2_检测成本": 1.0,
      "成品_次品率": 0.20, "成品_装配成本": 6.0, "成品_检测成本": 2.0, "成品_市场售价": 56.0,
      "不合格品_调换损失": 30.0, "不合格品_拆解费用": 5.0
    },
    # 情境5
    {
      "情境": 5,
      "零配件1_次品率": 0.10, "零配件1_购买单价": 4.0, "零配件1_检测成本": 8.0,
      "零配件2_次品率": 0.20, "零配件2_购买单价": 18.0, "零配件2_检测成本": 1.0,
      "成品_次品率": 0.10, "成品_装配成本": 6.0, "成品_检测成本": 2.0, "成品_市场售价": 56.0,
      "不合格品_调换损失": 10.0, "不合格品_拆解费用": 5.0
    },
    # 情境6
    {
      "情境": 6,
      "零配件1_次品率": 0.05, "零配件1_购买单价": 4.0, "零配件1_检测成本": 2.0,
      "零配件2_次品率": 0.05, "零配件2_购买单价": 18.0, "零配件2_检测成本": 3.0,
      "成品_次品率": 0.05, "成品_装配成本": 6.0, "成品_检测成本": 3.0, "成品_市场售价": 56.0,
      "不合格品_调换损失": 10.0, "不合格品_拆解费用": 40.0
    }
]


# 根据情境更新全局参数

def set_scenario_params(scenario):
    global C1b, C1d, P1_mean, C2b, C2d, P2_mean
    global Pf, Cf_a, Cf_d, S, L, Cfr
    C1b = torch.tensor(scenario["零配件1_购买单价"], device=device)
    C1d = torch.tensor(scenario["零配件1_检测成本"], device=device)
    P1_mean = scenario["零配件1_次品率"]
    
    C2b = torch.tensor(scenario["零配件2_购买单价"], device=device)
    C2d = torch.tensor(scenario["零配件2_检测成本"], device=device)
    P2_mean = scenario["零配件2_次品率"]
    
    Pf = scenario["成品_次品率"]
    Cf_a = torch.tensor(scenario["成品_装配成本"], device=device)
    Cf_d = torch.tensor(scenario["成品_检测成本"], device=device)
    S = torch.tensor(scenario["成品_市场售价"], device=device)
    L = torch.tensor(scenario["不合格品_调换损失"], device=device)
    
    Cfr = torch.tensor(scenario["不合格品_拆解费用"], device=device)

# 模拟单轮生产过程（含反馈一次）的函数
# 返回本轮平均利润及回收的零配件数量（count，不是货币价值）

def simulate_round(policy, scale=1.0, num_samples=num_samples):
    """
    policy: tensor shape (7,) 二值决策向量，对应：
      x1: 零配件1是否检测
      x2: 零配件2是否检测
      x3: 成品是否检测
      x4: 生产过程中不合格成品是否拆解
      x5: 拆解后零配件1是否检测
      x6: 拆解后零配件2是否检测
      x7: 用户退回的成品是否拆解
    scale: 本轮生产计划成品数（受库存限制）
    返回：(avg_profit, recovered_count1, recovered_count2)
    """
    cand = policy.float()
    
    P1_samples = torch.clamp(torch.normal(mean=P1_mean, std=P_std, size=(num_samples,), device=device), 0.0, 1.0)
    P2_samples = torch.clamp(torch.normal(mean=P2_mean, std=P_std, size=(num_samples,), device=device), 0.0, 1.0)
    
    effective_P1 = torch.where(cand[0] == 1, torch.zeros_like(P1_samples), P1_samples)
    effective_P2 = torch.where(cand[1] == 1, torch.zeros_like(P2_samples), P2_samples)
    
    good_prob = (1 - effective_P1) * (1 - effective_P2) * (1 - Pf)
    rand_vals = torch.rand(num_samples, device=device)
    product_good = (rand_vals < good_prob).float()
    
    if cand[2] == 1:
        revenue = scale * S * product_good
        prod_inspection_cost = scale * Cf_d
        replacement_loss = torch.zeros_like(revenue)
        disassemble_flag = cand[3]  # 生产过程中拆解策略
    else:
        revenue = scale * S * torch.ones_like(product_good)
        prod_inspection_cost = 0.0
        replacement_loss = scale * L * (1 - product_good)
        disassemble_flag = cand[6]  # 用户退回拆解策略
       
    cost_components = scale * (C1b + C2b)
    comp_inspection_cost = scale * (cand[0] * C1d + cand[1] * C2d)
    assembly_cost = scale * Cf_a
    
    disassembly_cost = disassemble_flag * scale * Cfr
    # 回收的零配件数量（不乘单价）
    recovered_count1 = disassemble_flag * scale * (cand[4] * R1_full + (1 - cand[4]) * R1_half)
    recovered_count2 = disassemble_flag * scale * (cand[5] * R2_full + (1 - cand[5]) * R2_half)
    
    recovered_value = disassemble_flag * scale * (
         (cand[4] * R1_full + (1 - cand[4]) * R1_half) * C1b +
         (cand[5] * R2_full + (1 - cand[5]) * R2_half) * C2b
    )
    
    total_cost = scale * (C1b + C2b) + comp_inspection_cost + assembly_cost + prod_inspection_cost + disassembly_cost
    total_loss = replacement_loss
    total_revenue = revenue + recovered_value
    profit_samples = total_revenue - total_cost - total_loss
    avg_profit = profit_samples.mean()
    
    return avg_profit, recovered_count1, recovered_count2


# 反馈循环模拟：考虑初始库存，每轮生产规模受库存限制，更新库存

def simulate_feedback(policy, num_samples=num_samples, tol=1e-3, max_iter=10, inv1_init=10000, inv2_init=10000):
    total_profit = 0.0
    inv1 = inv1_init
    inv2 = inv2_init
    iter_count = 0
    desired_scale = 1000.0  # 每轮计划生产的成品数量
    while iter_count < max_iter and inv1 > 0 and inv2 > 0:
        scale = min(desired_scale, inv1, inv2)
        profit, rec1, rec2 = simulate_round(policy, scale=scale, num_samples=num_samples)
        total_profit += profit.item()
        # 扣除本轮生产使用的零件（scale个）并加上回收数量
        inv1 = inv1 - scale + rec1.item()
        inv2 = inv2 - scale + rec2.item()
        print(f"迭代 {iter_count+1}: scale = {scale:.2f}, 库存更新: 零件1 = {inv1:.2f}, 零件2 = {inv2:.2f}, 本轮利润 = {profit.item():.2f}")
        if scale < tol:
            break
        iter_count += 1
    return total_profit


# 遗传算法：在7维二值空间中寻找累计总利润最大的策略

def initialize_population(pop_size, dim=7):
    return torch.randint(0, 2, (pop_size, dim), device=device)

def fitness_function(policy):
    return simulate_feedback(policy, num_samples=num_samples)

def selection(population, fitnesses, num_parents):
    sorted_indices = torch.argsort(fitnesses, descending=True)
    return population[sorted_indices[:num_parents]]

def crossover(parent1, parent2):
    dim = parent1.shape[0]
    point = random.randint(1, dim-1)
    child = torch.cat((parent1[:point], parent2[point:]), dim=0)
    return child

def mutate(child, mutation_rate=0.1):
    for i in range(child.shape[0]):
        if random.random() < mutation_rate:
            child[i] = 1 - child[i]
    return child

def genetic_algorithm(pop_size=50, num_generations=30, num_parents=20, mutation_rate=0.1):
    population = initialize_population(pop_size)
    best_policy = None
    best_fitness = -float('inf')
    
    for generation in range(num_generations):
        fitnesses = torch.zeros(pop_size, device=device)
        for i in range(pop_size):
            fitnesses[i] = fitness_function(population[i])
        
        current_best_fitness, best_idx = torch.max(fitnesses, dim=0)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness.item()
            best_policy = population[best_idx].clone()
        
        print(f"Generation {generation+1}: Best fitness = {current_best_fitness.item():.4f}")
        
        parents = selection(population, fitnesses, num_parents)
        new_population = []
        for _ in range(pop_size):
            parent1 = parents[random.randint(0, num_parents-1)]
            parent2 = parents[random.randint(0, num_parents-1)]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = torch.stack(new_population, dim=0)
    
    return best_policy, best_fitness


# 自动评估函数：对同一策略多次运行，计算平均累计利润与95%置信区间

def evaluate_strategy(policy, num_trials=20, num_samples=num_samples, inv1_init=10000, inv2_init=10000):
    profits = []
    for _ in range(num_trials):
        total_profit = simulate_feedback(policy, num_samples=num_samples, inv1_init=inv1_init, inv2_init=inv2_init)
        profits.append(total_profit)
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    ci_95 = 1.96 * std_profit / np.sqrt(num_trials)
    return mean_profit, ci_95, profits


# 敏感性分析函数：对指定参数进行敏感性分析

def sensitivity_analysis(param_name, param_values, scenario_index=0, num_trials=10):
    global P_std  # 声明全局变量
    scenario = scenarios[scenario_index]
    set_scenario_params(scenario)
    
    # 对于 P_std 分析，保存原始值
    if param_name == 'P_std':
        original_P_std = P_std

    results = []
    
    for param_value in param_values:
        profits = []
        for _ in range(num_trials):
            if param_name == 'P_std':
                P_std = param_value
                best_policy, _ = genetic_algorithm(pop_size=50, num_generations=30, num_parents=20, mutation_rate=0.1)
                P_std = original_P_std  # 恢复原始值
            elif param_name == 'pop_size':
                best_policy, _ = genetic_algorithm(pop_size=param_value, num_generations=30, num_parents=20, mutation_rate=0.1)
            elif param_name == 'num_generations':
                best_policy, _ = genetic_algorithm(pop_size=50, num_generations=param_value, num_parents=20, mutation_rate=0.1)
            elif param_name == 'mutation_rate':
                best_policy, _ = genetic_algorithm(pop_size=50, num_generations=30, num_parents=20, mutation_rate=param_value)
            elif param_name == 'num_parents':
                best_policy, _ = genetic_algorithm(pop_size=50, num_generations=30, num_parents=param_value, mutation_rate=0.1)
            else:
                raise ValueError(f"未知参数：{param_name}")
            
            total_profit = simulate_feedback(best_policy, num_samples=num_samples, inv1_init=10000, inv2_init=10000)
            profits.append(total_profit)
        
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)
        ci_95 = 1.96 * std_profit / np.sqrt(num_trials)
        
        results.append({
            'param_value': param_value,
            'mean_profit': mean_profit,
            'std_profit': std_profit,
            'ci_95': ci_95
        })
    
    results_df = pd.DataFrame(results)
    return results_df

def plot_sensitivity_analysis(results_df, param_name, scenario_index):
    plt.figure(figsize=(12, 6))
    plt.errorbar(results_df['param_value'], results_df['mean_profit'], yerr=results_df['ci_95'], fmt='o-', capsize=5)
    plt.xlabel(param_name)
    plt.ylabel('Mean Profit')
    plt.title(f'Sensitivity Analysis for {param_name} (Scenario {scenario_index+1})')
    plt.grid(True)
    plt.show()
    
    results_df.to_csv(f'sensitivity_analysis_{param_name}_scenario_{scenario_index+1}.csv', index=False)
    print(f'已保存敏感性分析结果为 sensitivity_analysis_{param_name}_scenario_{scenario_index+1}.csv')

if __name__ == "__main__":
    results = []
    # 针对所有情境的最优策略求解与评估
    for scenario in scenarios:
        print(f"\n====== 优化 情境 {scenario['情境']} ======")
        set_scenario_params(scenario)
        best_policy, best_fitness = genetic_algorithm(pop_size=50, num_generations=30, num_parents=20, mutation_rate=0.1)
        print("\n进行多次评估以验证策略稳定性...")
        mean_profit, ci_95, profits = evaluate_strategy(best_policy, num_trials=20, num_samples=num_samples,
                                                          inv1_init=10000, inv2_init=10000)
        results.append({
            "情境": scenario["情境"],
            "最优策略": best_policy.cpu().numpy(),
            "单次遗传算法最优累计利润": best_fitness,
            "平均累计利润": mean_profit,
            "95%置信区间": ci_95
        })
    
    print("\n所有情境下的最优决策结果：")
    for res in results:
        print(f"情境 {res['情境']}: 最优策略 (x1~x7) = {res['最优策略']}, 单次遗传算法最优累计利润 = {res['单次遗传算法最优累计利润']:.4f}")
        print(f"           平均累计利润 = {res['平均累计利润']:.2f}, 95%置信区间 = ±{res['95%置信区间']:.2f}\n")
    

    sensitivity_params = [
        ("P_std", P_std_values),
        ("pop_size", pop_size_values),
        ("num_generations", num_generations_values),
        ("mutation_rate", mutation_rate_values),
        ("num_parents", num_parents_values)
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n====== 情境 {scenario['情境']} 的敏感性分析 ======")
        set_scenario_params(scenario)
        for param_name, param_values in sensitivity_params:
            print(f"\n----- 情境 {scenario['情境']} 敏感性分析参数：{param_name} -----")
            results_df = sensitivity_analysis(param_name, param_values, scenario_index=i, num_trials=10)
            # 绘图并保存 csv 文件，函数内部会自动保存为 sensitivity_analysis_{param_name}_scenario_{i+1}.csv
            plot_sensitivity_analysis(results_df, param_name, i)

    

