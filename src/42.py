import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import betainc  # 用于计算 Beta 后验累积分布函数
from scipy.stats import norm       # 获取标准正态分布分位数

# -------------------------------
# 设置 GPU 设备（如果可用）
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 全局参数（初始默认值，以情境1为参考，后续根据情境更新）
# -------------------------------
# 零配件1参数
C1b   = torch.tensor(4.0, device=device)    # 采购单价
C1d   = torch.tensor(2.0, device=device)    # 检测成本
P1_mean = 0.10                             # 零配件1标称次品率

# 零配件2参数
C2b   = torch.tensor(18.0, device=device)   # 采购单价
C2d   = torch.tensor(3.0, device=device)    # 检测成本
P2_mean = 0.10                             # 零配件2标称次品率

# 成品参数
Pf    = 0.10                               # 成品次品率（除零配件外的其他不确定性）
Cf_a  = torch.tensor(6.0, device=device)    # 装配成本
Cf_d  = torch.tensor(3.0, device=device)    # 成品检测成本
S     = torch.tensor(56.0, device=device)    # 市场售价
L     = torch.tensor(6.0, device=device)     # 调换损失

# 拆解参数
Cfr   = torch.tensor(5.0, device=device)    # 拆解费用
# 回收率参数
R1_full = 0.8    # 拆解后，若检测则零配件1回收率
R1_half = 0.4    # 若不检测则零配件1部分回收
R2_full = 0.8    # 同理，零配件2检测时回收率
R2_half = 0.4

# 蒙特卡罗模拟参数：用于模拟真实生产中零配件次品率的波动
num_samples = 10000  
P_std = 0.02         # 次品率采样标准差

# 抽样检测参数（用于问题4扩展）：抽样时用到的样本量
sampling_size = 50   # 单次检测抽样样本量（备用参数）

# 默认双重抽样参数（第一轮和第二轮抽样样本数），后续可尝试从 CSV 文件加载
n1 = 30  # 第一轮抽样样本量
n2 = 20  # 第二轮抽样样本量

# -------------------------------
# 随机种子设置
# -------------------------------
torch.manual_seed(42)
random.seed(42)

# -------------------------------
# 参数范围定义（用于敏感性分析）
# -------------------------------
P_std_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
pop_size_values = [20, 50, 100, 200]
num_generations_values = [10, 20, 30, 50]
mutation_rate_values = [0.05, 0.1, 0.15, 0.2]
num_parents_values = [10, 20, 30, 40]

# -------------------------------
# 各情境参数设定（依据题目要求）
# -------------------------------
scenarios = [
    {"情境": 1, "零配件1_次品率": 0.10, "零配件1_购买单价": 4.0, "零配件1_检测成本": 2.0,
     "零配件2_次品率": 0.10, "零配件2_购买单价": 18.0, "零配件2_检测成本": 3.0,
     "成品_次品率": 0.10, "成品_装配成本": 6.0, "成品_检测成本": 3.0, "成品_市场售价": 56.0,
     "不合格品_调换损失": 6.0, "不合格品_拆解费用": 5.0},
    {"情境": 2, "零配件1_次品率": 0.20, "零配件1_购买单价": 4.0, "零配件1_检测成本": 2.0,
     "零配件2_次品率": 0.20, "零配件2_购买单价": 18.0, "零配件2_检测成本": 3.0,
     "成品_次品率": 0.20, "成品_装配成本": 6.0, "成品_检测成本": 3.0, "成品_市场售价": 56.0,
     "不合格品_调换损失": 6.0, "不合格品_拆解费用": 5.0},
    {"情境": 3, "零配件1_次品率": 0.10, "零配件1_购买单价": 4.0, "零配件1_检测成本": 2.0,
     "零配件2_次品率": 0.10, "零配件2_购买单价": 18.0, "零配件2_检测成本": 3.0,
     "成品_次品率": 0.10, "成品_装配成本": 6.0, "成品_检测成本": 3.0, "成品_市场售价": 56.0,
     "不合格品_调换损失": 30.0, "不合格品_拆解费用": 5.0},
    {"情境": 4, "零配件1_次品率": 0.20, "零配件1_购买单价": 4.0, "零配件1_检测成本": 1.0,
     "零配件2_次品率": 0.20, "零配件2_购买单价": 18.0, "零配件2_检测成本": 1.0,
     "成品_次品率": 0.20, "成品_装配成本": 6.0, "成品_检测成本": 2.0, "成品_市场售价": 56.0,
     "不合格品_调换损失": 30.0, "不合格品_拆解费用": 5.0},
    {"情境": 5, "零配件1_次品率": 0.10, "零配件1_购买单价": 4.0, "零配件1_检测成本": 8.0,
     "零配件2_次品率": 0.20, "零配件2_购买单价": 18.0, "零配件2_检测成本": 1.0,
     "成品_次品率": 0.10, "成品_装配成本": 6.0, "成品_检测成本": 2.0, "成品_市场售价": 56.0,
     "不合格品_调换损失": 10.0, "不合格品_拆解费用": 5.0},
    {"情境": 6, "零配件1_次品率": 0.05, "零配件1_购买单价": 4.0, "零配件1_检测成本": 2.0,
     "零配件2_次品率": 0.05, "零配件2_购买单价": 18.0, "零配件2_检测成本": 3.0,
     "成品_次品率": 0.05, "成品_装配成本": 6.0, "成品_检测成本": 3.0, "成品_市场售价": 56.0,
     "不合格品_调换损失": 10.0, "不合格品_拆解费用": 40.0}
]

# -------------------------------
# 根据情境更新全局参数
# -------------------------------
def set_scenario_params(scenario):
    global C1b, C1d, P1_mean, C2b, C2d, P2_mean, Pf, Cf_a, Cf_d, S, L, Cfr
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

# -------------------------------
# 从 CSV 文件加载 Part1 中的最优抽样参数（如 n1, n2）
# -------------------------------
def load_optimal_params(csv_file="optimization_results.csv", target_K=0.90):
    try:
        df = pd.read_csv(csv_file)
        # 筛选满足目标效度的方案
        df_valid = df[df["K_Target"] >= target_K]
        if not df_valid.empty:
            best_row = df_valid.sort_values("K_Target", ascending=False).iloc[0]
        else:
            best_row = df.sort_values("K_Target", ascending=False).iloc[0]
        opt_n1 = int(best_row["n1"])
        opt_n2 = int(best_row["n2"])
        print(f"加载 Part1 最优抽样参数: n1 = {opt_n1}, n2 = {opt_n2}")
        return opt_n1, opt_n2
    except Exception as e:
        print("加载 CSV 参数出错:", e)
        return None, None

# -------------------------------
# 贝叶斯双重抽样检测函数
# -------------------------------
def bayesian_double_sampling_decision(P_samples, n1, n2, P_nominal, T_accept=0.90, T_reject=0.10):
    # 第一轮抽样
    x1 = torch.distributions.Binomial(total_count=n1, probs=P_samples).sample()
    alpha1 = 1.0 + x1
    beta1 = 1.0 + n1 - x1
    alpha1_np = alpha1.cpu().numpy()
    beta1_np = beta1.cpu().numpy()
    P_nominal_np = np.full_like(alpha1_np, P_nominal)
    posterior_prob1_np = betainc(alpha1_np, beta1_np, P_nominal_np)
    posterior_prob1 = torch.from_numpy(posterior_prob1_np).to(device)
    
    # 初始化决策张量
    decision = torch.empty_like(posterior_prob1)
    accepted_mask = posterior_prob1 >= T_accept
    rejected_mask = posterior_prob1 <= T_reject
    decision[accepted_mask] = 1.0
    decision[rejected_mask] = 0.0
    second_round_mask = (~accepted_mask) & (~rejected_mask)
    if second_round_mask.sum() > 0:
        P_samples_second = P_samples[second_round_mask]
        x2 = torch.distributions.Binomial(total_count=n2, probs=P_samples_second).sample()
        alpha2 = alpha1[second_round_mask] + x2
        beta2 = beta1[second_round_mask] + n2 - x2
        alpha2_np = alpha2.cpu().numpy()
        beta2_np = beta2.cpu().numpy()
        posterior_prob2_np = betainc(alpha2_np, beta2_np, np.full_like(alpha2_np, P_nominal))
        posterior_prob2 = torch.from_numpy(posterior_prob2_np).to(device)
        decision_second = (posterior_prob2 >= T_accept).float()
        decision[second_round_mask] = decision_second
    return decision

# -------------------------------
# 更新后的效度函数 compute_K：使用 Part1 中理论型定义
# -------------------------------
def compute_K(n, p, threshold=0.10, alpha=0.05):
    u = norm.ppf(alpha)  # 例如 alpha=0.05 时 u ≈ -1.645
    term = p + np.sqrt(p*(1-p)/n) * u
    K_val = np.clip(term, 0, 1)
    return K_val

# -------------------------------
# 模拟单轮生产过程（包括检测、装配与拆解流程）的函数
# -------------------------------
def simulate_round(policy, scale=1.0, num_samples=num_samples):
    """
    policy: 长度为7的二值向量，依次表示:
      x1: 是否检测零配件1,
      x2: 是否检测零配件2,
      x3: 是否检测成品,
      x4: 是否拆解生产过程中不合格成品,
      x5: 拆解后零配件1是否检测,
      x6: 拆解后零配件2是否检测,
      x7: 用户退回不合格品是否拆解.
    scale: 本轮计划生产的成品数.
    返回：(平均利润, 回收零配件1数量, 回收零配件2数量)
    """
    cand = policy.float()
    # 生成零配件次品率的蒙特卡罗样本（体现生产中的不确定性）
    P1_samples = torch.clamp(torch.normal(mean=P1_mean, std=P_std, size=(num_samples,), device=device), 0.0, 1.0)
    P2_samples = torch.clamp(torch.normal(mean=P2_mean, std=P_std, size=(num_samples,), device=device), 0.0, 1.0)
    
    if cand[0] == 1:
        effective_P1 = bayesian_double_sampling_decision(P1_samples, n1, n2, P_nominal=P1_mean, T_accept=0.90, T_reject=0.10)
    else:
        effective_P1 = P1_samples

    if cand[1] == 1:
        effective_P2 = bayesian_double_sampling_decision(P2_samples, n1, n2, P_nominal=P2_mean, T_accept=0.90, T_reject=0.10)
    else:
        effective_P2 = P2_samples
    
    # 计算成品合格概率：只有当两个零配件均合格且成品检测合格时才认为成品合格
    good_prob = (1 - effective_P1) * (1 - effective_P2) * (1 - Pf)
    rand_vals = torch.rand(num_samples, device=device)
    product_good = (rand_vals < good_prob).float()
    
    # 成品检测策略
    if cand[2] == 1:
        revenue = scale * S * product_good
        prod_inspection_cost = scale * Cf_d
        replacement_loss = torch.zeros_like(revenue)
        disassemble_flag = cand[3]
    else:
        revenue = scale * S * torch.ones_like(product_good)
        prod_inspection_cost = 0.0
        replacement_loss = scale * L * (1 - product_good)
        disassemble_flag = cand[6]
    
    cost_components = scale * (C1b + C2b)
    comp_inspection_cost = scale * (cand[0] * C1d + cand[1] * C2d)
    assembly_cost = scale * Cf_a
    disassembly_cost = disassemble_flag * scale * Cfr
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

# -------------------------------
# 反馈循环模拟：根据初始库存更新生产与库存情况
# -------------------------------
def simulate_feedback(policy, num_samples=num_samples, tol=1e-3, max_iter=10, inv1_init=10000, inv2_init=10000):
    total_profit = 0.0
    inv1 = inv1_init
    inv2 = inv2_init
    iter_count = 0
    desired_scale = 1000.0  # 每轮生产的计划成品数
    while iter_count < max_iter and inv1 > 0 and inv2 > 0:
        scale = min(desired_scale, inv1, inv2)
        profit, rec1, rec2 = simulate_round(policy, scale=scale, num_samples=num_samples)
        total_profit += profit.item()
        inv1 = inv1 - scale + rec1.item()
        inv2 = inv2 - scale + rec2.item()
        print(f"迭代 {iter_count+1}: scale = {scale:.2f}, 零件1库存 = {inv1:.2f}, 零件2库存 = {inv2:.2f}, 本轮利润 = {profit.item():.2f}")
        if scale < tol:
            break
        iter_count += 1
    return total_profit

# -------------------------------
# 遗传算法部分
# -------------------------------
def initialize_population(pop_size, dim=7):
    return torch.randint(0, 2, (pop_size, dim), device=device)

def fitness_function(policy, num_samples=num_samples, inv1_init=10000, inv2_init=10000, n=50, w1=0.5, w2=0.5):
    total_profit = simulate_feedback(policy, num_samples=num_samples, inv1_init=inv1_init, inv2_init=inv2_init)
    if total_profit < 0:
        return 0
    # 蒙特卡罗生成零配件次品率样本并计算均值
    P1_samples = torch.clamp(torch.normal(mean=P1_mean, std=P_std, size=(num_samples,), device=device), 0.0, 1.0)
    P2_samples = torch.clamp(torch.normal(mean=P2_mean, std=P_std, size=(num_samples,), device=device), 0.0, 1.0)
    P1_actual = P1_samples.mean().item()
    P2_actual = P2_samples.mean().item()
    # 计算效度函数值
    K1 = compute_K(n, P1_actual)
    K2 = compute_K(n, P2_actual)
    # 加权组合效度
    fitness_value = total_profit * (w1 * K1 + w2 * K2)
    return fitness_value

def selection(population, fitnesses, num_parents):
    sorted_indices = torch.argsort(fitnesses, descending=True)
    return population[sorted_indices[:num_parents]]

def crossover(parent1, parent2):
    dim = parent1.shape[0]
    point = random.randint(1, dim - 1)
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
            parent1 = parents[random.randint(0, num_parents - 1)]
            parent2 = parents[random.randint(0, num_parents - 1)]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = torch.stack(new_population, dim=0)
    return best_policy, best_fitness

# -------------------------------
# 自动评估函数：对同一策略多次运行，计算平均累计利润与 95% 置信区间
# -------------------------------
def evaluate_strategy(policy, num_trials=20, num_samples=num_samples, inv1_init=10000, inv2_init=10000):
    profits = []
    for _ in range(num_trials):
        total_profit = simulate_feedback(policy, num_samples=num_samples, inv1_init=inv1_init, inv2_init=inv2_init)
        profits.append(total_profit)
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    ci_95 = 1.96 * std_profit / np.sqrt(num_trials)
    return mean_profit, ci_95, profits




# -------------------------------
# 主程序入口
# -------------------------------
if __name__ == "__main__":
    # 尝试加载 Part1 最优抽样参数，如果可行，则更新 n1 和 n2
    opt_n1, opt_n2 = load_optimal_params()
    if opt_n1 is not None and opt_n2 is not None:
        n1 = opt_n1
        n2 = opt_n2
    else:
        print("使用默认的 n1 =", n1, ", n2 =", n2)
    
    results = []
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
    

    
