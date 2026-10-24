import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
import os
import sys

# 添加当前目录到路径中，确保能导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入efficacy_function.py中的K_n函数
try:
    from efficacy_function import K_n
except ImportError:
    print("无法导入efficacy_function模块，请确保它在当前目录或Python路径中。")
    
    # 如果无法导入，定义一个简单的K_n函数作为备用
    def K_n(n):
        """简单K_n函数备用实现"""
        return 1 - np.exp(-0.01 * n)

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def load_double_sampling_data(csv_path='optimization_results.csv'):
    """
    加载双重抽样的数据
    
    参数:
    csv_path: CSV文件路径
    
    返回:
    DataFrame包含K_Target和期望抽样次数
    """
    try:
        df = pd.read_csv(csv_path)
        # 检查必要的列是否存在
        required_cols = ['K_Target', '期望抽样次数']
        for col in required_cols:
            if col not in df.columns:
                print(f"警告: CSV文件中缺少列 '{col}'")
        return df
    except Exception as e:
        print(f"加载CSV文件时出错: {e}")
        return None

def calculate_V(K, n, k1, k2):
    """
    计算评价指标V
    
    参数:
    K: 效度函数值
    n: 样本量
    k1, k2: 权重系数
    
    返回:
    V值
    """
    return k1 * K + k2 * (-n / 1000)  # 将n进行归一化处理，避免数值过大

def find_optimal_sample_size(k1=0.5, k2=0.5, max_n=1000, plot=True):
    """
    找到最优的样本量
    
    参数:
    k1, k2: 权重系数
    max_n: 最大样本量
    plot: 是否绘制图形
    
    返回:
    optimal_n_single: 单次抽样最优样本量
    optimal_n_double: 双重抽样最优样本量
    V_single_max: 单次抽样最大V值
    V_double_max: 双重抽样最大V值
    """
    # 获取单次抽样的效度和样本量
    n_values_single = np.arange(10, max_n + 1, 1)
    K_values_single = np.array([K_n(n) for n in n_values_single])
    
    # 计算单次抽样的V值
    V_values_single = np.array([calculate_V(K, n, k1, k2) for K, n in zip(K_values_single, n_values_single)])
    
    # 找到最大V值对应的索引
    optimal_idx_single = np.argmax(V_values_single)
    optimal_n_single = n_values_single[optimal_idx_single]
    V_single_max = V_values_single[optimal_idx_single]
    
    # 加载双重抽样数据
    df_double = load_double_sampling_data()
    
    if df_double is not None:
        # 计算双重抽样的V值
        df_double['V'] = df_double.apply(
            lambda row: calculate_V(row['K_Target'], row['期望抽样次数'], k1, k2), 
            axis=1
        )
        
        # 找到最大V值
        optimal_row = df_double.loc[df_double['V'].idxmax()]
        optimal_n_double = optimal_row['期望抽样次数']
        V_double_max = optimal_row['V']
        
        # 绘制图形
        if plot:
            plt.figure(figsize=(12, 8))
            
            # 单次抽样曲线
            plt.plot(n_values_single, V_values_single, 'b-', label='单次抽样')
            plt.scatter(optimal_n_single, V_single_max, color='red', marker='o', 
                        s=100, label=f'单次抽样最优点 (n={optimal_n_single:.1f}, V={V_single_max:.4f})')
            
            # 双重抽样曲线
            plt.plot(df_double['期望抽样次数'], df_double['V'], 'g-', label='双重抽样')
            plt.scatter(optimal_n_double, V_double_max, color='purple', marker='o', 
                        s=100, label=f'双重抽样最优点 (n={round(optimal_n_double)}, V={V_double_max:.4f})')
            
            plt.xlabel('样本量 (n)')
            plt.ylabel('评价指标 (V)')
            plt.title(f'评价指标V vs 样本量n (k1={k1}, k2={k2})')
            plt.grid(True)
            plt.legend()
            
            # 保存图像
            plt.savefig(f'V_vs_n_k1_{k1}_k2_{k2}.png')
            print(f'图像已保存为 V_vs_n_k1_{k1}_k2_{k2}.png')
        
        return optimal_n_single, optimal_n_double, V_single_max, V_double_max
    else:
        print("无法获取双重抽样数据，仅返回单次抽样结果")
        return optimal_n_single, None, V_single_max, None

def main():
    """主函数"""
    # 设置权重系数的多组组合
    weight_combinations = [
        (0.5, 0.5),
        (0.99, 0.01)
    ]
    
    # 对每组权重系数计算最优样本量
    for k1, k2 in weight_combinations:
        print(f"\n权重系数: k1={k1}, k2={k2}")
        optimal_n_single, optimal_n_double, V_single_max, V_double_max = find_optimal_sample_size(k1, k2)
        
        print("\n单次抽样结果:")
        print(f"  最优样本量 n = {optimal_n_single}")
        print(f"  最大评价指标 V = {V_single_max:.4f}")
        print(f"  对应效度 K(n) = {K_n(optimal_n_single):.4f}")
        
        if optimal_n_double is not None:
            print("\n双重抽样结果:")
            print(f"  最优期望样本量 E(n) = {round(optimal_n_double)}")
            print(f"  最大评价指标 V = {V_double_max:.4f}")
        
        # 确定最佳方案
        if optimal_n_double is not None:
            if V_single_max > V_double_max:
                print("\n推荐方案: 单次抽样")
                print(f"推荐样本量: n = {optimal_n_single}")
            else:
                print("\n推荐方案: 双重抽样")
                print(f"推荐期望样本量: E(n) = {round(optimal_n_double)}")
        else:
            print("\n推荐方案: 单次抽样")
            print(f"推荐样本量: n = {optimal_n_single}")

if __name__ == "__main__":
    main()
