import os
import pandas as pd

# 指定你的 CSV 文件所在的文件夹路径
folder_path = '.'  # 修改为你的实际路径

# 参数名与情境名称
parameters = ['P_std', 'pop_size', 'num_generations', 'mutation_rate', 'num_parents']
scenarios = [f'scenario_{i+1}' for i in range(6)]

# 存储所有读取的数据
all_data = []

# 遍历文件夹中所有符合命名规则的 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        for param in parameters:
            for scenario_idx, scenario_name in enumerate(scenarios):
                if f'sensitivity_analysis_{param}_{scenario_name}.csv' in filename:
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path)
                    
                    # 添加参数和情境信息
                    df['parameter'] = param
                    df['scenario'] = scenario_name
                    
                    all_data.append(df)
                    break  # 找到匹配文件就退出内层循环

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)

# 提取关键信息并计算平均值
summary_mean = combined_df.groupby(['parameter', 'scenario', 'param_value']).agg({
    'mean_profit': 'mean',
    'ci_95': 'mean'
}).reset_index()

# 保存为 CSV 文件
summary_mean_file = os.path.join(folder_path, 'sensitivity_analysis_summary_mean.csv')
summary_mean.to_csv(summary_mean_file, index=False)

# 提取最佳参数值
best_params = summary_mean.loc[summary_mean.groupby(['parameter', 'scenario'])['mean_profit'].idxmax()]
best_params_file = os.path.join(folder_path, 'best_parameters_summary.csv')
best_params.to_csv(best_params_file, index=False)

print(f"汇总完成！以下文件已生成：\n1. {summary_mean_file}\n2. {best_params_file}")
