from sympy import symbols, sqrt, integrate, exp, pi, simplify, erf
from sympy.stats import Normal, cdf
from sympy import *
init_printing()

# 定义符号变量
n, p, p0, u_alpha, sigma = symbols('n p p0 u_alpha sigma', positive=True)
x = symbols('x')
# 定义效度函数K(n,p)的原始表达式
# 假设p0是实际次品率，p是标称次品率
p_critical = p + sqrt(p*(1-p)/n)*u_alpha

# 定义正态分布密度函数
sigma_val = sqrt(p0*(1-p0)/n)  # 二项分布标准差
f_p0 = (1/(sqrt(2*pi*sigma_val**2))) * exp(-(p0-p)**2/(2*sigma_val**2))

# 效度函数的积分表达式
K_n_p_integral = integrate(f_p0, (p0, p_critical, 1))

# 使用标准正态分布的性质简化
# 设z = (p0-p)/sigma_val
# 则p0 = p + sigma_val*z
# 积分转换为标准正态分布的积分

# 积分下限对应的z值
z_lower = (p_critical - p)/sigma_val
# 积分上限对应的z值
z_upper = (1 - p)/sigma_val

# 使用标准正态分布的累积分布函数
X = Normal('X', 0, 1)
K_n_p_simplified = 1 - cdf(X)(z_lower)

# 进一步化简
K_n_p_final = simplify(K_n_p_simplified)

# 对于特定参数，比如p=0.1, u_alpha=1.645（对应95%的信度）
K_n_p_specific = K_n_p_final.subs({p: 0.1, u_alpha: 1.645})

# 展示结果
print("原始K(n,p)积分表达式：")
pprint(K_n_p_integral)
print("\n使用标准正态分布性质简化后：")
pprint(K_n_p_simplified)
print("\n进一步化简：")
pprint(K_n_p_final)
print("\n在p=0.1, u_alpha=1.645时的表达式：")
pprint(K_n_p_specific)

# 对K(n)也进行类似处理
# K(n) = ∫(0.10 to 1) K(n,p)·p(p0) dp0，其中p(p0)=1
# 这是平均效度函数
K_n = integrate(K_n_p_final, (p, 0.1, 1))
K_n_simplified = simplify(K_n)

print("\n平均效度K(n)的表达式：")
pprint(K_n_simplified)
