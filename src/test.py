import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
def probability_accept_H1(n1, c1, r1, n2, r2, p=0.1):
    # 在第一轮直接接受的概率
    z1 = (c1 - p) / np.sqrt(p * (1 - p) / n1)
    prob_direct_accept = norm.cdf(z1)
    
    # 在第二轮接受的概率
    z2 = (r2 - p) / np.sqrt(p * (1 - p) / n2)
    prob_accept_2nd_round = (norm.cdf((r1 - p) / np.sqrt(p * (1 - p) / n1)) - 
                           norm.cdf((c1 - p) / np.sqrt(p * (1 - p) / n1))) * norm.cdf(z2)
    
    return prob_direct_accept + prob_accept_2nd_round

n1= 100
c1= 0.05
r1= 0.20
n2= 100
r2= 0.1

# 绘制函数在p取 0 ~ 1 之间的图像
p = np.linspace(0, 1, 1000)
prob = probability_accept_H1(n1, c1, r1, n2, r2, p)

# 绘制图像
plt.plot(p, prob)
plt.show()

# print(probability_reject_H1(n1, c1, r1, n2, r2, 0.1))
# print(probability_accept_H1(n1, c1, r1, n2, r2, 0.1)+probability_reject_H0(n1, c1, r1, n2, r2, 0.05))