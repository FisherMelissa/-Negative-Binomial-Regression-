import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- 1. 生成模拟数据 (这次故意制造“过度离散”) ---
# 我们使用与项目一相同的设置，但在生成数据时引入更多随机性

np.random.seed(42)
N = 500

data = pd.DataFrame({
    'PeerRisk': np.random.uniform(0, 10, N),
    'ParentingStyle': np.random.randint(0, 2, N)
})

# 真实关系
true_log_mean = 0.1 + 0.2 * data['PeerRisk'] + 0.5 * data['ParentingStyle']

# 关键：使用负二项分布生成数据，而不是泊松
# alpha (或 1/size) 是离散参数，alpha > 0 表示过度离散
alpha = 0.5  
size = 1 / alpha
prob = size / (size + np.exp(true_log_mean))
data['MonthlyIncidents'] = np.random.negative_binomial(size, prob, N)

print("--- 模拟数据 (前5行) ---")
print(data.head())
print("\n--- 数据描述 ---")
print(f"平均事件数: {data['MonthlyIncidents'].mean():.2f}")
print(f"事件数方差: {data['MonthlyIncidents'].var():.2f}")
print("(注意：方差现在远大于均值，这是'过度离散'的典型标志)\n")

# --- 2. 拟合负二项回归模型 ---
X = sm.add_constant(data[['PeerRisk', 'ParentingStyle']])
y = data['MonthlyIncidents']

# 拟合 GLM, 指定 family 为 NegativeBinomial
# 这对应于你 README 中的 "NB Baseline"
nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
nb_results = nb_model.fit()

# --- 3. 打印结果 ---
print("\n--- 负二项回归模型结果 ---")
print(nb_results.summary())

print("\n--- 结论 ---")
print("模型摘要中的 'alpha' 值 (通常在末尾) 是对过度离散的度量。")
print("如果 alpha 显著大于0，说明使用 NB 模型是正确的。")
print("下一步：如果数据中还有大量的'0'，那么 NB 也不够，就需要你项目中的 ZINB 模型了。")
