from joblib import dump
from Data.data_processing import default_credit_card,kaggle_ieee_fraud
from costcla.datasets import load_creditscoring1,load_creditscoring2

# 加载第一个数据集
D = load_creditscoring1() 
X1, y1, cost_mat1, feature_name1 = D.data, D.target, D.cost_mat, D.feature_names
dump((X1, y1, cost_mat1, feature_name1), 'credit_data_1.joblib')

# 加载第二个数据集
Data = load_creditscoring2() 
X2, y2, cost_mat2, feature_name2 = Data.data, Data.target, Data.cost_mat, Data.feature_names
dump((X2, y2, cost_mat2, feature_name2), 'credit_data_2.joblib')

# 加载IEEE Fraud数据集
Data1, labels1, cost_matrix1, feature_names1 = kaggle_ieee_fraud() 
dump((Data1, labels1, cost_matrix1, feature_names1), 'data_IEEE.joblib')

# 加载Default Credit Card数据集
Data2, labels2, cost_matrix2, feature_names2 = default_credit_card() 
dump((Data2, labels2, cost_matrix2, feature_names2), 'data_DCCC.joblib')

# 输出保存成功的信息（可选）
print("数据已成功保存到文件：")
print("credit_data_1.joblib")
print("credit_data_2.joblib")
print("data_IEEE.joblib")
print("data_DCCC.joblib")

