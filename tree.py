import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
complaint_data = pd.read_csv('投诉数据.csv')
non_complaint_data = pd.read_csv('未投诉数据.csv')

# 添加标签列
complaint_data['label'] = 1
non_complaint_data['label'] = 0

# 合并数据
data = pd.concat([complaint_data, non_complaint_data], ignore_index=True)

# 显示数据的基本信息
data.info()

# 分离特征和标签
X = data.drop('label', axis=1)
y = data['label']

# 填充缺失值和处理分类变量
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = preprocessor.fit_transform(X)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 决策树模型
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 随机森林模型
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 评价模型
print("决策树模型评价:")
print(classification_report(y_test, y_pred_dt))
print("随机森林模型评价:")
print(classification_report(y_test, y_pred_rf))

# 混淆矩阵
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# 可视化混淆矩阵
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap="Blues")
plt.title("confusion matrix for decision tree")
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Greens")
plt.title("confusion matrix for random forest")
plt.show()

# 特征重要性
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制特征重要性条形图
plt.figure(figsize=(12, 6))
plt.title("importance of model features")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), indices, rotation=90)
plt.show()

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=numerical_features.tolist() + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)), class_names=['not complained', 'complained'], rounded=True)
plt.show()

# 计算投诉行为概率
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# 保存结果到 CSV 文件
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf, 'Probability': y_proba_rf})
results.to_csv('prediction_results.csv', index=False)
