# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:05:40 2024

@author: User
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 資料載入與前處理
df = pd.read_csv(r'C:\Users\User\Downloads\train.csv')

# 處理缺失值
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 類別變數獨熱編碼
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 定義特徵與標籤
X = df[['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 使用 RFE 選擇重要特徵
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=4)  # 選擇4個重要特徵
rfe.fit(X_scaled, y)

# 列出被選中的特徵
selected_features = X.columns[rfe.support_]
print("被選中的特徵:", selected_features)

# 篩選出重要特徵
X_selected = X_scaled[:, rfe.support_]

# 3. 訓練與評估模型
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 使用邏輯回歸進行模型訓練
model.fit(X_train, y_train)

# 進行測試集預測並評估準確率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在測試集上的準確率: {accuracy:.4f}")

# 4. 測試集處理與預測
test_df = pd.read_csv(r'C:\Users\User\Downloads\test.csv')

# 處理缺失值
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# 使用相同的獨熱編碼
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)

# 保持測試集的特徵與訓練集一致
for column in ['Sex_male', 'Embarked_Q', 'Embarked_S']:
    if column not in test_df.columns:
        test_df[column] = 0

# 標準化測試集數據
X_test_scaled = scaler.transform(test_df[['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']])
X_test_selected = X_test_scaled[:, rfe.support_]

# 進行預測
y_pred_test = model.predict(X_test_selected)

# 5. 輸出預測結果
output_file = r'C:\Users\User\Downloads\titanic_predictions_rfe.csv'
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred_test})
output.to_csv(output_file, index=False)

print(f"預測結果已保存到: {output_file}")

