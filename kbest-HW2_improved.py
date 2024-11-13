# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:17:16 2024

@author: User
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# 1. 資料載入與前處理
# 載入訓練集資料
df = pd.read_csv(r'C:\Users\User\Downloads\train.csv')

# 處理缺失值
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 類別變數編碼
label_encoder_sex = LabelEncoder()
label_encoder_embarked = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['Embarked'] = label_encoder_embarked.fit_transform(df['Embarked'])

# 定義特徵與標籤
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# 2. 使用 SelectKBest 選擇重要特徵
# 標準化數據（尤其是用於特徵選擇時需要標準化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用卡方檢驗進行特徵選擇
kbest = SelectKBest(score_func=f_classif, k=4)  # 選擇4個最重要的特徵
X_selected = kbest.fit_transform(X_scaled, y)

# 列出被選中的特徵
print("被選中的特徵:", X.columns[kbest.get_support()])

# 3. 訓練與評估模型
# 將資料拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 使用隨機森林進行模型訓練
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 進行測試集預測並評估準確率
y_pred_train = model.predict(X_test)
accuracy_train = accuracy_score(y_test, y_pred_train)
print(f"模型在測試集上的準確率: {accuracy_train:.4f}")

# 4. 測試集處理與預測
# 載入測試集資料
test_df = pd.read_csv(r'C:\Users\User\Downloads\test.csv')

# 處理缺失值
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# 類別變數編碼
test_df['Sex'] = label_encoder_sex.transform(test_df['Sex'])
test_df['Embarked'] = label_encoder_embarked.transform(test_df['Embarked'])

# 使用標準化進行處理
X_test_scaled = scaler.transform(test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])

# 使用 SelectKBest 選擇的特徵進行預測
X_test_selected = X_test_scaled[:, kbest.get_support()]

# 進行預測
y_pred_test = model.predict(X_test_selected)

# 5. 輸出預測結果
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred_test})
output_file = r'C:\Users\User\Downloads\titanic_predictions_kbest.csv'
output.to_csv(output_file, index=False)

print(f"預測結果已保存到: {output_file}")
print(f"模型在訓練集上的準確率: {accuracy_train:.4f}")

