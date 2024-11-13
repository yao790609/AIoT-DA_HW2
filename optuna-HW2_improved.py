# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:44:52 2024

@author: User
"""
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from optuna.samplers import TPESampler

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

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 使用 Optuna 進行特徵選擇和模型優化
def objective(trial):
    # 使用 SelectKBest 進行特徵選擇
    k = trial.suggest_int('k', 1, X.shape[1])  # 搜索 k 個最佳特徵
    kbest = SelectKBest(score_func=f_classif, k=k)
    X_kbest = kbest.fit_transform(X_scaled, y)
    
    # Logistic Regression 參數優化
    C = trial.suggest_loguniform('C', 1e-4, 10.0)  # 正則化強度參數
    solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
    
    # 使用邏輯回歸進行模型訓練
    X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=C, solver=solver, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 評估模型表現
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# 使用 Optuna 的 TPE 方法來進行超參數優化
study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=50)

# 獲取最佳參數
best_trial = study.best_trial
print(f"最佳參數: {best_trial.params}")
print(f"最佳準確率: {best_trial.value:.4f}")

# 3. 使用最佳參數進行最終模型訓練
k = best_trial.params['k']
C = best_trial.params['C']
solver = best_trial.params['solver']

# 根據最佳特徵數量進行特徵選擇
kbest = SelectKBest(score_func=f_classif, k=k)
X_selected = kbest.fit_transform(X_scaled, y)

# 將資料拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 使用最佳參數進行邏輯回歸模型訓練
model = LogisticRegression(C=C, solver=solver, max_iter=1000)
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

# 類別變數編碼，使用訓練集的編碼器
test_df['Sex'] = label_encoder_sex.transform(test_df['Sex'])
test_df['Embarked'] = label_encoder_embarked.transform(test_df['Embarked'])

# 使用標準化進行處理
X_test_scaled = scaler.transform(test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])

# 使用 SelectKBest 選擇的最佳特徵進行預測
X_test_selected = kbest.transform(X_test_scaled)

# 進行預測
y_pred_test = model.predict(X_test_selected)

# 5. 輸出預測結果
output_file = r'C:\Users\User\Downloads\titanic_predictions_optuna.csv'
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred_test})
output.to_csv(output_file, index=False)

print(f"預測結果已保存到: {output_file}")
print(f"模型在訓練集上的準確率: {accuracy_train:.4f}")
