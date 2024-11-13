import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. 資料載入與前處理
# 載入訓練集資料
df = pd.read_csv(r'C:\Users\User\Downloads\train.csv')

# 處理缺失值
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 類別變數編碼
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# 定義特徵與標籤
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# 2. 使用 RFE 選擇重要特徵
# 使用邏輯回歸模型進行遞迴特徵消除
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=4)  # 選擇4個重要特徵
rfe.fit(X, y)

# 列出被選中的特徵
print("被選中的特徵:", X.columns[rfe.support_])

# 篩選出重要特徵
X_selected = X[X.columns[rfe.support_]]

# 3. 訓練與評估模型
# 將資料拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 使用邏輯回歸進行模型訓練
model.fit(X_train, y_train)

# 進行測試集預測並評估準確率
y_pred_train = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_train)
print(f"模型在訓練集上的準確率: {accuracy:.4f}")

# 4. 測試集處理與預測
# 載入測試集資料
test_df = pd.read_csv(r'C:\Users\User\Downloads\test.csv')

# 處理缺失值
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# 類別變數編碼
test_df['Sex'] = label_encoder.fit_transform(test_df['Sex'])
test_df['Embarked'] = label_encoder.fit_transform(test_df['Embarked'])

# 使用 RFE 選擇的特徵進行預測
X_test_selected = test_df[X_selected.columns]

# 進行預測
y_pred_test = model.predict(X_test_selected)

# 5. 輸出預測結果
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred_test})
output.to_csv(r'C:\Users\User\Downloads\titanic_predictions_rfe.csv', index=False)
