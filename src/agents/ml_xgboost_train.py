import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import joblib
from sklearn.preprocessing import LabelEncoder

# 读取特征工程生成的csv
TRAIN_CSV = 'ml_training_data.csv'  # 你可以根据实际路径修改
CLASSIFIER_MODEL_PATH = 'xgb_signal_classifier.model'
REGRESSOR_MODEL_PATH = 'xgb_return_regressor.model'

def train_xgboost_models(train_csv=TRAIN_CSV):
    df = pd.read_csv(train_csv)
    # 选择特征列（去除目标变量和无关列）
    feature_cols = [col for col in df.columns if col not in ['ml_signal', 'next_return', 'date']]
    X = df[feature_cols]
    y_cls = df['ml_signal']
    y_reg = df['next_return']
    # 对所有object类型的特征做编码
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    # 标签映射：-1->0, 0->1, 1->2
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    y_cls_mapped = y_cls.map(label_map)
    # 分类模型（信号）
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls_mapped, test_size=0.2, random_state=42, stratify=y_cls_mapped)
    clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # 还原标签
    y_test_inv = y_test.map(inv_label_map)
    y_pred_inv = pd.Series(y_pred).map(inv_label_map)
    print('信号分类模型评估:')
    print(classification_report(y_test_inv, y_pred_inv))
    print('准确率:', accuracy_score(y_test_inv, y_pred_inv))
    joblib.dump(clf, CLASSIFIER_MODEL_PATH)
    # 回归模型（收益率）
    # 去除回归标签为NaN的样本
    reg_mask = ~y_reg.isna()
    X_reg = X[reg_mask]
    y_reg_clean = y_reg[reg_mask]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg_clean, test_size=0.2, random_state=42)
    reg = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)
    print('收益率回归模型评估:')
    print('MSE:', mean_squared_error(y_test_r, y_pred_r))
    joblib.dump(reg, REGRESSOR_MODEL_PATH)
    print('模型已保存:', CLASSIFIER_MODEL_PATH, REGRESSOR_MODEL_PATH)

if __name__ == '__main__':
    train_xgboost_models() 