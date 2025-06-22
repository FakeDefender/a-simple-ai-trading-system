# 需要安装 imbalanced-learn: pip install imbalanced-learn
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import joblib
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

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
    # 编码所有object类型特征为数字
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    # 标签映射：-1->0, 0->1, 1->2
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    y_cls_mapped = y_cls.map(label_map)
    # 填补所有特征的缺失值（用中位数）
    X = X.fillna(X.median(numeric_only=True))
    # 剔除y中为NaN的样本
    mask = ~y_cls_mapped.isna()
    X = X[mask]
    y_cls_mapped = y_cls_mapped[mask]
    # 检查并只保留有样本的类别
    unique_classes = np.array(sorted(y_cls_mapped.dropna().unique()))
    # 计算类别权重
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=unique_classes, y=y_cls_mapped)
    class_weight_dict = {k: v for k, v in zip(unique_classes, weights)}
    X_train, X_test, y_train, y_test = train_test_split(X, y_cls_mapped, test_size=0.2, random_state=42, stratify=y_cls_mapped)
    clf = xgb.XGBClassifier(objective='multi:softmax', num_class=len(unique_classes), eval_metric='mlogloss', scale_pos_weight=1)
    # xgboost不直接支持class_weight参数，需手动加sample_weight
    sample_weight = y_train.map(class_weight_dict)
    try:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = clf.predict(X_test)
        y_test_inv = y_test.map(inv_label_map)
        y_pred_inv = pd.Series(y_pred).map(inv_label_map)
        print('信号分类模型评估:')
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        print(classification_report(y_test_inv, y_pred_inv))
        print('准确率:', accuracy_score(y_test_inv, y_pred_inv))
        cm = confusion_matrix(y_test_inv, y_pred_inv)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('信号分类混淆矩阵')
        plt.savefig('signal_confusion_matrix.png')
        plt.close()
        # 特征重要性
        xgb.plot_importance(clf, max_num_features=10)
        plt.title('信号分类特征重要性')
        plt.savefig('signal_feature_importance.png')
        plt.close()
        # 特征选择建议
        importance = clf.feature_importances_
        feature_names = X.columns
        low_importance = [feature_names[i] for i, v in enumerate(importance) if v < 0.01]
        if len(low_importance) > 0:
            print('建议剔除低重要性特征:', low_importance)
        try:
            corr = pd.read_csv('feature_correlation.csv', index_col=0)
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.95:
                        high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
            if high_corr:
                print('建议检查高相关性特征对:', high_corr)
        except Exception as e:
            print('未能读取特征相关性文件:', e)
        joblib.dump(clf, CLASSIFIER_MODEL_PATH)
    except Exception as e:
        print('分类模型训练或评估异常:', e)
    try:
        # 回归模型（收益率）
        reg_mask = ~y_reg.isna()
        X_reg = X[reg_mask]
        y_reg_clean = y_reg[reg_mask]
        # 删除所有object类型特征
        X_reg = X_reg.select_dtypes(include=[int, float, bool])
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg_clean, test_size=0.2, random_state=42)
        reg = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
        reg.fit(X_train_r, y_train_r)
        y_pred_r = reg.predict(X_test_r)
        print('收益率回归模型评估:')
        from sklearn.metrics import r2_score
        print('MSE:', mean_squared_error(y_test_r, y_pred_r))
        print('R2:', r2_score(y_test_r, y_pred_r))
        # 残差图
        plt.scatter(y_test_r, y_pred_r - y_test_r)
        plt.xlabel('真实收益率')
        plt.ylabel('残差')
        plt.title('回归残差图')
        plt.savefig('regression_residuals.png')
        plt.close()
        # 预测vs真实
        plt.scatter(y_test_r, y_pred_r)
        plt.xlabel('真实收益率')
        plt.ylabel('预测收益率')
        plt.title('预测vs真实收益率')
        plt.savefig('regression_pred_vs_true.png')
        plt.close()
        joblib.dump(reg, REGRESSOR_MODEL_PATH)
        print('模型已保存:', CLASSIFIER_MODEL_PATH, REGRESSOR_MODEL_PATH)
    except Exception as e:
        print('回归模型训练或评估异常:', e)

if __name__ == '__main__':
    train_xgboost_models() 