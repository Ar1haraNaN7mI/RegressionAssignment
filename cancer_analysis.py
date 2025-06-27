import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['font.family'] = 'sans-serif'

# 加载数据
print("正在加载数据...")
df = pd.read_excel('cancerdeaths.xlsx')

# 查看数据基本信息
print(f"数据形状: {df.shape}")
print("\n列名:")
print(df.columns.tolist())

# 查看数据样本
print("\n前5行数据:")
print(df.head())

# 数据清洗 - 处理非法字符
print("\n========== 数据清洗 ==========")
illegal_chars = ["*", "**", "***", "?", "？", "??", "？？"]

# 检查每列中是否包含非法字符
for column in df.columns:
    if df[column].dtype == 'object':
        print(f"检查列 {column} 中的非法字符...")
        for char in illegal_chars:
            mask = df[column].astype(str).str.contains(char, regex=False, na=False)
            if mask.any():
                print(f"  在列 {column} 中发现 {mask.sum()} 个 '{char}' 字符")
                df.loc[mask, column] = np.nan
    else:
        # 对于数值列，检查是否有非法字符（如果被读取为字符串）
        temp_series = df[column].astype(str)
        for char in illegal_chars:
            mask = temp_series.str.contains(char, regex=False, na=False)
            if mask.any():
                print(f"  在数值列 {column} 中发现 {mask.sum()} 个 '{char}' 字符")
                df.loc[mask, column] = np.nan

print("数据清洗完成！")

# 分析缺失值
print("\n========== 缺失值分析 ==========")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失值数量': missing_data,
    '缺失百分比': missing_percent
}).sort_values('缺失百分比', ascending=False)

print("缺失值统计:")
print(missing_df[missing_df['缺失值数量'] > 0])

# 识别数值列和类别列
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n数值列 ({len(numerical_columns)}): {numerical_columns}")
print(f"类别列 ({len(categorical_columns)}): {categorical_columns}")

# 数据预处理
print("\n========== 数据预处理 ==========")
df_processed = df.copy()

# 1. 处理类别变量 - 填充最频繁的类别
print("处理类别变量...")
for col in categorical_columns:
    if df_processed[col].isnull().any():
        mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "unknown"
        df_processed[col].fillna(mode_value, inplace=True)
        print(f"  {col}: 填充 {df_processed[col].isnull().sum()} 个缺失值，使用最频繁值: {mode_value}")

# 2. 对类别变量进行标签编码（替代独热编码）
print("\n进行标签编码...")
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} 个唯一值 -> 编码为 0-{len(le.classes_)-1}")

# 删除原始类别列，保留编码后的列
df_processed = df_processed.drop(columns=categorical_columns)
print(f"标签编码后的数据形状: {df_processed.shape}")

# 3. 处理数值变量 - 使用KNN插补
print("\n处理数值变量...")
numerical_columns_current = df_processed.select_dtypes(include=[np.number]).columns.tolist()

# 检查数值变量中的缺失值
missing_numerical = df_processed[numerical_columns_current].isnull().sum()
print("数值变量缺失值统计:")
print(missing_numerical[missing_numerical > 0])

if missing_numerical.sum() > 0:
    print("使用KNN插补填充数值缺失值...")
    knn_imputer = KNNImputer(n_neighbors=5)
    df_processed[numerical_columns_current] = knn_imputer.fit_transform(df_processed[numerical_columns_current])
    print("KNN插补完成！")

# 验证没有缺失值
print(f"\n预处理后的缺失值总数: {df_processed.isnull().sum().sum()}")

# 探索性数据分析
print("\n========== 探索性数据分析 ==========")

# 目标变量分析
target_columns = ['incidenceRate', 'deathRate']
for target in target_columns:
    if target in df_processed.columns:
        print(f"\n{target} 统计信息:")
        print(df_processed[target].describe())

# 相关性分析
print("\n========== 相关性分析 ==========")
correlation_matrix = df_processed.corr()

# 找出与目标变量相关性最高的特征
for target in target_columns:
    if target in df_processed.columns:
        correlations = correlation_matrix[target].abs().sort_values(ascending=False)
        print(f"\n与 {target} 相关性最高的前10个特征:")
        print(correlations.head(11))  # 包括自己，所以取11个

# 建立回归模型
print("\n========== 建立回归模型 ==========")

# 准备特征和目标变量
feature_columns = [col for col in df_processed.columns if col not in target_columns]
X = df_processed[feature_columns]
y_incidence = df_processed['incidenceRate']
y_death = df_processed['deathRate']

print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

# 分别为两个目标变量建立模型
results = {}

for target_name, y in [('incidenceRate', y_incidence), ('deathRate', y_death)]:
    print(f"\n========== {target_name} 回归分析 ==========")
    
    # 训练测试集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 随机森林模型（减少树的数量以加快运行）
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 模型预测
    y_pred = rf_model.predict(X_test)
    
    # 模型评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"随机森林模型性能:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n前10个最重要特征:")
    print(feature_importance.head(10))
    
    # 保存结果
    results[target_name] = {
        'model': rf_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'mse': mse,
        'r2': r2,
        'feature_importance': feature_importance
    }

# SHAP分析
print("\n========== SHAP分析 ==========")

for target_name in results.keys():
    print(f"\n========== {target_name} SHAP分析 ==========")
    
    model = results[target_name]['model']
    X_train = results[target_name]['X_train']
    X_test = results[target_name]['X_test']
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    # 计算SHAP值 - 使用较小的样本以加快计算
    sample_size = min(500, len(X_test))  # 进一步减少样本数
    X_sample = X_test.iloc[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    print(f"SHAP值计算完成 (样本数: {sample_size})")
    
    # 保存SHAP值
    results[target_name]['shap_values'] = shap_values
    results[target_name]['X_sample'] = X_sample
    results[target_name]['explainer'] = explainer
    
    # 特征重要性 (SHAP)
    shap_importance = pd.DataFrame({
        'feature': feature_columns,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    print(f"基于SHAP的前10个最重要特征:")
    print(shap_importance.head(10))
    
    # 保存SHAP重要性
    results[target_name]['shap_importance'] = shap_importance

# 可视化和结果输出
print("\n========== 结果总结 ==========")

for target_name in results.keys():
    print(f"\n{target_name} 模型总结:")
    print(f"  R² 分数: {results[target_name]['r2']:.4f}")
    print(f"  MSE: {results[target_name]['mse']:.4f}")
    
    # 输出最重要的特征
    top_features = results[target_name]['feature_importance'].head(5)
    print(f"  前5个最重要特征 (Random Forest):")
    for idx, row in top_features.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # 输出SHAP最重要的特征
    if 'shap_importance' in results[target_name]:
        top_shap_features = results[target_name]['shap_importance'].head(5)
        print(f"  前5个最重要特征 (SHAP):")
        for idx, row in top_shap_features.iterrows():
            print(f"    {row['feature']}: {row['shap_importance']:.4f}")

print("\n========== 类别变量编码信息 ==========")
for col, encoder in label_encoders.items():
    print(f"{col}: {len(encoder.classes_)} 个类别")
    if len(encoder.classes_) <= 10:
        print(f"  类别: {list(encoder.classes_)}")
    else:
        print(f"  前5个类别: {list(encoder.classes_[:5])}")

print("\n========== 分析完成 ==========")
print("所有预处理和SHAP分析已完成！")
print("结果已保存在 results 字典中，包含:")
print("- 模型对象")
print("- 训练和测试数据")
print("- 预测结果")
print("- 特征重要性")
print("- SHAP值和解释器")
print("- SHAP特征重要性") 