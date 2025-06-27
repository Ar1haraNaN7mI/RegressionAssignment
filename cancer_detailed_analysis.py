import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['font.family'] = 'sans-serif'

print("正在加载数据...")
df = pd.read_excel('cancerdeaths.xlsx')
print(f"原始数据形状: {df.shape}")

# 数据清洗 - 处理非法字符
print("\n========== 数据清洗 ==========")
illegal_chars = ["*", "**", "***", "?", "？", "??", "？？"]

for column in df.columns:
    if df[column].dtype == 'object':
        for char in illegal_chars:
            mask = df[column].astype(str).str.contains(char, regex=False, na=False)
            if mask.any():
                print(f"清洗列 {column}: 发现 {mask.sum()} 个 '{char}' 字符")
                df.loc[mask, column] = np.nan

# 识别数值列和类别列
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n数值列数量: {len(numerical_columns)}")
print(f"类别列数量: {len(categorical_columns)}")

# 保存编码器信息
encoders_info = {}

# 数据预处理
print("\n========== 数据预处理 ==========")
df_processed = df.copy()

# 1. 处理类别变量 - 标签编码
print("标签编码...")
for col in categorical_columns:
    # 填充缺失值
    if df_processed[col].isnull().any():
        mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "unknown"
        df_processed[col].fillna(mode_value, inplace=True)
    
    # 标签编码
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    encoders_info[col] = {
        'encoder': le,
        'classes': len(le.classes_),
        'categories': list(le.classes_)
    }
    print(f"  {col}: {len(le.classes_)} 个类别")

# 2. 处理数值变量 - KNN插补
numerical_missing = df_processed[numerical_columns].isnull().sum().sum()
if numerical_missing > 0:
    print(f"KNN插补: {numerical_missing} 个缺失值")
    knn_imputer = KNNImputer(n_neighbors=5)
    df_processed[numerical_columns] = knn_imputer.fit_transform(df_processed[numerical_columns])

print(f"预处理后数据形状: {df_processed.shape}")
print(f"缺失值总数: {df_processed.isnull().sum().sum()}")

# 准备建模数据
target_columns = ['incidenceRate', 'deathRate']
feature_columns = [col for col in df_processed.columns if col not in target_columns]
X = df_processed[feature_columns]

print(f"\n特征数量: {len(feature_columns)}")

# 分别为两个目标变量建立模型
results = {}

for target_name in target_columns:
    print(f"\n========== {target_name} 回归分析 ==========")
    
    y = df_processed[target_name]
    
    # 训练测试集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 随机森林模型
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 模型预测和评估
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"模型性能: R² = {r2:.4f}, MSE = {mse:.4f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("前10个最重要特征 (Random Forest):")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # SHAP分析
    print("计算SHAP值...")
    explainer = shap.TreeExplainer(rf_model)
    
    # 使用小样本计算SHAP
    sample_size = min(500, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP特征重要性
    shap_importance = pd.DataFrame({
        'feature': feature_columns,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    print("前10个最重要特征 (SHAP):")
    for i, row in shap_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['shap_importance']:.4f}")
    
    # 保存结果
    results[target_name] = {
        'model': rf_model,
        'r2': r2,
        'mse': mse,
        'feature_importance': feature_importance,
        'shap_importance': shap_importance,
        'shap_values': shap_values,
        'X_sample': X_sample,
        'explainer': explainer,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_train': X_train,
        'X_test': X_test
    }

# 相关性分析
print("\n========== 相关性分析 ==========")
correlation_matrix = df_processed.corr()

for target in target_columns:
    correlations = correlation_matrix[target].abs().sort_values(ascending=False)
    print(f"\n与 {target} 相关性最高的前10个特征:")
    for feature, corr in correlations.head(11).items():  
        if feature != target:
            print(f"  {feature}: {corr:.4f}")

# 创建可视化
print("\n========== 创建可视化 ==========")

# 1. 特征重要性对比图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('特征重要性分析', fontsize=16)

for i, target_name in enumerate(target_columns):
    # Random Forest 特征重要性
    rf_top = results[target_name]['feature_importance'].head(10)
    axes[i, 0].barh(range(len(rf_top)), rf_top['importance'])
    axes[i, 0].set_yticks(range(len(rf_top)))
    axes[i, 0].set_yticklabels(rf_top['feature'])
    axes[i, 0].set_title(f'{target_name} - Random Forest 特征重要性')
    axes[i, 0].set_xlabel('重要性')
    
    # SHAP 特征重要性
    shap_top = results[target_name]['shap_importance'].head(10)
    axes[i, 1].barh(range(len(shap_top)), shap_top['shap_importance'])
    axes[i, 1].set_yticks(range(len(shap_top)))
    axes[i, 1].set_yticklabels(shap_top['feature'])
    axes[i, 1].set_title(f'{target_name} - SHAP 特征重要性')
    axes[i, 1].set_xlabel('SHAP 重要性')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("保存图片: feature_importance_comparison.png")

# 2. 目标变量分布图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, target in enumerate(target_columns):
    axes[i].hist(df_processed[target], bins=50, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{target} 分布')
    axes[i].set_xlabel(target)
    axes[i].set_ylabel('频率')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_distributions.png', dpi=300, bbox_inches='tight')
print("保存图片: target_distributions.png")

# 3. 相关性热图（选择重要特征）
important_features = []
for target_name in target_columns:
    top_features = results[target_name]['feature_importance'].head(5)['feature'].tolist()
    important_features.extend(top_features)

important_features = list(set(important_features + target_columns))
corr_subset = correlation_matrix.loc[important_features, important_features]

plt.figure(figsize=(12, 10))
sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('重要特征相关性热图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("保存图片: correlation_heatmap.png")

print("\n========== 模型总结 ==========")
for target_name in results.keys():
    result = results[target_name]
    print(f"\n{target_name}:")
    print(f"  R² 分数: {result['r2']:.4f}")
    print(f"  MSE: {result['mse']:.4f}")
    print(f"  RMSE: {np.sqrt(result['mse']):.4f}")
    
    print("  前5个最重要特征 (Random Forest):")
    for i, row in result['feature_importance'].head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    print("  前5个最重要特征 (SHAP):")
    for i, row in result['shap_importance'].head(5).iterrows():
        print(f"    {row['feature']}: {row['shap_importance']:.4f}")

print("\n========== 类别变量编码信息 ==========")
for col_name, info in encoders_info.items():
    print(f"{col_name}: {info['classes']} 个类别")
    if info['classes'] <= 10:
        print(f"  类别: {info['categories']}")
    else:
        print(f"  前5个类别: {info['categories'][:5]}")

print("\n========== 业务解释 ==========")
print("基于SHAP分析的关键发现:")

for target_name in results.keys():
    print(f"\n{target_name} 关键影响因素:")
    shap_top = results[target_name]['shap_importance'].head(5)
    
    for i, row in shap_top.iterrows():
        feature_name = row['feature']
        shap_val = row['shap_importance']
        
        # 获取相关性信息
        corr_val = correlation_matrix.loc[feature_name, target_name]
        
        print(f"  {feature_name}:")
        print(f"    SHAP重要性: {shap_val:.4f}")
        print(f"    相关系数: {corr_val:.4f}")
        
        # 如果是编码的类别变量，提供解释
        if feature_name in encoders_info:
            print(f"    类别数量: {encoders_info[feature_name]['classes']}")

print("\n========== 分析完成 ==========")
print("已完成:")
print("- 数据清洗和预处理")  
print("- 标签编码（避免维度爆炸）")
print("- KNN缺失值插补")
print("- 随机森林回归建模")
print("- SHAP可解释性分析")
print("- 特征重要性排序")
print("- 相关性分析")
print("- 可视化图表生成")
print("- 业务解释总结")

print("\n生成的文件:")
print("- feature_importance_comparison.png: 特征重要性对比")
print("- target_distributions.png: 目标变量分布")
print("- correlation_heatmap.png: 相关性热图") 