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

# 设置字体支持，确保正负号正确显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['font.family'] = 'sans-serif'

print("正在加载数据和重建模型...")
df = pd.read_excel('cancerdeaths.xlsx')

# 数据预处理 (复制之前的预处理步骤)
print("数据预处理...")
illegal_chars = ["*", "**", "***", "?", "？", "??", "？？"]

for column in df.columns:
    if df[column].dtype == 'object':
        for char in illegal_chars:
            mask = df[column].astype(str).str.contains(char, regex=False, na=False)
            if mask.any():
                df.loc[mask, column] = np.nan

numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

df_processed = df.copy()
encoders_info = {}

# 标签编码
for col in categorical_columns:
    if df_processed[col].isnull().any():
        mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "unknown"
        df_processed[col].fillna(mode_value, inplace=True)
    
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    encoders_info[col] = {
        'encoder': le,
        'classes': len(le.classes_),
        'categories': list(le.classes_)
    }

# KNN插补
numerical_missing = df_processed[numerical_columns].isnull().sum().sum()
if numerical_missing > 0:
    knn_imputer = KNNImputer(n_neighbors=5)
    df_processed[numerical_columns] = knn_imputer.fit_transform(df_processed[numerical_columns])

# 准备建模数据
target_columns = ['incidenceRate', 'deathRate']
feature_columns = [col for col in df_processed.columns if col not in target_columns]
X = df_processed[feature_columns]

print("重建模型和计算SHAP值...")

# 为每个目标变量生成完整的SHAP可视化
for target_name in target_columns:
    print(f"\n========== {target_name} SHAP完整可视化 ==========")
    
    y = df_processed[target_name]
    
    # 训练测试集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 随机森林模型
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(rf_model)
    
    # 使用较大样本计算SHAP值
    sample_size = min(1000, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    print(f"SHAP值计算完成，样本数: {sample_size}")
    
    # 1. SHAP摘要图 (Summary Plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_columns, show=False)
    plt.title(f'{target_name} - SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(f'shap_summary_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"保存: shap_summary_{target_name}.png")
    plt.close()
    
    # 2. SHAP条形图 (Bar Plot) - 特征重要性
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_columns, plot_type="bar", show=False)
    plt.title(f'{target_name} - SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(f'shap_bar_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"保存: shap_bar_{target_name}.png")
    plt.close()
    
    # 3. SHAP蜂群图 (Beeswarm Plot)
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap.Explanation(values=shap_values, 
                                        data=X_sample.values, 
                                        feature_names=feature_columns), show=False)
    plt.title(f'{target_name} - SHAP Beeswarm Plot')
    plt.tight_layout()
    plt.savefig(f'shap_beeswarm_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"保存: shap_beeswarm_{target_name}.png")
    plt.close()
    
    # 4. SHAP依赖图 (Dependence Plot) - 前5个最重要特征
    shap_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    top_features = shap_importance.head(5)['feature'].tolist()
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_sample, 
                           feature_names=feature_columns, show=False)
        plt.title(f'{target_name} - SHAP Dependence Plot: {feature}')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{target_name}_{feature}.png', dpi=300, bbox_inches='tight')
        print(f"保存: shap_dependence_{target_name}_{feature}.png")
        plt.close()
    
    # 5. SHAP散点图 (Scatter Plot) - 前5个最重要特征
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{target_name} - SHAP Scatter Plots', fontsize=16)
    
    for idx, feature in enumerate(top_features):
        row = idx // 3
        col = idx % 3
        
        if row < 2 and col < 3:
            ax = axes[row, col]
            
            # 获取特征值和SHAP值
            feature_values = X_sample[feature].values
            shap_feature_values = shap_values[:, feature_columns.index(feature)]
            
            # 创建散点图
            scatter = ax.scatter(feature_values, shap_feature_values, 
                               alpha=0.6, c=feature_values, cmap='viridis')
            ax.set_xlabel(feature)
            ax.set_ylabel('SHAP Value')
            ax.set_title(f'{feature}')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax)
    
    # 隐藏空的子图
    if len(top_features) < 6:
        for idx in range(len(top_features), 6):
            row = idx // 3
            col = idx % 3
            if row < 2 and col < 3:
                axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'shap_scatter_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"保存: shap_scatter_{target_name}.png")
    plt.close()
    
    # 6. SHAP交互作用图 (Interaction Plot) - 前3对特征
    print(f"生成交互作用图...")
    top_3_features = top_features[:3]
    
    for i in range(len(top_3_features)):
        for j in range(i+1, len(top_3_features)):
            feature1, feature2 = top_3_features[i], top_3_features[j]
            
            plt.figure(figsize=(10, 8))
            shap.dependence_plot(feature1, shap_values, X_sample, 
                               interaction_index=feature2,
                               feature_names=feature_columns, show=False)
            plt.title(f'{target_name} - SHAP Interaction: {feature1} vs {feature2}')
            plt.tight_layout()
            plt.savefig(f'shap_interaction_{target_name}_{feature1}_vs_{feature2}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"保存: shap_interaction_{target_name}_{feature1}_vs_{feature2}.png")
            plt.close()
    
    # 7. SHAP决策图 (Decision Plot) - 随机100个样本
    plt.figure(figsize=(12, 10))
    np.random.seed(42)  # 设置随机种子确保可重现
    sample_indices = np.random.choice(len(X_sample), size=min(100, len(X_sample)), replace=False)
    shap.decision_plot(explainer.expected_value, shap_values[sample_indices], 
                      X_sample.iloc[sample_indices], feature_names=feature_columns, show=False)
    plt.title(f'{target_name} - SHAP Decision Plot (Random 100 Samples)')
    plt.tight_layout()
    plt.savefig(f'shap_decision_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"保存: shap_decision_{target_name}.png")
    plt.close()
    
    # 8. SHAP力图 (Force Plot) - 转换为静态图
    # 选择一个样本创建力图
    sample_idx = 0
    plt.figure(figsize=(14, 4))
    
    # 手动创建力图数据
    base_value = explainer.expected_value
    shap_vals = shap_values[sample_idx]
    feature_vals = X_sample.iloc[sample_idx]
    
    # 创建简化的力图
    pos_values = shap_vals[shap_vals > 0]
    neg_values = shap_vals[shap_vals < 0]
    pos_features = [feature_columns[i] for i, val in enumerate(shap_vals) if val > 0]
    neg_features = [feature_columns[i] for i, val in enumerate(shap_vals) if val < 0]
    
    # 绘制条形图表示力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 正向贡献
    if len(pos_values) > 0:
        top_pos_indices = np.argsort(pos_values)[-10:]  # 取前10个
        ax1.barh(range(len(top_pos_indices)), pos_values[top_pos_indices], color='red', alpha=0.7)
        ax1.set_yticks(range(len(top_pos_indices)))
        ax1.set_yticklabels([pos_features[i] for i in top_pos_indices])
        ax1.set_title(f'{target_name} - Positive SHAP Contributions (Sample {sample_idx+1})')
        ax1.set_xlabel('SHAP Value')
    
    # 负向贡献
    if len(neg_values) > 0:
        top_neg_indices = np.argsort(np.abs(neg_values))[-10:]  # 取前10个
        ax2.barh(range(len(top_neg_indices)), neg_values[top_neg_indices], color='blue', alpha=0.7)
        ax2.set_yticks(range(len(top_neg_indices)))
        ax2.set_yticklabels([neg_features[i] for i in top_neg_indices])
        ax2.set_title(f'{target_name} - Negative SHAP Contributions (Sample {sample_idx+1})')
        ax2.set_xlabel('SHAP Value')
    
    plt.tight_layout()
    plt.savefig(f'shap_force_{target_name}_sample_{sample_idx+1}.png', dpi=300, bbox_inches='tight')
    print(f"保存: shap_force_{target_name}_sample_{sample_idx+1}.png")
    plt.close()

# 9. 创建综合SHAP对比图
print("\n========== 创建综合SHAP对比分析 ==========")

# 重新计算两个目标变量的SHAP值用于对比
results_comparison = {}

for target_name in target_columns:
    y = df_processed[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(rf_model)
    sample_size = min(800, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    # 计算特征重要性
    shap_importance = pd.DataFrame({
        'feature': feature_columns,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    results_comparison[target_name] = {
        'shap_values': shap_values,
        'X_sample': X_sample,
        'shap_importance': shap_importance
    }

# 10. 特征重要性对比图
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for i, target_name in enumerate(target_columns):
    # 获取前15个最重要特征
    top_features = results_comparison[target_name]['shap_importance'].head(15)
    
    # 创建条形图
    axes[i].barh(range(len(top_features)), top_features['shap_importance'])
    axes[i].set_yticks(range(len(top_features)))
    axes[i].set_yticklabels(top_features['feature'])
    axes[i].set_xlabel('SHAP Importance')
    axes[i].set_title(f'{target_name} - Top 15 Features')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shap_comparison_importance.png', dpi=300, bbox_inches='tight')
print("保存: shap_comparison_importance.png")
plt.close()

# 11. 双目标SHAP值分布对比
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SHAP Values Distribution Comparison', fontsize=16)

for i, target_name in enumerate(target_columns):
    shap_values = results_comparison[target_name]['shap_values']
    
    # SHAP值分布直方图
    axes[i, 0].hist(shap_values.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[i, 0].set_title(f'{target_name} - SHAP Values Distribution')
    axes[i, 0].set_xlabel('SHAP Value')
    axes[i, 0].set_ylabel('Frequency')
    axes[i, 0].grid(True, alpha=0.3)
    
    # 特征重要性累积图
    importance = results_comparison[target_name]['shap_importance'].head(10)
    cumulative = importance['shap_importance'].cumsum() / importance['shap_importance'].sum()
    
    axes[i, 1].plot(range(1, len(cumulative)+1), cumulative, marker='o')
    axes[i, 1].set_title(f'{target_name} - Cumulative SHAP Importance')
    axes[i, 1].set_xlabel('Top N Features')
    axes[i, 1].set_ylabel('Cumulative Importance Ratio')
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].set_xticks(range(1, len(cumulative)+1))
    axes[i, 1].set_xticklabels([f'Top{i+1}' for i in range(len(cumulative))], rotation=45)

plt.tight_layout()
plt.savefig('shap_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("保存: shap_distribution_comparison.png")
plt.close()

print("\n========== SHAP可视化完成 ==========")
print("生成的SHAP图表文件:")

# 统计生成的文件
import os
shap_files = [f for f in os.listdir('.') if f.startswith('shap_') and f.endswith('.png')]
shap_files.sort()

for file in shap_files:
    print(f"- {file}")

print(f"\n总共生成了 {len(shap_files)} 个SHAP可视化图表！")

print("\n图表类型说明:")
print("1. Summary Plot: 显示所有特征对所有样本的SHAP值分布")
print("2. Bar Plot: 特征重要性条形图")
print("3. Beeswarm Plot: 蜂群图显示SHAP值分布")
print("4. Dependence Plot: 单个特征的SHAP值与特征值关系")
print("5. Scatter Plot: 散点图显示特征值与SHAP值关系")
print("6. Interaction Plot: 特征交互作用图")
print("7. Decision Plot: 决策图显示预测路径")
print("8. Force Plot: 力图显示单个样本的SHAP贡献")
print("9. Comparison Plot: 双目标对比图")
print("10. Distribution Plot: SHAP值分布对比图") 