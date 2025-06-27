import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import pearsonr
import shap
import warnings
warnings.filterwarnings('ignore')

# 尝试导入geopandas，如果没有安装则跳过地理分析
try:
    import geopandas as gpd
    from shapely.geometry import Point
    import folium
    from folium import plugins
    GEOPANDAS_AVAILABLE = True
    print("✅ Geopandas可用，将进行地理空间分析")
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠️ Geopandas不可用，跳过地理空间分析")

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
    
    # 5. 改进的完整散点图 (Enhanced Scatter Plot) - 使用所有特征，带统计指标
    print(f"生成完整统计散点图...")
    
    # 计算所有特征的统计指标
    stats_data = []
    for i, feature in enumerate(feature_columns):
        feature_values = X_sample[feature].values
        shap_feature_values = shap_values[:, i]
        
        # 计算相关系数和p值
        corr_coef, p_value = pearsonr(feature_values, shap_feature_values)
        
        # 计算R²
        r_squared = corr_coef ** 2
        
        # 计算均值和标准差
        shap_mean = np.mean(shap_feature_values)
        shap_std = np.std(shap_feature_values)
        
        stats_data.append({
            'feature': feature,
            'correlation': corr_coef,
            'p_value': p_value,
            'r_squared': r_squared,
            'shap_mean': shap_mean,
            'shap_std': shap_std,
            'shap_importance': np.abs(shap_feature_values).mean()
        })
    
    stats_df = pd.DataFrame(stats_data).sort_values('shap_importance', ascending=False)
    
    # 创建完整的散点图矩阵 - 所有16个特征
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle(f'{target_name} - Complete SHAP Scatter Plots with Statistics (All Features)', fontsize=18)
    
    for idx, feature in enumerate(feature_columns):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        # 获取特征值和SHAP值
        feature_values = X_sample[feature].values
        shap_feature_values = shap_values[:, idx]
        
        # 获取统计数据
        feature_stats = stats_df[stats_df['feature'] == feature].iloc[0]
        
        # 创建散点图
        scatter = ax.scatter(feature_values, shap_feature_values, 
                           alpha=0.6, c=feature_values, cmap='viridis', s=15)
        
        # 添加趋势线
        if len(np.unique(feature_values)) > 1:
            z = np.polyfit(feature_values, shap_feature_values, 1)
            p = np.poly1d(z)
            ax.plot(feature_values, p(feature_values), "r--", alpha=0.8, linewidth=2)
        
        # 设置标签和标题
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('SHAP Value', fontsize=9)
        
        # 添加统计信息到标题
        title = f'{feature}\np={feature_stats["p_value"]:.2e}'
        if feature_stats["p_value"] < 0.001:
            title += " ***"
        elif feature_stats["p_value"] < 0.01:
            title += " **"
        elif feature_stats["p_value"] < 0.05:
            title += " *"
        
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(f'shap_scatter_complete_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"保存: shap_scatter_complete_{target_name}.png")
    plt.close()
    
    # 保存统计数据表
    stats_df.to_csv(f'shap_statistics_{target_name}.csv', index=False)
    print(f"保存统计数据: shap_statistics_{target_name}.csv")
    
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
    
    # 8. 标准SHAP瀑布图 (Standard Waterfall Plot) - 前3个样本
    print(f"生成标准瀑布图...")
    for i in range(min(3, len(X_sample))):
        # 使用SHAP标准瀑布图
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap.Explanation(values=shap_values[i], 
                                            base_values=explainer.expected_value, 
                                            data=X_sample.iloc[i].values,
                                            feature_names=feature_columns), 
                           max_display=15, show=False)
        plt.title(f'{target_name} - Standard SHAP Waterfall Plot (Sample {i+1})')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_standard_{target_name}_sample_{i+1}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"保存: shap_waterfall_standard_{target_name}_sample_{i+1}.png")
        plt.close()
    
    # 9. 标准SHAP力图 (Standard Force Plot)
    print(f"生成标准力图...")
    for i in range(min(3, len(X_sample))):
        # 创建标准SHAP force plot
        force_plot = shap.force_plot(explainer.expected_value, shap_values[i], 
                                   X_sample.iloc[i], feature_names=feature_columns,
                                   matplotlib=True, show=False)
        
        plt.title(f'{target_name} - Standard SHAP Force Plot (Sample {i+1})')
        plt.tight_layout()
        plt.savefig(f'shap_force_standard_{target_name}_sample_{i+1}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"保存: shap_force_standard_{target_name}_sample_{i+1}.png")
        plt.close()

# 10. 创建综合SHAP对比图
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

# 11. 特征重要性对比图
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

# 12. 双目标SHAP值分布对比
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
csv_files = [f for f in os.listdir('.') if f.startswith('shap_') and f.endswith('.csv')]
shap_files.sort()
csv_files.sort()

print("\nPNG图表文件:")
for file in shap_files:
    print(f"- {file}")

print("\nCSV统计文件:")
for file in csv_files:
    print(f"- {file}")

# 地理空间SHAP分析
if GEOPANDAS_AVAILABLE:
    print("\n========== 地理空间SHAP分析 ==========")
    
    # 创建地理数据框
    geo_data = df_processed.copy()
    
    # 根据zipCode和State创建地理特征
    # 这里我们使用模拟的经纬度，实际项目中应该使用真实的地理编码数据
    np.random.seed(42)  # 确保可重现性
    
    # 为美国各州创建模拟坐标范围
    state_coords = {
        'CA': {'lat_range': [32.5, 42.0], 'lon_range': [-124.5, -114.0]},
        'TX': {'lat_range': [25.8, 36.5], 'lon_range': [-106.6, -93.5]},
        'FL': {'lat_range': [24.5, 31.0], 'lon_range': [-87.6, -80.0]},
        'NY': {'lat_range': [40.5, 45.0], 'lon_range': [-79.8, -71.8]},
        'PA': {'lat_range': [39.7, 42.3], 'lon_range': [-80.5, -74.7]},
        'IL': {'lat_range': [36.9, 42.5], 'lon_range': [-91.5, -87.0]},
        'OH': {'lat_range': [38.4, 41.9], 'lon_range': [-84.8, -80.5]},
        'GA': {'lat_range': [30.4, 35.0], 'lon_range': [-85.6, -80.8]},
        'NC': {'lat_range': [33.8, 36.6], 'lon_range': [-84.3, -75.5]},
        'MI': {'lat_range': [41.7, 48.2], 'lon_range': [-90.4, -82.4]}
    }
    
    # 生成模拟的经纬度
    lats, lons = [], []
    for idx, row in geo_data.iterrows():
        state_code = int(row['State'])  # 确保是整数索引
        if state_code < len(encoders_info['State']['encoder'].classes_):
            state = list(encoders_info['State']['encoder'].classes_)[state_code]
            if state in state_coords:
                lat_range = state_coords[state]['lat_range']
                lon_range = state_coords[state]['lon_range']
                lat = np.random.uniform(lat_range[0], lat_range[1])
                lon = np.random.uniform(lon_range[0], lon_range[1])
            else:
                # 默认美国中心位置
                lat = np.random.uniform(39.0, 40.0)
                lon = np.random.uniform(-98.0, -97.0)
        else:
            # 默认美国中心位置
            lat = np.random.uniform(39.0, 40.0)
            lon = np.random.uniform(-98.0, -97.0)
        lats.append(lat)
        lons.append(lon)
    
    geo_data['latitude'] = lats
    geo_data['longitude'] = lons
    
    # 创建地理点
    geometry = [Point(xy) for xy in zip(geo_data['longitude'], geo_data['latitude'])]
    geo_df = gpd.GeoDataFrame(geo_data, geometry=geometry)
    
    print("地理数据框创建完成")
    
    # 重建模型获取SHAP值
    for target_name in target_columns:
        print(f"\n生成{target_name}的地理SHAP分析...")
        
        y = geo_data[target_name]
        X_geo = geo_data[feature_columns]
        
        X_train, X_test, y_train, y_test = train_test_split(X_geo, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(rf_model)
        sample_size = min(500, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        
        # 计算每个样本的SHAP值总和
        shap_sums = np.sum(np.abs(shap_values), axis=1)
        
        # 创建地理SHAP数据
        geo_sample = geo_df.iloc[X_test.index[:sample_size]].copy()
        geo_sample['shap_total'] = shap_sums
        geo_sample['predicted_value'] = rf_model.predict(X_sample)
        geo_sample['actual_value'] = y_test.iloc[:sample_size].values
        
        # 1. 地理SHAP热力图
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 创建散点图，颜色表示SHAP值总和
        scatter = ax.scatter(geo_sample['longitude'], geo_sample['latitude'], 
                           c=geo_sample['shap_total'], cmap='viridis', 
                           s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'{target_name} - Geographic SHAP Importance Heat Map', fontsize=14)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Total SHAP Importance', fontsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'shap_geographic_heatmap_{target_name}.png', dpi=300, bbox_inches='tight')
        print(f"保存: shap_geographic_heatmap_{target_name}.png")
        plt.close()
        
        # 2. 预测值vs实际值地理分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 实际值分布
        scatter1 = ax1.scatter(geo_sample['longitude'], geo_sample['latitude'], 
                             c=geo_sample['actual_value'], cmap='Reds', 
                             s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.set_title(f'{target_name} - Actual Values Geographic Distribution', fontsize=12)
        plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        ax1.grid(True, alpha=0.3)
        
        # 预测值分布
        scatter2 = ax2.scatter(geo_sample['longitude'], geo_sample['latitude'], 
                             c=geo_sample['predicted_value'], cmap='Blues', 
                             s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Latitude', fontsize=12)
        ax2.set_title(f'{target_name} - Predicted Values Geographic Distribution', fontsize=12)
        plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'shap_geographic_comparison_{target_name}.png', dpi=300, bbox_inches='tight')
        print(f"保存: shap_geographic_comparison_{target_name}.png")
        plt.close()
        
        # 3. 特征重要性地理分布 - 选择前3个最重要特征
        shap_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        top_3_features = shap_importance.head(3)['feature'].tolist()
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        for idx, feature in enumerate(top_3_features):
            feature_idx = feature_columns.index(feature)
            feature_shap = shap_values[:, feature_idx]
            
            scatter = axes[idx].scatter(geo_sample['longitude'], geo_sample['latitude'], 
                                      c=feature_shap, cmap='RdBu_r', 
                                      s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
            axes[idx].set_xlabel('Longitude', fontsize=10)
            axes[idx].set_ylabel('Latitude', fontsize=10)
            axes[idx].set_title(f'{target_name} - {feature}\nSHAP Values Geographic Distribution', fontsize=10)
            plt.colorbar(scatter, ax=axes[idx], shrink=0.8)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'shap_geographic_features_{target_name}.png', dpi=300, bbox_inches='tight')
        print(f"保存: shap_geographic_features_{target_name}.png")
        plt.close()
else:
    print("跳过地理空间分析（需要安装geopandas）")

print(f"\n总共生成了SHAP可视化图表和统计文件！")

print("\n改进内容:")
print("1. ✅ 删除了所有旧图表")
print("2. ✅ Scatter Plot: 使用全部16个特征，仅保留P-value显著性标记")
print("3. ✅ Waterfall Plot: 使用SHAP标准瀑布图格式")
print("4. ✅ Force Plot: 使用SHAP标准力图格式")
print("5. ✅ 生成CSV文件保存所有统计数据")
print("6. ✅ 添加地理空间SHAP分析（如果geopandas可用）")
print("7. ✅ 所有图表都达到300 DPI高清标准") 