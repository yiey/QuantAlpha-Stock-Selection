import pandas as pd
from pathlib import Path
from stock_selection_system import DataLoader, FeatureEngineer, FeatureSelector, StockModel, ReportGenerator
import yaml
import warnings
warnings.filterwarnings('ignore')

def load_config():
    config_file = Path(__file__).parent / 'config.yaml'
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

data_dir = Path(__file__).parent
start_date = config.get('start_date', '20250501')
ic_threshold = config.get('ic_threshold', 0.01)
ir_threshold = config.get('ir_threshold', 0.3)
model_type = config.get('model_type', 'random_forest')
lookahead_days = config.get('lookahead_days', 5)
top_n = config.get('top_n', 20)

print("=" * 70)
print("股票选股系统 - 每日选股池版")
print("=" * 70)

print("\n【系统配置】")
print(f"  数据目录: {data_dir}")
print(f"  起始日期: 2025-05-01")
print(f"  预测天数: 5天")
print(f"  IC阈值: 0.01")
print(f"  IR阈值: 0.3")
print(f"  模型类型: RandomForest")

print("\n" + "=" * 70)
print("【步骤 1/6】加载数据")
print("=" * 70)

loader = DataLoader(data_dir, start_date='20250501')
loader.load_all_stocks()

all_data = loader.get_all_data()
print(f"✓ 成功加载 {len(loader.stock_data)} 只股票")
print(f"✓ 总记录数: {len(all_data):,}")
print(f"✓ 时间范围: {all_data['交易日期'].min().date()} 至 {all_data['交易日期'].max().date()}")

print("\n" + "=" * 70)
print("【步骤 2/6】构建特征")
print("=" * 70)

engineer = FeatureEngineer()
feature_data = engineer.create_features(all_data)
print(f"✓ 特征数据形状: {feature_data.shape}")

feature_cols = engineer.get_feature_columns(feature_data)
print(f"✓ 特征总数: {len(feature_cols)}")

print("\n" + "=" * 70)
print("【步骤 3/6】准备目标变量")
print("=" * 70)

model = StockModel()
feature_data_with_target = model.prepare_target(feature_data, forward_days=5)
feature_data_for_training = feature_data_with_target.dropna(subset=['future_return'])

print(f"✓ 目标变量: 未来5日收益率")
print(f"✓ 训练样本数: {len(feature_data_for_training):,}")
print(f"✓ 预测样本数: {len(feature_data_with_target):,}")

print("\n" + "=" * 70)
print("【步骤 4/6】特征选择 (IC/IR筛选)")
print("=" * 70)

selector = FeatureSelector(ic_threshold=0.01, ir_threshold=0.3)
selected_features = selector.select_features(feature_data_for_training, feature_cols, 'future_return')

print(f"✓ 筛选标准:")
print(f"  - IC均值绝对值 >= 0.01")
print(f"  - IR绝对值 >= 0.3")
print(f"✓ 选中特征数: {len(selected_features)}")
print(f"✓ 选中特征: {', '.join(selected_features)}")

print("\n" + "=" * 70)
print("【步骤 5/6】训练模型")
print("=" * 70)

train_data = feature_data_for_training[feature_data_for_training['交易日期'] < feature_data_for_training['交易日期'].max()]
X_train = train_data[selected_features]
y_train = train_data['future_return']

print(f"✓ 训练集大小: {X_train.shape[0]:,} 样本")
print(f"✓ 特征维度: {X_train.shape[1]}")

model.model_type = 'random_forest'
model.train(X_train, y_train)

print(f"✓ 模型训练完成")

print("\n" + "=" * 70)
print("【步骤 6/6】生成每日选股池")
print("=" * 70)

print(f"正在为每个交易日生成选股池...")

unique_dates = sorted(feature_data_with_target['交易日期'].unique())
selection_pools = []

for i, date in enumerate(unique_dates):
    date_data = feature_data_with_target[feature_data_with_target['交易日期'] == date].copy()
    
    if len(date_data) < 10:
        continue
    
    X_date = date_data[selected_features]
    predictions = model.predict(X_date)
    
    date_data['predicted_return'] = predictions
    top_stocks = date_data.nlargest(20, 'predicted_return')
    top_stocks['rank'] = range(1, len(top_stocks) + 1)
    top_stocks['交易日期'] = date
    
    selection_pools.append(top_stocks[['rank', '交易日期', '股票代码', '名称', '收盘价', 
                                        'predicted_return', '市盈率', '市净率', 
                                        '换手率(%)', '总市值(万元)']])
    
    if (i + 1) % 10 == 0:
        print(f"  已处理 {i+1}/{len(unique_dates)} 个交易日")

if selection_pools:
    daily_pool = pd.concat(selection_pools, ignore_index=True)
    daily_pool.to_csv('reports/daily_selection_pool.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 每日选股池已生成")
    print(f"✓ 总记录数: {len(daily_pool):,}")
    print(f"✓ 交易日期范围: {daily_pool['交易日期'].min().date()} 至 {daily_pool['交易日期'].max().date()}")
    print(f"✓ 交易日数量: {len(daily_pool['交易日期'].unique())}")

report_gen = ReportGenerator()

ic_report = selector.get_ic_report()
importance_report = model.get_feature_importance()

report_gen.generate_ic_report(ic_report, 'reports/ic_report.csv')
report_gen.generate_feature_importance_report(importance_report, 'reports/feature_importance_report.csv')

print(f"✓ IC报告: reports/ic_report.csv")
print(f"✓ 特征重要性报告: reports/feature_importance_report.csv")
print(f"✓ 每日选股池: reports/daily_selection_pool.csv")

print("\n" + "=" * 70)
print("【分析结果】")
print("=" * 70)

print("\n【Top 10 特征 - 按IC均值】")
print("-" * 70)
print(f"{'特征名称':<25} {'IC均值':<12} {'IR':<10} {'样本数'}")
print("-" * 70)
for idx, (feature, row) in enumerate(ic_report.head(10).iterrows(), 1):
    print(f"{idx:2d}. {feature:<23} {row['ic_mean_abs']:>10.4f}   {row['ir']:>8.4f}   {int(row['count']):>6}")

print("\n【Top 10 特征 - 按模型重要性】")
print("-" * 70)
print(f"{'特征名称':<25} {'重要性得分':<15} {'重要性占比'}")
print("-" * 70)
importance_report_csv = pd.read_csv('reports/feature_importance_report.csv')
for idx, row in importance_report_csv.head(10).iterrows():
    print(f"{idx+1:2d}. {row['feature']:<23} {row['importance']:>12.6f}   {row['importance_pct']:>8.2f}%")

print("\n【每日选股池统计】")
print("-" * 70)
print(f"{'交易日期':<15} {'股票数量':<10} {'平均预测收益率':<15}")
print("-" * 70)

date_stats = daily_pool.groupby('交易日期').agg({
    '股票代码': 'count',
    'predicted_return': 'mean'
}).rename(columns={'股票代码': '股票数量', 'predicted_return': '平均预测收益率'})

for date, row in date_stats.head(20).iterrows():
    print(f"{date.date()}        {int(row['股票数量']):>8}       {row['平均预测收益率']:>14.4f}")

print("\n【最新选股池 (Top 20) - " + str(daily_pool['交易日期'].max().date()) + "】")
print("-" * 70)
print(f"{'排名':<6} {'股票代码':<10} {'名称':<12} {'收盘价':<10} {'预测收益率':<12} {'市盈率':<10}")
print("-" * 70)
latest_pool = daily_pool[daily_pool['交易日期'] == daily_pool['交易日期'].max()]
for idx, row in latest_pool.iterrows():
    print(f"{int(row['rank']):<6} {row['股票代码']:<10} {row['名称']:<12} {row['收盘价']:>8.2f}   {row['predicted_return']:>10.4f}   {row['市盈率']:>8.2f}")

print("\n" + "=" * 70)
print("【系统运行完成】")
print("=" * 70)
print(f"所有报告已保存至: reports/")
print(f"✓ 系统成功运行！")
print("=" * 70)
