#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票选股与交易系统主程序
"""

import os
import yaml
import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='stock_system.log')
logger = logging.getLogger(__name__)

# 导入自定义模块
try:
    from src.data_processing.data_loader import StockDataLoader
    from src.data_processing.feature_engineering import FeatureEngineer
    from src.strategy.stock_selector import StockSelector
    from src.trading.trading_system import TradingSystem
    logger.info("成功导入所有模块")
except Exception as e:
    logger.error(f"导入模块失败: {e}")
    raise

def load_config(config_path):
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

# 定义并行处理单只股票的辅助函数
# 注意：必须定义在main函数外面，否则Windows下ProcessPoolExecutor无法pickle序列化
def process_single_stock(args):
    feature_engineer, stock_code, df = args
    try:
        df_with_features = feature_engineer.generate_features(df)
        if df_with_features is not None and not df_with_features.empty:
            return stock_code, df_with_features
        return None
    except Exception as e:
        logger.error(f"为股票 {stock_code} 生成特征失败: {e}")
        return None

def main():
    """主程序入口"""
    logger.info("开始运行股票选股与交易系统")
    
    try:
        # 1. 加载配置
        config_path = os.path.join('config', 'config.yaml')
        config = load_config(config_path)
        
        # 2. 数据加载
        logger.info("开始加载股票数据")
        data_loader = StockDataLoader(
            base_path=config['data_path']['base_path'],
            directories=config['data_path']['directories']
        )
        
        stock_data = data_loader.load_all_stocks()
        if not stock_data:
            logger.error("没有加载到任何股票数据")
            return
        logger.info(f"成功加载 {len(stock_data)} 支股票数据")
        
        # 3. 特征工程
        logger.info("开始特征工程")
        print("\n开始特征工程...")
        feature_engineer = FeatureEngineer()
        
        # 为所有股票生成特征（并行处理）
        enhanced_stock_data = {}
        total_stocks = len(stock_data)
        processed_stocks = 0
        
        # 使用ProcessPoolExecutor进行并行处理
        max_workers = os.cpu_count()  # 使用所有可用CPU核心
        print(f"  使用 {max_workers} 个进程进行并行处理...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，传递feature_engineer、stock_code和df作为参数
            futures = {executor.submit(process_single_stock, (feature_engineer, stock_code, df)): stock_code for stock_code, df in stock_data.items()}
            
            # 处理完成的任务
            for future in as_completed(futures):
                result = future.result()
                if result:
                    stock_code, df_with_features = result
                    enhanced_stock_data[stock_code] = df_with_features
                processed_stocks += 1
                if processed_stocks % 50 == 0:
                    print(f"  已处理 {processed_stocks}/{total_stocks} 支股票")
        
        if not enhanced_stock_data:
            logger.error("特征工程失败")
            return
        logger.info(f"成功为 {len(enhanced_stock_data)} 支股票生成特征")
        
        # 打印一些内存使用信息，帮助诊断问题
        import psutil
        process = psutil.Process(os.getpid())
        logger.info(f"当前内存使用情况: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # 4. 选股策略
        logger.info("开始选股策略")
        
        # 只处理 default: true 的策略
        strategy_names = [name for name, settings in config['strategies'].items() if settings.get('default', False)]
        stock_selectors = {}
        logger.info(f"将处理的策略列表: {strategy_names}")
        
        for strategy_name in strategy_names:
            logger.info(f"正在处理 {strategy_name} 策略")
            print(f"\n正在处理 {strategy_name} 策略...")
            stock_selector = StockSelector(config, strategy_name)
            
            # 使用完整股票数据进行训练
            print(f"  使用完整股票数据进行训练，共 {len(enhanced_stock_data)} 支股票")
            
            # 准备训练数据
            print(f"  正在准备 {strategy_name} 策略的训练数据...")
            training_data = stock_selector.prepare_training_data(
                enhanced_stock_data,
                lookahead_days=config['feature_engineering']['lookahead_days']
            )
            
            if training_data.empty:
                logger.error(f"{strategy_name} 策略没有生成训练数据")
                print(f"  {strategy_name} 策略没有生成训练数据")
                continue
            logger.info(f"{strategy_name} 策略成功生成训练数据，共 {len(training_data)} 条记录")
            print(f"  {strategy_name} 策略成功生成训练数据，共 {len(training_data)} 条记录")
            
            # 训练模型
            print(f"  正在训练 {strategy_name} 策略模型...")
            success = stock_selector.train_model(training_data)
            if not success:
                logger.error(f"{strategy_name} 策略模型训练失败")
                print(f"  {strategy_name} 策略模型训练失败")
                continue
            logger.info(f"{strategy_name} 策略模型训练成功")
            print(f"  {strategy_name} 策略模型训练成功")
            
            stock_selectors[strategy_name] = stock_selector
        
        if not stock_selectors:
            logger.error("所有策略模型训练失败")
            return
        logger.info(f"成功训练了 {len(stock_selectors)} 个策略模型")
        
        # 5. 交易系统
        logger.info("开始交易系统回测")
        trading_system = TradingSystem(config)
        
        # 6. 结果输出
        logger.info("回测完成，生成回测报告")
        
        # 保存回测结果
        report_dir = 'reports'
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        # 生成并保存历史选股池数据
        logger.info("开始生成历史选股池数据")
        start_date = pd.to_datetime(config['backtest']['start_date'])
        end_date = pd.to_datetime(config['backtest']['end_date'])
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 用于存储所有策略的综合选股池数据（避免重复读取文件）
        all_strategies_pools = []
        
        # 为每个策略生成选股池数据
        for strategy_name, stock_selector in stock_selectors.items():
            logger.info(f"正在生成 {strategy_name} 策略的选股池数据")
            print(f"\n正在生成 {strategy_name} 策略的选股池数据...")
            all_daily_pools = []
            total_dates = len(date_range)
            processed_dates = 0
            for date in date_range:
                daily_pool = stock_selector.generate_daily_pool(enhanced_stock_data, date)
                if not daily_pool.empty:
                    daily_pool['日期'] = date
                    daily_pool['策略名称'] = strategy_name
                    all_daily_pools.append(daily_pool)
                processed_dates += 1
                if processed_dates % 10 == 0:
                    print(f"  已处理 {processed_dates}/{total_dates} 个交易日")
            
            if all_daily_pools:
                all_daily_pools_df = pd.concat(all_daily_pools, ignore_index=True)
                daily_pools_file = os.path.join(report_dir, f'daily_pools_{strategy_name}.csv')
                all_daily_pools_df.to_csv(daily_pools_file, index=False, encoding='utf-8')
                logger.info(f"成功保存 {strategy_name} 策略 {len(all_daily_pools)} 天的选股池数据")
                print(f"  成功保存 {strategy_name} 策略 {len(all_daily_pools)} 天的选股池数据")
                
                # 同时添加到综合数据中（避免重复读取文件）
                all_strategies_pools.append(all_daily_pools_df)
        
        # 生成包含所有策略的综合选股池数据（直接使用内存中的数据，避免重复读取文件）
        if all_strategies_pools:
            all_pools_df = pd.concat(all_strategies_pools, ignore_index=True)
            all_pools_file = os.path.join(report_dir, 'daily_pools.csv')
            all_pools_df.to_csv(all_pools_file, index=False, encoding='utf-8')
            logger.info(f"成功保存所有策略的综合选股池数据，共 {len(all_pools_df)} 条记录")
        
        print(f"\n准备执行回测，找到 {len(stock_selectors)} 个策略")
        print(f"可用的策略: {list(stock_selectors.keys())}")
        
        # 存储所有策略的回测结果，避免重复调用
        all_strategy_results = {}
        
        # 为每个策略执行回测，包括所有三种卖出策略（只调用一次！）
        for strategy_name, stock_selector in stock_selectors.items():
            logger.info(f"开始 {strategy_name} 策略的回测（包含所有卖出策略）")
            print(f"\n开始 {strategy_name} 策略的回测（包含所有卖出策略）...")
            print(f"  调用 backtest_all_sell_strategies 方法...")
            
            # 使用 TradingSystem 的 backtest_all_sell_strategies 方法回测所有卖出策略
            all_sell_strategy_results = trading_system.backtest_all_sell_strategies(
                processed_stock_data=enhanced_stock_data,
                stock_selector=stock_selector,
                start_date=start_date,
                end_date=end_date
            )
            
            # 保存结果供后续使用
            all_strategy_results[strategy_name] = all_sell_strategy_results
            
            print(f"  回测完成，获得 {len(all_sell_strategy_results)} 种卖出策略的结果")
            print(f"  卖出策略列表: {list(all_sell_strategy_results.keys())}")
            
            # 保存每种卖出策略的回测结果
            for sell_strategy_name, sell_strategy_result in all_sell_strategy_results.items():
                logger.info(f"  处理卖出策略: {sell_strategy_name}")
                print(f"  处理卖出策略: {sell_strategy_name}")
                
                result = sell_strategy_result['result']
                if not result:
                    logger.error(f"    {sell_strategy_name} 回测结果为空")
                    continue
                
                # 构造文件名
                base_name = f"{strategy_name}_{sell_strategy_name}"
                base_name = base_name.replace(' ', '_')  # 替换空格为下划线
                
                # 保存每日价值数据
                daily_values_df = pd.DataFrame(result['每日价值'])
                daily_values_file = os.path.join(report_dir, f'daily_values_{base_name}.csv')
                daily_values_df.to_csv(daily_values_file, index=False, encoding='utf-8')
                print(f"    保存每日价值到: {daily_values_file}")
                
                # 保存交易记录
                trades_df = pd.DataFrame(result['交易记录'])
                trades_file = os.path.join(report_dir, f'trades_{base_name}.csv')
                trades_df.to_csv(trades_file, index=False, encoding='utf-8')
                print(f"    保存交易记录到: {trades_file}")
                
                # 打印回测结果摘要
                logger.info(f"\n=== {strategy_name} + {sell_strategy_name} 回测结果摘要 ===")
                logger.info(f"初始资金: {result['初始资金']} 元")
                logger.info(f"最终资金: {result['最终资金']:.2f} 元")
                logger.info(f"总收益率: {result['总收益率']:.2%}")
                logger.info(f"年化收益率: {result['年化收益率']:.2%}")
                logger.info(f"夏普比率: {result['夏普比率']:.2f}")
                logger.info(f"最大回撤: {result['最大回撤']:.2%}")
                logger.info(f"交易次数: {result['交易次数']}")
                logger.info(f"胜率: {result['胜率']:.2%}")
                
                print(f"\n=== {strategy_name} + {sell_strategy_name} 回测结果摘要 ===")
                print(f"初始资金: {result['初始资金']} 元")
                print(f"最终资金: {result['最终资金']:.2f} 元")
                print(f"总收益率: {result['总收益率']:.2%}")
                print(f"交易次数: {result['交易次数']}")
        
        # 生成三种卖出策略的对比数据
        logger.info("开始生成三种卖出策略的对比数据")
        print("\n开始生成三种卖出策略的对比数据...")
        
        # 收集所有策略的所有卖出策略结果（使用已存储的结果，避免重复调用！）
        all_daily_values = []
        
        for strategy_name, all_sell_strategy_results in all_strategy_results.items():
            logger.info(f"处理 {strategy_name} 策略的卖出策略结果")
            
            # 收集所有卖出策略的每日价值数据
            for sell_strategy_name, sell_strategy_result in all_sell_strategy_results.items():
                if 'result' in sell_strategy_result and '每日价值' in sell_strategy_result['result']:
                    daily_values_df = pd.DataFrame(sell_strategy_result['result']['每日价值'])
                    daily_values_df['策略名称'] = f"{strategy_name}_{sell_strategy_name}"
                    daily_values_df['累计收益率'] = (daily_values_df['总价值'] / 1000000 - 1) * 100
                    all_daily_values.append(daily_values_df)
        
        # 合并所有策略的每日价值数据
        if all_daily_values:
            combined_daily_values = pd.concat(all_daily_values, ignore_index=True)
            strategy_comparison_daily_values_file = os.path.join(report_dir, 'strategy_comparison_daily_values.csv')
            combined_daily_values.to_csv(strategy_comparison_daily_values_file, index=False, encoding='utf-8')
            logger.info(f"成功保存策略对比每日价值数据到: {strategy_comparison_daily_values_file}")
            print(f"  已保存策略对比每日价值数据到: {strategy_comparison_daily_values_file}")
        
        # 生成策略性能指标对比数据
        metrics_comparison = []
        
        for strategy_name, all_sell_strategy_results in all_strategy_results.items():
            logger.info(f"处理 {strategy_name} 策略的性能指标")
            
            # 计算每个卖出策略的性能指标（使用已存储的结果，避免重复调用！）
            for sell_strategy_name, sell_strategy_result in all_sell_strategy_results.items():
                if 'result' in sell_strategy_result and '每日价值' in sell_strategy_result['result']:
                    daily_values_df = pd.DataFrame(sell_strategy_result['result']['每日价值'])
                    
                    final_value = daily_values_df['总价值'].iloc[-1]
                    total_return = (final_value / 1000000 - 1) * 100
                    
                    # 计算年化收益率
                    days = len(daily_values_df)
                    if days > 0:
                        annualized_return = ((final_value / 1000000) ** (365 / days) - 1) * 100
                    else:
                        annualized_return = 0
                    
                    # 计算最大回撤
                    daily_values_df['峰值'] = daily_values_df['总价值'].cummax()
                    daily_values_df['回撤'] = (daily_values_df['总价值'] - daily_values_df['峰值']) / daily_values_df['峰值'] * 100
                    max_drawdown = daily_values_df['回撤'].min()
                    
                    metrics_comparison.append({
                        '策略名称': f"{strategy_name}_{sell_strategy_name}",
                        '总收益率(%)': round(total_return, 2),
                        '年化收益率(%)': round(annualized_return, 2),
                        '最大回撤(%)': round(max_drawdown, 2)
                    })
        
        # 保存性能指标对比数据
        if metrics_comparison:
            metrics_comparison_df = pd.DataFrame(metrics_comparison)
            strategy_comparison_metrics_file = os.path.join(report_dir, 'strategy_comparison_metrics.csv')
            metrics_comparison_df.to_csv(strategy_comparison_metrics_file, index=False, encoding='utf-8')
            logger.info(f"成功保存策略对比性能指标到: {strategy_comparison_metrics_file}")
            print(f"  已保存策略对比性能指标到: {strategy_comparison_metrics_file}")
        
        logger.info("\n股票选股与交易系统运行完成")
        print("\n股票选股与交易系统运行完成")
        
    except Exception as e:
        logger.error(f"系统运行失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()
