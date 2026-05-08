#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
选股策略模块
负责构建选股模型和生成每日选股池
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class StockSelector:
    """股票选择器"""
    
    def __init__(self, config, strategy_name=None):
        """初始化
        
        Args:
            config: 配置字典
            strategy_name: 策略名称（仅支持mixed）
        """
        self.config = config
        
        # 使用传入的策略名称或默认启用的策略
        if strategy_name:
            self.strategy_name = strategy_name
        else:
            # 如果没有指定策略，使用配置中默认启用的第一个策略
            enabled_strategies = [name for name, settings in self.config['strategies'].items() \
                                 if settings.get('default', False)]
            if enabled_strategies:
                self.strategy_name = enabled_strategies[0]
            else:
                self.strategy_name = 'mixed'  # 如果没有默认启用的策略，默认使用'mixed'
        
        self.strategy_config = config['strategies'][self.strategy_name]
        
        # 加载策略参数
        self.features = self.strategy_config['features']
        self.window_size = self.strategy_config['window_size']
        self.top_n = self.strategy_config['top_n']
        self.score_threshold = self.strategy_config['score_threshold']
        self.model = None
        self.strategy_models = {}  # 存储不同策略的模型
    
    def prepare_training_data(self, processed_stock_data, lookahead_days=5):
        """准备训练数据
        
        Args:
            processed_stock_data: 处理后的股票数据字典
            lookahead_days: 预测未来几天的涨跌幅
            
        Returns:
            pd.DataFrame: 训练数据
        """
        training_data = []
        print(f"  开始处理 {len(processed_stock_data)} 支股票的数据...")
        
        # 使用 tqdm 显示进度
        for i, (stock_code, df) in enumerate(tqdm(processed_stock_data.items(), desc="处理股票数据")):
            if df.empty or len(df) < self.window_size + lookahead_days:
                continue
            
            # 计算未来收益
            df['未来收益'] = df['收盘价'].pct_change(periods=lookahead_days, fill_method=None).shift(-lookahead_days)
            
            # 标记是否为上涨股票
            df['上涨'] = (df['未来收益'] > 0).astype(int)
            
            # 保留需要的列
            df = df[['交易日期', '上涨'] + self.features].dropna()
            
            if not df.empty:
                training_data.append(df)
            
            # 每处理100支股票打印一次进度
            if (i + 1) % 100 == 0:
                print(f"    已处理 {i + 1}/{len(processed_stock_data)} 支股票")
        
        print(f"  所有股票处理完成，共收集到 {len(training_data)} 个数据块")
        
        if training_data:
            final_data = pd.concat(training_data, ignore_index=True)
            print(f"  合并完成，训练数据总量: {len(final_data)} 条")
            return final_data
        else:
            print(f"  没有收集到任何训练数据")
            return pd.DataFrame()
    
    def train_model(self, training_data):
        """训练选股模型
        
        Args:
            training_data: 训练数据
            
        Returns:
            bool: 训练是否成功
        """
        if training_data.empty:
            logger.error("训练数据为空，无法训练模型")
            return False
        
        try:
            # 限制训练数据量以提高效率
            if len(training_data) > 100000:
                training_data = training_data.sample(100000, random_state=42)
                logger.info(f"训练数据量过大，已随机抽样至100000条记录")
            
            # 分离特征和标签
            X = training_data[self.features]
            y = training_data['上涨']
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练随机森林模型
            self.model = RandomForestClassifier(
                n_estimators=50,  # 减少树的数量以提高速度
                max_depth=8,      # 限制树深度以防止过拟合
                min_samples_split=5,  # 增加最小分裂样本数
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            logger.info(f"模型训练完成 - 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")
            
            # 获取特征重要性
            self.feature_importance = pd.DataFrame({
                '特征': self.features,
                '重要性': self.model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            logger.info("特征重要性:")
            for _, row in self.feature_importance.iterrows():
                logger.info(f"{row['特征']}: {row['重要性']:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"模型训练失败: {e}", exc_info=True)
            return False
    
    def generate_daily_pool(self, processed_stock_data, date, show_progress=False):
        """生成指定日期的选股池
        
        Args:
            processed_stock_data: 处理后的股票数据字典
            date: 指定日期
            show_progress: 是否显示进度信息
            
        Returns:
            pd.DataFrame: 选股池
        """
        daily_stocks = []
        total_stocks = len(processed_stock_data)
        
        for idx, (stock_code, df) in enumerate(processed_stock_data.items(), 1):
            if df.empty:
                continue
            
            # 获取指定日期的数据
            date_data = df[df['交易日期'] == date]
            if date_data.empty:
                continue
            
            # 检查是否有足够的特征数据
            if not all(col in date_data.columns for col in self.features):
                continue
            
            # 获取特征值（保留DataFrame格式以保留特征名称）
            feature_df = date_data[self.features].iloc[[0]]
            
            # 计算模型得分
            if self.model is None:
                # 如果没有模型，使用简单的特征加权得分
                weights = np.ones(len(self.features)) / len(self.features)
                feature_values = feature_df.values
                normalized_features = (feature_values - np.mean(feature_values)) / np.std(feature_values)
                score = np.sum(normalized_features * weights)
            else:
                # 使用模型预测
                score = self.model.predict_proba(feature_df)[0, 1]
            
            # 添加到选股池
            stock_info = date_data.iloc[0].to_dict()
            stock_info['股票代码'] = stock_code
            stock_info['模型得分'] = score
            
            daily_stocks.append(stock_info)
            
            # 进度监控
            if show_progress and (idx % 200 == 0 or idx == total_stocks):
                print(f"    已处理 {idx}/{total_stocks} 只股票 ({idx/total_stocks*100:.1f}%)")
        
        if daily_stocks:
            # 创建选股池
            pool = pd.DataFrame(daily_stocks)
            
            # 按模型得分排序
            pool = pool.sort_values('模型得分', ascending=False)
            
            # 选择前N个股票
            pool = pool.head(self.top_n)
            
            return pool
        else:
            return pd.DataFrame()
    
    def generate_daily_pool_cached(self, processed_stock_data, date):
        """生成指定日期的选股池（带缓存优化）
        
        Args:
            processed_stock_data: 处理后的股票数据字典
            date: 指定日期
            
        Returns:
            pd.DataFrame: 选股池
        """
        if not hasattr(self, '_daily_pool_cache'):
            self._daily_pool_cache = {}
        
        date_str = str(date)
        if date_str in self._daily_pool_cache:
            return self._daily_pool_cache[date_str]
        
        daily_stocks = []
        
        for stock_code, df in processed_stock_data.items():
            if df.empty:
                continue
            
            # 获取指定日期的数据
            date_data = df[df['交易日期'] == date]
            if date_data.empty:
                continue
            
            # 检查是否有足够的特征数据
            if not all(col in date_data.columns for col in self.features):
                continue
            
            # 获取特征值（保留DataFrame格式以保留特征名称）
            feature_df = date_data[self.features].iloc[[0]]
            
            # 计算模型得分
            if self.model is None:
                # 如果没有模型，使用简单的特征加权得分
                weights = np.ones(len(self.features)) / len(self.features)
                feature_values = feature_df.values
                normalized_features = (feature_values - np.mean(feature_values)) / np.std(feature_values)
                score = np.sum(normalized_features * weights)
            else:
                # 使用模型预测
                score = self.model.predict_proba(feature_df)[0, 1]
            
            # 添加到选股池
            stock_info = date_data.iloc[0].to_dict()
            stock_info['股票代码'] = stock_code
            stock_info['模型得分'] = score
            
            daily_stocks.append(stock_info)
        
        if daily_stocks:
            # 创建选股池
            pool = pd.DataFrame(daily_stocks)
            
            # 按模型得分排序
            pool = pool.sort_values('模型得分', ascending=False)
            
            # 选择前N个股票
            pool = pool.head(self.top_n)
            
            self._daily_pool_cache[date_str] = pool
            return pool
        else:
            self._daily_pool_cache[date_str] = pd.DataFrame()
            return pd.DataFrame()
    
    def backtest_strategy(self, processed_stock_data, start_date=None, end_date=None):
        """回测选股策略
        
        Args:
            processed_stock_data: 处理后的股票数据字典
            start_date: 回测开始日期
            end_date: 回测结束日期
            
        Returns:
            pd.DataFrame: 回测结果
        """
        # 获取所有股票的日期范围
        all_dates = []
        for df in processed_stock_data.values():
            if '交易日期' in df.columns:
                all_dates.extend(df['交易日期'].tolist())
        
        if not all_dates:
            return pd.DataFrame()
        
        # 确定回测日期范围
        if start_date is None:
            start_date = min(all_dates)
        if end_date is None:
            end_date = max(all_dates)
        
        # 生成回测日期序列
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        backtest_results = []
        
        for date in date_range:
            # 生成当日选股池
            daily_pool = self.generate_daily_pool(processed_stock_data, date)
            
            if not daily_pool.empty:
                # 计算未来收益
                for idx, row in daily_pool.iterrows():
                    stock_code = row['股票代码']
                    if stock_code in processed_stock_data:
                        df = processed_stock_data[stock_code]
                        # 找到当前日期的索引
                        date_idx = df[df['交易日期'] == date].index
                        if date_idx and len(date_idx) > 0:
                            date_idx = date_idx[0]
                            # 计算未来5日收益
                            if date_idx + 5 < len(df):
                                future_return = df.loc[date_idx + 5, '收盘价'] / df.loc[date_idx, '收盘价'] - 1
                                backtest_results.append({
                                    '日期': date,
                                    '股票代码': stock_code,
                                    '模型得分': row['模型得分'],
                                    '未来5日收益': future_return
                                })
        
        if backtest_results:
            return pd.DataFrame(backtest_results)
        else:
            return pd.DataFrame()
    
    def evaluate_strategy(self, backtest_results):
        """评估选股策略
        
        Args:
            backtest_results: 回测结果
            
        Returns:
            dict: 策略评估指标
        """
        if backtest_results.empty:
            return {}
        
        # 计算基本指标
        avg_return = backtest_results['未来5日收益'].mean()
        win_rate = (backtest_results['未来5日收益'] > 0).mean()
        Sharpe_ratio = backtest_results['未来5日收益'].mean() / backtest_results['未来5日收益'].std() if backtest_results['未来5日收益'].std() != 0 else 0
        max_return = backtest_results['未来5日收益'].max()
        min_return = backtest_results['未来5日收益'].min()
        
        # 计算分组收益
        backtest_results['分组'] = pd.qcut(backtest_results['模型得分'], q=5, labels=False)
        group_returns = backtest_results.groupby('分组')['未来5日收益'].mean()
        
        evaluation = {
            '平均收益': avg_return,
            '胜率': win_rate,
            '夏普比率': Sharpe_ratio,
            '最大收益': max_return,
            '最小收益': min_return,
            '分组收益': group_returns.to_dict()
        }
        
        return evaluation
        
    def generate_and_save_daily_pools(self, data_loader=None, feature_engineer=None, reports_dir=None, lookahead_days=5, processed_stock_data=None, start_date=None, end_date=None):
        """生成并保存每日选股池到CSV文件
        
        Args:
            data_loader: StockDataLoader实例，用于加载股票数据（可选，如果提供了processed_stock_data则不需要）
            feature_engineer: FeatureEngineer实例，用于特征工程（可选，如果提供了processed_stock_data则不需要）
            reports_dir: 报告保存目录路径
            lookahead_days: 预测未来几天的涨跌幅
            processed_stock_data: 已经处理好的股票数据字典（可选，如果提供了则直接使用）
            start_date: 回测开始日期
            end_date: 回测结束日期
            
        Returns:
            bool: 是否成功生成并保存
        """
        try:
            # 如果提供了已经处理好的数据，直接使用
            if processed_stock_data is not None and isinstance(processed_stock_data, dict):
                logger.info(f"直接使用提供的已处理数据，共 {len(processed_stock_data)} 支股票")
            else:
                # 没有提供已处理数据，需要重新加载和处理
                if not data_loader or not feature_engineer:
                    logger.error("没有提供已处理数据，且缺少必要的data_loader或feature_engineer参数")
                    return False
                    
                # 加载所有股票数据
                logger.info("开始加载所有股票数据...")
                raw_stock_data = data_loader.load_all_stocks()
                
                if not raw_stock_data:
                    logger.error("没有加载到任何股票数据")
                    return False
                
                # 数据清洗和特征工程
                logger.info("开始进行数据清洗和特征工程...")
                processed_stock_data = {}
                for stock_code, df in tqdm(raw_stock_data.items(), desc="处理股票数据"):
                    # 数据清洗
                    df_clean = feature_engineer.clean_data(df)
                    
                    # 特征工程
                    df_with_features = feature_engineer.generate_features(df_clean)
                    
                    if df_with_features is not None and not df_with_features.empty:
                        processed_stock_data[stock_code] = df_with_features
                
                logger.info(f"成功处理 {len(processed_stock_data)} 支股票的数据")
            
            # 准备训练数据
            logger.info("准备训练数据...")
            training_data = self.prepare_training_data(processed_stock_data, lookahead_days)
            
            if training_data.empty:
                logger.error("没有生成有效的训练数据")
                return False
            
            # 训练模型
            logger.info("训练选股模型...")
            if not self.train_model(training_data):
                logger.error("模型训练失败")
                return False
            
            # 获取所有交易日期
            logger.info("获取所有交易日期...")
            if data_loader:
                all_dates = data_loader.get_all_trading_dates(processed_stock_data)
            else:
                # 如果没有提供data_loader，直接从processed_stock_data中提取交易日期
                all_dates = set()
                for df in processed_stock_data.values():
                    if not df.empty and '交易日期' in df.columns:
                        all_dates.update(df['交易日期'].tolist())
                all_dates = sorted(list(all_dates))
                
                # 如果提供了start_date和end_date，过滤日期范围
                if start_date is not None and end_date is not None:
                    all_dates = [date for date in all_dates if start_date <= date <= end_date]
                elif start_date is not None:
                    all_dates = [date for date in all_dates if date >= start_date]
                elif end_date is not None:
                    all_dates = [date for date in all_dates if date <= end_date]
            
            if not all_dates:
                logger.error("没有找到交易日期")
                return False
            
            # 生成每日选股池
            logger.info(f"开始生成每日选股池，共 {len(all_dates)} 个交易日...")
            all_daily_pools = []
            
            for date in tqdm(all_dates, desc="生成每日选股池"):
                daily_pool = self.generate_daily_pool(processed_stock_data, date)
                if not daily_pool.empty:
                    daily_pool['日期'] = date
                    all_daily_pools.append(daily_pool)
            
            if not all_daily_pools:
                logger.error("没有生成任何选股池数据")
                return False
            
            # 合并所有选股池
            logger.info("合并所有选股池数据...")
            combined_pools = pd.concat(all_daily_pools, ignore_index=True)
            
            # 保存到CSV文件
            logger.info("保存选股池数据到CSV文件...")
            pools_filename = f"daily_pools_{self.strategy_name}.csv"
            pools_path = os.path.join(reports_dir, pools_filename)
            combined_pools.to_csv(pools_path, index=False, encoding='utf-8')
            
            logger.info(f"成功保存选股池到 {pools_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成并保存每日选股池失败: {e}", exc_info=True)
            return False
