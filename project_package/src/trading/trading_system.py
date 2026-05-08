#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易系统模块
负责模拟自动交易和回测
"""

import pandas as pd
import numpy as np
import logging
import json
import os

logger = logging.getLogger(__name__)

class TradingSystem:
    """交易系统类"""
    
    def __init__(self, config):
        """初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.initial_capital = config['trading']['initial_capital']
        self.buy_amount = config['trading']['buy_amount']
        self.profit_threshold = config['trading']['profit_threshold']
        self.loss_threshold = config['trading']['loss_threshold']
        self.max_positions = config['trading']['max_positions']
        
        # 加载卖出策略配置
        self.sell_strategies = config['trading'].get('sell_strategies', {})
        # 止损止盈策略（默认启用）
        self.profit_loss_enabled = self.sell_strategies.get('profit_loss', {}).get('enabled', True)
        # 时间基础卖出策略
        self.time_based_enabled = self.sell_strategies.get('time_based', {}).get('enabled', False)
        self.holding_days = self.sell_strategies.get('time_based', {}).get('holding_days', 10)
        # RSI超买卖出策略
        self.rsi_overbought_enabled = self.sell_strategies.get('rsi_overbought', {}).get('enabled', False)
        self.rsi_threshold = self.sell_strategies.get('rsi_overbought', {}).get('rsi_threshold', 70)
        
        # 初始化交易状态
        self.reset()
    
    def reset(self):
        """重置交易系统状态"""
        logger.info(f"DEBUG reset: 重置前 buy_amount = {self.buy_amount}, initial_capital = {self.initial_capital}")
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_values = []
        self.current_date = None
        logger.info(f"DEBUG reset: 重置后 buy_amount = {self.buy_amount}")
    
    def buy_stock(self, stock_code, price, date):
        """买入股票
        
        Args:
            stock_code: 股票代码
            price: 买入价格
            date: 买入日期
            
        Returns:
            bool: 买入是否成功
        """
        logger.info(f"DEBUG buy_stock: 当前 buy_amount = {self.buy_amount}, 可用资金 = {self.capital}, 股票代码 = {stock_code}, 价格 = {price}")
        # 检查资金是否足够
        if self.capital < self.buy_amount:
            logger.warning(f"资金不足，无法买入 {stock_code}")
            return False
        
        # 检查持仓数量是否达到上限
        if len(self.positions) >= self.max_positions:
            logger.warning(f"持仓数量已达上限，无法买入 {stock_code}")
            return False
        
        # 计算买入股数（取整数）
        shares = int(self.buy_amount / price)
        if shares == 0:
            logger.warning(f"买入金额太小，无法买入 {stock_code}")
            return False
        
        # 计算实际买入金额
        actual_amount = round(shares * price, 2)
        
        # 更新资金和持仓
        self.capital = round(self.capital - actual_amount, 2)
        self.positions[stock_code] = {
            '价格': price,
            '股数': shares,
            '买入日期': date
        }
        
        # 记录交易
        self.trades.append({
            '日期': date,
            '股票代码': stock_code,
            '类型': '买入',
            '价格': price,
            '股数': shares,
            '金额': actual_amount,
            '剩余资金': self.capital
        })
        
        logger.info(f"在 {date} 以 {price} 元买入 {shares} 股 {stock_code}")
        return True
    
    def sell_stock(self, stock_code, price, date, sell_reason=""):
        """卖出股票
        
        Args:
            stock_code: 股票代码
            price: 卖出价格
            date: 卖出日期
            sell_reason: 卖出原因
            
        Returns:
            bool: 卖出是否成功
        """
        if stock_code not in self.positions:
            logger.warning(f"没有持仓 {stock_code}")
            return False
        
        # 获取持仓信息
        position = self.positions[stock_code]
        shares = position['股数']
        buy_price = position['价格']
        
        # 计算卖出金额
        sell_amount = round(shares * price, 2)
        
        # 计算收益
        profit = round(sell_amount - (shares * buy_price), 2)
        profit_rate = round((price / buy_price - 1), 4)  # 保留4位小数，方便显示为百分比时保留2位
        
        # 更新资金和持仓
        self.capital = round(self.capital + sell_amount, 2)
        del self.positions[stock_code]
        
        # 记录交易
        trade = {
            '日期': date,
            '股票代码': stock_code,
            '类型': '卖出',
            '价格': price,
            '股数': shares,
            '金额': sell_amount,
            '剩余资金': self.capital,
            '收益': profit,
            '收益率': profit_rate
        }
        
        # 添加卖出原因
        if sell_reason:
            trade['卖出原因'] = sell_reason
        
        self.trades.append(trade)
        
        logger.info(f"在 {date} 以 {price} 元卖出 {shares} 股 {stock_code}，收益: {profit:.2f} 元，收益率: {profit_rate:.2%}")
        return True
    
    def check_stop_condition(self, stock_code, current_price, date, stock_data):
        """检查是否需要卖出股票
        
        Args:
            stock_code: 股票代码
            current_price: 当前价格
            date: 当前日期
            stock_data: 股票数据字典
            
        Returns:
            tuple: (bool, str) - 是否需要卖出，卖出原因
        """
        if stock_code not in self.positions:
            return False, ""
        
        # 获取持仓信息
        position = self.positions[stock_code]
        buy_price = position['价格']
        buy_date = position['买入日期']
        
        # 1. 检查止损或止盈条件（原有策略）
        if self.profit_loss_enabled:
            profit_rate = (current_price / buy_price - 1)
            if profit_rate >= self.profit_threshold:
                return True, f"止损止盈策略：盈利达到 {self.profit_threshold*100}%"
            elif profit_rate <= self.loss_threshold:
                return True, f"止损止盈策略：亏损达到 {self.loss_threshold*100}%"
        
        # 2. 时间基础卖出策略（持有超过指定天数）
        if self.time_based_enabled:
            # 计算持有天数
            buy_dt = pd.to_datetime(buy_date)
            current_dt = pd.to_datetime(date)
            holding_days = (current_dt - buy_dt).days
            if holding_days >= self.holding_days:
                return True, f"时间基础策略：持有时间达到 {self.holding_days} 天"
        
        # 3. RSI超买卖出策略
        if self.rsi_overbought_enabled and stock_code in stock_data:
            df = stock_data[stock_code]
            # 获取当日数据
            date_data = df[df['交易日期'] == date]
            if not date_data.empty and 'RSI14' in date_data.columns:
                rsi_value = date_data.iloc[0]['RSI14']
                if rsi_value > self.rsi_threshold:
                    return True, f"RSI超买策略：RSI超过 {self.rsi_threshold}"
        
        return False, ""
    
    def run_daily_trading(self, date, stock_data, daily_pool):
        """执行每日交易
        
        Args:
            date: 交易日期
            stock_data: 股票数据字典
            daily_pool: 当日选股池
        """
        self.current_date = date
        
        # 检查现有持仓是否需要卖出
        stocks_to_sell = []
        for stock_code in self.positions:
            if stock_code in stock_data:
                df = stock_data[stock_code]
                # 获取当日价格
                date_data = df[df['交易日期'] == date]
                if not date_data.empty:
                    current_price = date_data.iloc[0]['收盘价']
                    should_sell, sell_reason = self.check_stop_condition(stock_code, current_price, date, stock_data)
                    if should_sell:
                        stocks_to_sell.append((stock_code, current_price, sell_reason))
        
        # 执行卖出
        for stock_code, price, sell_reason in stocks_to_sell:
            self.sell_stock(stock_code, price, date, sell_reason)
        
        # 买入新股票
        if daily_pool is not None and not daily_pool.empty:
            # 按模型得分排序
            daily_pool = daily_pool.sort_values('模型得分', ascending=False)
            
            # 买入前N个股票，直到持仓满或资金不足
            for _, stock in daily_pool.iterrows():
                stock_code = stock['股票代码']
                if stock_code not in self.positions and stock_code in stock_data:
                    df = stock_data[stock_code]
                    date_data = df[df['交易日期'] == date]
                    if not date_data.empty:
                        price = date_data.iloc[0]['收盘价']
                        self.buy_stock(stock_code, price, date)
    
    def calculate_portfolio_value(self, date, stock_data):
        """计算投资组合价值
        
        Args:
            date: 日期
            stock_data: 股票数据字典
            
        Returns:
            float: 投资组合总价值
        """
        # 计算持仓价值
        portfolio_value = self.capital
        
        for stock_code, position in self.positions.items():
            if stock_code in stock_data:
                df = stock_data[stock_code]
                date_data = df[df['交易日期'] == date]
                if not date_data.empty:
                    current_price = date_data.iloc[0]['收盘价']
                    portfolio_value += position['股数'] * current_price
        
        # 记录每日价值
        self.daily_values.append({
            '日期': date,
            '资金': self.capital,
            '持仓价值': portfolio_value - self.capital,
            '总价值': portfolio_value
        })
        
        return portfolio_value
    
    def save_daily_snapshot(self, date, report_dir, strategy_name):
        """保存每日状态快照
        
        Args:
            date: 交易日期
            report_dir: 报告目录路径
            strategy_name: 策略名称
        """
        try:
            # 计算持仓市值
            holdings_value = 0
            holdings_detail = {}
            
            for stock_code, position in self.positions.items():
                holdings_detail[stock_code] = {
                    'shares': position['股数'],
                    'buy_price': position['价格'],
                    'buy_date': position['买入日期']
                }
            
            # 计算总资产（现金 + 持仓市值）
            # 注意：这里需要传入stock_data才能计算持仓市值，但为了简化，我们只保存现金和持仓信息
            # 持仓市值可以在前端查询时根据当日收盘价计算
            total_value = self.capital
            
            # 构建快照数据
            snapshot = {
                'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                'cash': float(self.capital),
                'holdings': holdings_detail,
                'total_value': float(total_value),
                'position_count': len(self.positions)
            }
            
            # 确保报告目录存在
            os.makedirs(report_dir, exist_ok=True)
            
            # 保存为JSON文件
            snapshot_file = os.path.join(report_dir, f'snapshot_{strategy_name}_{date.strftime("%Y%m%d") if hasattr(date, "strftime") else date}.json')
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            
            logger.info(f"成功保存每日快照: {snapshot_file}")
            
        except Exception as e:
            logger.error(f"保存每日快照失败: {e}", exc_info=True)
    
    def backtest(self, processed_stock_data, stock_selector, start_date=None, end_date=None, report_dir=None, strategy_name='default'):
        """回测交易策略
        
        Args:
            processed_stock_data: 处理后的股票数据字典
            stock_selector: 选股器
            start_date: 回测开始日期
            end_date: 回测结束日期
            report_dir: 报告目录路径（用于保存快照）
            strategy_name: 策略名称（用于快照文件名）
            
        Returns:
            dict: 回测结果
        """
        # 重置交易系统
        self.reset()
        
        # 获取所有交易日期
        all_dates = []
        for df in processed_stock_data.values():
            if '交易日期' in df.columns:
                all_dates.extend(df['交易日期'].tolist())
        
        if not all_dates:
            logger.error("没有交易日期数据")
            return {}
        
        # 去重并排序
        all_dates = sorted(list(set(all_dates)))
        
        # 确保start_date和end_date是datetime类型
        if start_date is not None and not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date)
        if end_date is not None and not isinstance(end_date, pd.Timestamp):
            end_date = pd.to_datetime(end_date)
        
        # 确定回测日期范围
        if start_date is None:
            start_date = min(all_dates)
        if end_date is None:
            end_date = max(all_dates)
        
        # 过滤出实际的交易日期（只保留在股票数据中存在的日期，并且在start_date和end_date之间）
        trading_dates = [date for date in all_dates if start_date <= date <= end_date]
        
        logger.info(f"开始回测，回测日期范围: {start_date} 到 {end_date}")
        logger.info(f"实际交易日期数量: {len(trading_dates)}")
        print(f"\n  回测日期范围: {start_date} 到 {end_date}")
        print(f"  实际交易日期数量: {len(trading_dates)}")
        
        # 每日执行交易
        for idx, date in enumerate(trading_dates, 1):
            # 生成当日选股池（使用缓存优化）
            daily_pool = stock_selector.generate_daily_pool_cached(processed_stock_data, date)
            
            # 执行交易
            self.run_daily_trading(date, processed_stock_data, daily_pool)
            
            # 计算当日投资组合价值
            self.calculate_portfolio_value(date, processed_stock_data)
            
            # 保存每日快照（如果指定了报告目录）
            if report_dir:
                self.save_daily_snapshot(date, report_dir, strategy_name)
            
            # 进度监控：每10个交易日打印一次
            if idx % 10 == 0 or idx == len(trading_dates):
                print(f"  已处理 {idx}/{len(trading_dates)} 个交易日 ({idx/len(trading_dates)*100:.1f}%)")
                logger.info(f"回测进度: {idx}/{len(trading_dates)} 个交易日")
        
        # 计算最终投资组合价值
        last_date = trading_dates[-1] if trading_dates else end_date
        final_value = self.calculate_portfolio_value(last_date, processed_stock_data)
        
        # 平仓所有持仓（用于生成交易记录和更新最终价值）
        for stock_code in list(self.positions.keys()):
            if stock_code in processed_stock_data:
                df = processed_stock_data[stock_code]
                # 找到最后一个交易日
                last_trade_date = df['交易日期'].max()
                date_data = df[df['交易日期'] == last_trade_date]
                if not date_data.empty:
                    price = date_data.iloc[0]['收盘价']
                    self.sell_stock(stock_code, price, last_trade_date, "回测结束：强制平仓")
        
        # 强制平仓后重新计算最终投资组合价值
        final_value_after_closeout = self.calculate_portfolio_value(last_date, processed_stock_data)
        
        # 更新 daily_values 的最后一条记录，使其包含强制平仓后的状态
        if self.daily_values:
            self.daily_values[-1]['资金'] = self.capital
            self.daily_values[-1]['持仓价值'] = final_value_after_closeout - self.capital
            self.daily_values[-1]['总价值'] = final_value_after_closeout
        
        # 生成回测报告
        return self.generate_backtest_report()
    
    def generate_backtest_report(self):
        """生成回测报告
        
        Returns:
            dict: 回测报告
        """
        if not self.daily_values:
            return {}
        
        # 转换为DataFrame
        daily_values_df = pd.DataFrame(self.daily_values)
        daily_values_df = daily_values_df.sort_values('日期')
        
        # 计算收益率
        daily_values_df['收益率'] = daily_values_df['总价值'].pct_change()
        daily_values_df['累计收益率'] = (daily_values_df['收益率'] + 1).cumprod() - 1
        
        # 计算回测指标
        total_return = daily_values_df['累计收益率'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(daily_values_df)) - 1
        sharpe_ratio = (daily_values_df['收益率'].mean() * 252) / (daily_values_df['收益率'].std() * np.sqrt(252)) if daily_values_df['收益率'].std() != 0 else 0
        
        # 计算最大回撤
        daily_values_df['峰值'] = daily_values_df['总价值'].cummax()
        daily_values_df['回撤'] = (daily_values_df['总价值'] - daily_values_df['峰值']) / daily_values_df['峰值']
        max_drawdown = daily_values_df['回撤'].min()
        
        # 计算胜率
        if len(self.trades) > 0:
            sell_trades = [trade for trade in self.trades if trade['类型'] == '卖出']
            if sell_trades:
                win_rate = sum(1 for trade in sell_trades if trade['收益'] > 0) / len(sell_trades)
            else:
                win_rate = 0
            avg_profit_per_trade = sum(trade['收益'] for trade in sell_trades) / len(sell_trades) if sell_trades else 0
        else:
            win_rate = 0
            avg_profit_per_trade = 0
        
        report = {
            '初始资金': self.initial_capital,
            '最终资金': daily_values_df['总价值'].iloc[-1],
            '总收益率': total_return,
            '年化收益率': annual_return,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '交易次数': len(self.trades),
            '胜率': win_rate,
            '平均每次交易收益': avg_profit_per_trade,
            '每日价值': daily_values_df.to_dict('records'),
            '交易记录': self.trades
        }
        
        logger.info(f"回测完成，总收益率: {total_return:.2%}, 年化收益率: {annual_return:.2%}, 夏普比率: {sharpe_ratio:.2f}, 最大回撤: {max_drawdown:.2%}")
        
        return report
    
    def backtest_all_sell_strategies(self, processed_stock_data, stock_selector, start_date=None, end_date=None, report_dir=None, base_strategy_name='default'):
        """回测所有卖出策略并生成对比数据
        
        Args:
            processed_stock_data: 处理后的股票数据字典
            stock_selector: 选股器
            start_date: 回测开始日期
            end_date: 回测结束日期
            report_dir: 报告目录路径（用于保存快照）
            base_strategy_name: 基础策略名称（用于快照文件名）
            
        Returns:
            dict: 包含所有策略回测结果的字典
        """
        # 定义三种卖出策略的配置
        strategies = [
            {
                'name': '止损止盈策略',
                'description': '仅使用止损止盈条件卖出',
                'config': {
                    'profit_loss': {'enabled': True},
                    'time_based': {'enabled': False},
                    'rsi_overbought': {'enabled': False}
                }
            },
            {
                'name': '时间基础策略',
                'description': '仅使用持有时间条件卖出',
                'config': {
                    'profit_loss': {'enabled': False},
                    'time_based': {'enabled': True},
                    'rsi_overbought': {'enabled': False}
                }
            },
            {
                'name': 'RSI超买策略',
                'description': '仅使用RSI超买条件卖出',
                'config': {
                    'profit_loss': {'enabled': False},
                    'time_based': {'enabled': False},
                    'rsi_overbought': {'enabled': True}
                }
            }
        ]
        
        # 运行所有策略的回测
        results = {}
        
        for idx, strategy in enumerate(strategies, 1):
            logger.info(f"开始回测: {strategy['name']}")
            print(f"\n  开始回测 {idx}/{len(strategies)}: {strategy['name']}")
            print(f"  策略描述: {strategy['description']}")
            
            # 复制原始配置
            strategy_config = self.config.copy()
            
            # 完全替换卖出策略配置，确保只使用当前策略的配置
            # 这样可以避免原始配置中的其他策略设置影响当前策略
            strategy_config['trading']['sell_strategies'] = strategy['config']
            
            # 确保所有策略类型都有明确的配置
            # 为未指定的策略类型设置enabled=False
            default_strategies = ['profit_loss', 'time_based', 'rsi_overbought']
            for strategy_type in default_strategies:
                if strategy_type not in strategy_config['trading']['sell_strategies']:
                    strategy_config['trading']['sell_strategies'][strategy_type] = {'enabled': False}
            
            # 创建新的交易系统实例
            trading_system = TradingSystem(strategy_config)
            
            # 构建策略名称（基础策略名 + 卖出策略名）
            strategy_name_en = {
                '止损止盈策略': 'profit_loss',
                '时间基础策略': 'time_based',
                'RSI超买策略': 'rsi_overbought'
            }
            sell_strategy_name_en = strategy_name_en.get(strategy['name'], strategy['name'])
            full_strategy_name = f"{base_strategy_name}_{sell_strategy_name_en}"
            
            # 运行回测
            backtest_result = trading_system.backtest(processed_stock_data, stock_selector, start_date, end_date, report_dir, full_strategy_name)
            
            # 保存结果
            results[strategy['name']] = {
                'description': strategy['description'],
                'config': strategy['config'],
                'result': backtest_result
            }
            
            print(f"  ✓ {strategy['name']} 回测完成")
        
        return results
