#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网站应用模块
负责用户界面和数据可视化
"""

import os
import sys
from pathlib import Path
from functools import lru_cache

# 将src目录添加到Python搜索路径
src_dir = Path(__file__).parent.parent
if src_dir not in sys.path:
    sys.path.insert(0, str(src_dir))

from flask import Flask, render_template, request, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging

# 配置日志
logs_dir = Path(__file__).parent.parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)  # 确保日志目录存在

# 添加全局锁，防止同时执行多个回测任务
import threading

backtest_lock = threading.Lock()
backtest_running = False

# 策略名称映射（中文策略名 -> 英文策略名）
STRATEGY_NAME_MAP = {
    'profit_loss': 'profit_loss',
    'time_based': 'time_based',
    'rsi_overbought': 'rsi_overbought',
    '止损止盈策略': 'profit_loss',
    '时间基础策略': 'time_based',
    'RSI超买策略': 'rsi_overbought'
}


# 不使用缓存，确保每次都加载最新数据
def load_csv_with_cache(file_path, date_column=None):
    """加载CSV文件（不缓存结果，确保每次都获取最新数据）

    Args:
        file_path: 文件路径
        date_column: 需要转换为日期类型的列名

    Returns:
        pd.DataFrame: 加载并处理后的数据，如果加载失败则返回空DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return pd.DataFrame()
    
    # 检查文件大小是否为0
    if os.path.getsize(file_path) == 0:
        logger.warning(f"文件为空: {file_path}")
        return pd.DataFrame()
    
    # 尝试不同的编码格式
    encodings = ['utf-8-sig', 'gbk', 'gb2312', 'utf-16', 'latin-1']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            # 检查是否有列数据
            if df.empty:
                logger.warning(f"文件内容为空: {file_path}")
                return pd.DataFrame()
            # 检查是否有乱码列名（如果列名包含非ASCII字符，尝试解码）
            logger.info(f"使用编码 {encoding} 成功加载文件: {file_path}, 列名: {list(df.columns)}, 行数: {len(df)}")
            break
        except Exception as e:
            logger.warning(f"使用编码 {encoding} 加载文件失败: {file_path}, 错误: {e}")
            continue

    if df is None:
        logger.error(f"无法使用任何编码加载文件: {file_path}")
        return pd.DataFrame()  # 返回空DataFrame而不是抛出异常

    # 清理列名，移除可能的空格和特殊字符
    df.columns = df.columns.str.strip()

    if date_column:
            if date_column in df.columns:
                try:
                    # 转换为datetime类型，保留为datetime对象以便进行后续比较
                    df[date_column] = pd.to_datetime(df[date_column],
                                                     format='%Y-%m-%d',
                                                     errors='coerce')
                    # 检查是否有解析失败的日期
                    if df[date_column].isnull().any():
                        logger.warning(f"部分日期解析失败: {file_path}, {date_column}列")
                        logger.warning(f"原始值示例: {df[date_column].dropna().head(5).tolist()}")
                        logger.warning(f"解析后值示例: {df[date_column].head(5).tolist()}")
                    # 不转换为字符串，保留datetime类型以支持日期比较
                except Exception as e:
                    logger.error(f"日期转换失败: {e}", exc_info=True)
            else:
                logger.warning(f"日期列'{date_column}'不在文件中，可用列: {list(df.columns)}")

    return df


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'website.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['DEBUG'] = True

# 初始化登录管理器
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# 上下文处理器，用于在所有模板中注入当前年份
@app.context_processor
def inject_current_year():
    import datetime
    return {'current_year': datetime.datetime.now().year}


# 模拟用户数据
users = {
    'admin': {'password': 'password'}
}


class User(UserMixin):
    def __init__(self, id):
        self.id = id


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """退出登录"""
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """首页"""
    try:
        from pathlib import Path
        import datetime
        import pandas as pd

        # 获取当前文件的绝对路径
        current_file = Path(__file__).resolve()

        # 计算项目根目录
        base_dir = current_file.parents[2]

        # 构建报告目录的绝对路径
        reports_dir = base_dir / 'reports'

        # 1. 计算今日选股数量
        today_stock_count = 0

        # 获取今日日期
        today = datetime.date.today().strftime('%Y-%m-%d')

        # 加载混合策略的选股池数据
        # 根据配置文件，默认的卖出策略是时间基础策略
        pools_path = reports_dir / 'daily_pools_mixed_时间基础策略.csv'
        if not pools_path.exists():
            # 如果时间基础策略文件不存在，尝试其他策略
            for strategy in ['止损止盈策略', 'RSI超买策略']:
                pools_path = reports_dir / f'daily_pools_mixed_{strategy}.csv'
                if pools_path.exists():
                    break
        if not pools_path.exists():
            # 如果都不存在，尝试不带策略后缀的文件
            pools_path = reports_dir / 'daily_pools_mixed.csv'
        if pools_path.exists():
            try:
                daily_pools = load_csv_with_cache(str(pools_path), date_column='日期')

                # 检查是否有今日的选股数据
                if '日期' in daily_pools.columns and pd.api.types.is_datetime64_any_dtype(daily_pools['日期']):
                    # 将日期转换为字符串格式进行比较
                    daily_pools['日期_str'] = daily_pools['日期'].dt.strftime('%Y-%m-%d')
                    today_pools = daily_pools[daily_pools['日期_str'] == today]
                    today_stock_count = len(today_pools)
            except Exception as e:
                logger.error(f"加载选股池数据失败: {e}")

        # 2. 计算当前持仓股票数量
        current_position_count = 0

        # 加载混合策略的交易记录
        # 根据配置文件，默认的卖出策略是时间基础策略
        trades_path = reports_dir / 'trades_mixed_时间基础策略.csv'
        if not trades_path.exists():
            # 如果时间基础策略文件不存在，尝试其他策略
            for strategy in ['止损止盈策略', 'RSI超买策略']:
                trades_path = reports_dir / f'trades_mixed_{strategy}.csv'
                if trades_path.exists():
                    break
        if trades_path.exists():
            trades = load_csv_with_cache(str(trades_path), date_column='日期')

            # 计算当前持仓
            positions = []
            if '股票代码' in trades.columns and '类型' in trades.columns and '股数' in trades.columns:
                stock_codes = trades['股票代码'].unique()

                for stock_code in stock_codes:
                    stock_trades = trades[trades['股票代码'] == stock_code].sort_values('日期')
                    total_buy = stock_trades[stock_trades['类型'] == '买入']['股数'].sum()
                    total_sell = stock_trades[stock_trades['类型'] == '卖出']['股数'].sum()
                    current_shares = total_buy - total_sell

                    if current_shares > 0:
                        positions.append({
                            '股票代码': stock_code,
                            '持仓数量': current_shares
                        })

                current_position_count = len(positions)

        # 3. 计算策略年化收益和最大回撤
        current_year_return = 0
        max_drawdown = 0

        # 加载混合策略的每日价值数据
        # 根据配置文件，默认的卖出策略是时间基础策略
        daily_values_path = reports_dir / 'daily_values_mixed_时间基础策略.csv'
        if not daily_values_path.exists():
            # 如果时间基础策略文件不存在，尝试其他策略
            for strategy in ['止损止盈策略', 'RSI超买策略']:
                daily_values_path = reports_dir / f'daily_values_mixed_{strategy}.csv'
                if daily_values_path.exists():
                    break
        if daily_values_path.exists():
            daily_values = load_csv_with_cache(str(daily_values_path), date_column='日期')

            if '总价值' in daily_values.columns and '日期' in daily_values.columns:
                # 计算累计收益率
                if len(daily_values) > 0:
                    # 假设初始资金为100万
                    initial_value = 1000000
                    latest_value = daily_values['总价值'].iloc[-1]

                    # 计算累计收益率
                    cumulative_return = (latest_value / initial_value - 1) * 100

                    # 计算年化收益率（假设一年252个交易日）
                    days = (daily_values['日期'].iloc[-1] - daily_values['日期'].iloc[0]).days
                    if days > 0:
                        current_year_return = ((latest_value / initial_value) ** (365 / days) - 1) * 100

                    # 计算最大回撤
                    daily_values['峰值'] = daily_values['总价值'].cummax()
                    daily_values['回撤'] = (daily_values['总价值'] - daily_values['峰值']) / daily_values['峰值'] * 100
                    max_drawdown = daily_values['回撤'].min()

        # 4. 加载当前持仓数据
        current_positions = []

        if trades_path.exists() and positions:
            # 加载选股池数据以获取股票名称
            pools_path = reports_dir / 'daily_pools.csv'
            stock_name_map = {}
            if pools_path.exists():
                daily_pools = load_csv_with_cache(str(pools_path))
                if '股票代码' in daily_pools.columns and '名称' in daily_pools.columns:
                    stock_name_map = dict(zip(daily_pools['股票代码'], daily_pools['名称']))

            # 价格最小变动单位调整函数
            def adjust_price_by_tick(price, stock_code):
                """根据股票代码和价格应用价格最小变动单位规则

                Args:
                    price: 原始价格
                    stock_code: 股票代码

                Returns:
                    float: 调整后的价格，符合最小变动单位要求
                """
                # A股价格最小变动单位规则：
                # 一般为0.01元人民币
                # 科创板/创业板某些高价股可能为0.05元

                # 根据股票代码判断板块：
                # 科创板股票代码以688开头，创业板以300开头
                # 这里简化处理：科创板/创业板股票价格>=100元时使用0.05元变动单位
                tick_size = 0.01

                # 判断是否为科创板或创业板股票
                stock_code_str = str(stock_code)
                if stock_code_str.startswith('688') or stock_code_str.startswith('300'):
                    # 高价股使用0.05元变动单位
                    if price >= 100:
                        tick_size = 0.05

                # 应用最小变动单位调整，使用更精确的计算方法避免浮点数精度问题
                # 计算需要调整的倍数
                multiplier = round(price / tick_size)
                # 使用format函数确保结果的小数位数正确
                if tick_size == 0.01:
                    # 对于0.01的变动单位，保留2位小数
                    adjusted_price = float(f"{multiplier * tick_size:.2f}")
                elif tick_size == 0.05:
                    # 对于0.05的变动单位，保留2位小数（0.05的倍数）
                    adjusted_price = float(f"{multiplier * tick_size:.2f}")
                else:
                    # 其他情况保留2位小数
                    adjusted_price = float(f"{multiplier * tick_size:.2f}")
                return adjusted_price

            # 重新计算持仓，包含更多详细信息，遵循A股交易规则（100股一手）
            current_positions = []
            trades = load_csv_with_cache(str(trades_path), date_column='日期')

            # A股交易规则：100股一手
            A_SHARE_LOT_SIZE = 100

            if '股票代码' in trades.columns and '类型' in trades.columns and '股数' in trades.columns and '价格' in trades.columns:
                stock_codes = trades['股票代码'].unique()

                for stock_code in stock_codes:
                    stock_trades = trades[trades['股票代码'] == stock_code].sort_values('日期')

                    # 按照A股交易规则调整交易股数（100股一手）
                    # 买入和卖出的股数都向下取整到最接近的100的倍数
                    buy_trades = stock_trades[stock_trades['类型'] == '买入'].copy()
                    sell_trades = stock_trades[stock_trades['类型'] == '卖出'].copy()

                    # 调整买入股数
                    buy_trades['调整后股数'] = (buy_trades['股数'] // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE
                    # 调整卖出股数
                    sell_trades['调整后股数'] = (sell_trades['股数'] // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE

                    # 计算总买入和总卖出股数
                    total_buy = buy_trades['调整后股数'].sum()
                    total_sell = sell_trades['调整后股数'].sum()

                    # 如果没有有效交易，跳过
                    if total_buy == 0:
                        continue

                    current_shares = total_buy - total_sell

                    if current_shares > 0:
                        # 获取最后一次有效买入的价格作为参考买入价格
                        # 过滤掉调整后股数为0的买入记录
                        valid_buy_trades = buy_trades[buy_trades['调整后股数'] > 0]
                        if not valid_buy_trades.empty:
                            last_buy_price = valid_buy_trades['价格'].iloc[-1]
                            # 应用价格最小变动单位调整
                            last_buy_price = adjust_price_by_tick(last_buy_price, stock_code)
                        else:
                            last_buy_price = 0
                        
                        # 获取用户选择日期的收盘价作为当前价格
                        current_price = None
                        
                        # 确定用户选择的日期
                        if selected_date:
                            selected_datetime = pd.to_datetime(selected_date)
                            
                            # 尝试从选股池数据中获取该日期的收盘价
                            if stock_code in stock_price_data and not stock_price_data[stock_code].empty:
                                price_data = stock_price_data[stock_code]
                                
                                # 查找指定日期的收盘价
                                if selected_datetime in price_data.index:
                                    current_price = price_data.loc[selected_datetime, '收盘价']
                                else:
                                    # 如果没有找到指定日期，查找最接近的前一个交易日的收盘价
                                    # 获取所有在指定日期之前的价格数据
                                    before_selected = price_data[price_data.index <= selected_datetime]
                                    if not before_selected.empty:
                                        # 取最后一个（最接近指定日期）的收盘价
                                        current_price = before_selected.iloc[-1]['收盘价']
                        
                        # 如果没有找到合适的收盘价，使用最后一次交易的价格作为备选
                        if current_price is None:
                            current_price = stock_trades['价格'].iloc[-1]
                        
                        # 应用价格最小变动单位调整
                        current_price = adjust_price_by_tick(current_price, stock_code)

                        # 计算收益率
                        if last_buy_price > 0:
                            return_rate = (current_price - last_buy_price) / last_buy_price * 100
                        else:
                            return_rate = 0

                        # 使用股票名称映射获取实际股票名称，如果没有找到则使用股票代码
                        stock_name = stock_name_map.get(stock_code, stock_code)

                        # 计算持仓手数（100股为1手）
                        lot_count = current_shares // A_SHARE_LOT_SIZE

                        current_positions.append({
                            '股票代码': stock_code,
                            '名称': stock_name,
                            '持仓数量': current_shares,
                            '持仓手数': lot_count,
                            '买入价格': last_buy_price,
                            '当前价格': current_price,
                            '收益率(%)': round(return_rate, 2)
                        })

        return render_template('index.html',
                               today_stock_count=today_stock_count,
                               current_position_count=current_position_count,
                               current_year_return=round(current_year_return, 2),
                               max_drawdown=round(max_drawdown, 2),
                               current_positions=current_positions)
    except Exception as e:
        logger.error(f"首页数据处理失败: {e}", exc_info=True)
        # 如果发生错误，返回默认值
        return render_template('index.html',
                               today_stock_count=0,
                               current_position_count=0,
                               current_year_return=0,
                               max_drawdown=0,
                               current_positions=[])


@app.route('/stock_pool')
@login_required
def stock_pool():
    """选股池页面"""
    try:
        # 始终使用混合策略
        selected_strategy = 'mixed'
        logger.info(f"选股池页面请求，策略: {selected_strategy}")

        # 使用pathlib处理路径，更现代、更可靠
        from pathlib import Path

        # 获取当前文件的绝对路径
        current_file = Path(__file__).resolve()
        logger.info(f"当前文件路径: {current_file}")

        # 计算项目根目录
        # app.py位于src/website/目录下，所以需要向上走2级目录
        base_dir = current_file.parents[2]
        logger.info(f"项目根目录: {base_dir}")

        # 构建报告目录的绝对路径
        reports_dir = base_dir / 'reports'
        logger.info(f"报告目录: {reports_dir}")

        # 构建选股池文件的绝对路径
        pools_path = reports_dir / f'daily_pools_{selected_strategy}.csv'

        logger.info(f"选股池文件路径: {pools_path}")

        # 确保文件存在
        if not pools_path.exists():
            logger.error(f"选股池文件不存在: {pools_path}")
            # 返回空数据而不是抛出异常，避免页面崩溃
            return render_template('stock_pool.html',
                                   stock_pool=[],
                                   available_dates=[],
                                   selected_date=None,
                                   selected_strategy=selected_strategy)

        logger.info(f"加载选股池文件: {pools_path}")
        daily_pools = load_csv_with_cache(str(pools_path), date_column='日期')
        logger.info(f"选股池数据形状: {daily_pools.shape}")

        # 检查日期列是否为datetime类型
        if not pd.api.types.is_datetime64_any_dtype(daily_pools['日期']):
            logger.error("日期列不是datetime类型，尝试重新转换")
            try:
                daily_pools['日期'] = pd.to_datetime(daily_pools['日期'], errors='coerce')
            except Exception as e:
                logger.error(f"重新转换日期失败: {e}", exc_info=True)
                # 如果重新转换也失败，使用原始值，但不能使用.dt访问器
                logger.warning("无法转换日期为datetime类型，将使用原始日期值")

        # 转换日期格式（只在日期列是datetime类型时执行）
        if pd.api.types.is_datetime64_any_dtype(daily_pools['日期']):
            daily_pools['日期'] = daily_pools['日期'].dt.date

        # 获取所有可用日期
        available_dates = sorted(daily_pools['日期'].unique(), reverse=True)
        logger.info(f"可用日期数量: {len(available_dates)}")
        logger.info(f"最新日期: {available_dates[0]}")

        # 获取用户选择的日期，如果没有则使用最新日期
        selected_date = request.args.get('date')
        if selected_date:
            selected_date = pd.to_datetime(selected_date).date()
        else:
            selected_date = available_dates[0]

        logger.info(f"选择的日期: {selected_date}")

        # 根据选择的日期筛选选股池数据
        filtered_pool = daily_pools[daily_pools['日期'] == selected_date]
        logger.info(f"筛选后的数据形状: {filtered_pool.shape}")

        # 只保留需要的列
        selected_columns = ['股票代码', '名称', '所属行业', '模型得分', '涨跌幅(%)', '市盈率', '市净率']
        logger.info(f"选择的列: {selected_columns}")
        filtered_pool = filtered_pool[selected_columns]

        # 按模型得分降序排序
        filtered_pool = filtered_pool.sort_values('模型得分', ascending=False)

        logger.info(f"返回选股池数据，记录数: {len(filtered_pool)}")
        return render_template('stock_pool.html',
                               stock_pool=filtered_pool.to_dict('records'),
                               available_dates=available_dates,
                               selected_date=selected_date,
                               selected_strategy=selected_strategy)
    except Exception as e:
        logger.error(f"选股池页面处理失败: {e}", exc_info=True)
        raise


@app.route('/trading_status')
@login_required
def trading_status():
    """交易状态页面"""
    # 获取选择的策略，默认使用止损止盈策略
    selected_strategy = request.args.get('strategy', '止损止盈策略')   
    selected_date = request.args.get('selected_date', '')  # 获取用户选择的日期
    print(f'[DEBUG] trading_status called with strategy: {selected_strategy}, selected_date: {selected_date}')
    print(f'[DEBUG] Request args: {dict(request.args)}')
    
    # 导入需要的库（确保导入在使用之前）
    from pathlib import Path
    import yaml
    
    # 获取当前文件的绝对路径
    current_file = Path(__file__).resolve()
    
    # 计算项目根目录
    base_dir = current_file.parents[2]
    
    # 读取配置文件
    config_path = base_dir / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取配置文件中的回测日期范围
    date_start = pd.to_datetime(config['backtest']['start_date'])
    date_end = pd.to_datetime(config['backtest']['end_date'])

    # 构建报告目录的绝对路径
    reports_dir = base_dir / 'reports'
    print(f'[DEBUG] Reports dir: {reports_dir}')

    # 关键修复：根据选择的策略正确加载对应的文件！
    trades_path = reports_dir / f'trades_mixed_{selected_strategy}.csv'
    daily_values_path = reports_dir / f'daily_values_mixed_{selected_strategy}.csv'
    print(f'[DEBUG] Trying to load trades from: {trades_path}')
    print(f'[DEBUG] Trying to load daily values from: {daily_values_path}')
    
    # 检查文件是否存在
    if not trades_path.exists():
        logger.error(f"交易文件不存在: {trades_path}")
    if not daily_values_path.exists():
        logger.error(f"每日价值文件不存在: {daily_values_path}")

    # 确保文件存在
    if not trades_path.exists():
        logger.error(f"交易记录文件不存在: {trades_path}")
        return render_template('trading_status.html', positions=[], latest_total_value=0, trades=[],
                               selected_strategy=selected_strategy)
    if not daily_values_path.exists():
        logger.error(f"每日价值文件不存在: {daily_values_path}")
        return render_template('trading_status.html', positions=[], latest_total_value=0, trades=[],
                               selected_strategy=selected_strategy)

    trades = load_csv_with_cache(str(trades_path), date_column='日期')
    print(f'[DEBUG] Loaded trades from {trades_path}: {len(trades)} records')

    # 根据配置文件中的日期范围过滤交易数据
    trades = trades[(trades['日期'] >= date_start) & (trades['日期'] <= date_end)]
    
    # 保存原始交易记录用于计算持仓（截至所选日期的所有交易）
    trades_for_position = trades.copy()
    
    # 如果用户选择了特定日期，则筛选当日交易用于显示
    if selected_date:
        selected_date_dt = pd.to_datetime(selected_date)
        # 只过滤用于显示的交易记录，保留完整记录用于计算持仓
        trades = trades[trades['日期'] == selected_date_dt]

    # 加载选股池数据，用于获取股票名称和收盘价
    pools_path = reports_dir / 'daily_pools.csv'
    daily_pools = None
    stock_name_map = {}
    stock_price_data = {}
    
    if pools_path.exists():
        daily_pools = load_csv_with_cache(str(pools_path))
        # 创建股票代码到名称的映射
        stock_name_map = dict(zip(daily_pools['股票代码'], daily_pools['名称']))
        
        # 创建股票代码到价格数据的映射（按日期索引）
        if not daily_pools.empty and '股票代码' in daily_pools.columns and '日期' in daily_pools.columns and '收盘价' in daily_pools.columns:
            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(daily_pools['日期']):
                daily_pools['日期'] = pd.to_datetime(daily_pools['日期'], errors='coerce')
            
            # 按股票代码分组，创建每个股票的价格数据框（按日期排序）
            for stock_code, group in daily_pools.groupby('股票代码'):
                # 按日期排序并创建日期索引
                sorted_group = group.sort_values('日期').set_index('日期')
                stock_price_data[stock_code] = sorted_group
    else:
        stock_name_map = {}
        stock_price_data = {}

    # 价格最小变动单位调整函数
    def adjust_price_by_tick(price, stock_code):
        """根据股票代码和价格应用价格最小变动单位规则

        Args:
            price: 原始价格
            stock_code: 股票代码

        Returns:
            float: 调整后的价格，符合最小变动单位要求
        """
        # A股价格最小变动单位规则：
        # 一般为0.01元人民币
        # 科创板/创业板某些高价股可能为0.05元

        # 根据股票代码判断板块：
        # 科创板股票代码以688开头，创业板以300开头
        # 这里简化处理：科创板/创业板股票价格>=100元时使用0.05元变动单位
        tick_size = 0.01

        # 判断是否为科创板或创业板股票
        stock_code_str = str(stock_code)
        if stock_code_str.startswith('688') or stock_code_str.startswith('300'):
            # 高价股使用0.05元变动单位
            if price >= 100:
                tick_size = 0.05

        # 应用最小变动单位调整，使用更精确的计算方法避免浮点数精度问题
        # 计算需要调整的倍数
        multiplier = round(price / tick_size)
        # 使用format函数确保结果的小数位数正确
        if tick_size == 0.01:
            # 对于0.01的变动单位，保留2位小数
            adjusted_price = float(f"{multiplier * tick_size:.2f}")
        elif tick_size == 0.05:
            # 对于0.05的变动单位，保留2位小数（0.05的倍数）
            adjusted_price = float(f"{multiplier * tick_size:.2f}")
        else:
            # 其他情况保留2位小数
            adjusted_price = float(f"{multiplier * tick_size:.2f}")
        return adjusted_price

    # 计算当前持仓，按照A股交易规则（100股一手）
    positions = []

    # A股交易规则：100股一手
    # 获取用户选择的日期
    selected_date = request.args.get('selected_date', '')

    # 确定需要处理的交易记录范围（全部或截止到选定日期）
    if selected_date:
        # 确保日期类型一致后再比较
        if pd.api.types.is_datetime64_any_dtype(trades_for_position['日期']):
            # 如果日期列是datetime类型，将selected_date转换为datetime
            selected_datetime = pd.to_datetime(selected_date)
            processing_trades = trades_for_position[trades_for_position['日期'] <= selected_datetime]
        else:
            # 如果日期列是字符串类型，直接比较
            processing_trades = trades_for_position[trades_for_position['日期'] <= selected_date]
    else:
        # 处理全部交易
        processing_trades = trades_for_position.copy()

    # 计算持仓数据
    positions = []
    if not processing_trades.empty and '股票代码' in processing_trades.columns and '类型' in processing_trades.columns and '股数' in processing_trades.columns:
        stock_codes = processing_trades['股票代码'].unique()
        A_SHARE_LOT_SIZE = 100

        for stock_code in stock_codes:
            stock_trades = processing_trades[processing_trades['股票代码'] == stock_code].sort_values('日期')

            # 按照A股交易规则调整交易股数（100股一手）
            # 买入和卖出的股数都向下取整到最接近的100的倍数
            buy_trades = stock_trades[stock_trades['类型'] == '买入'].copy()
            sell_trades = stock_trades[stock_trades['类型'] == '卖出'].copy()

            # 调整买入股数
            buy_trades['调整后股数'] = (buy_trades['股数'] // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE
            # 调整卖出股数
            sell_trades['调整后股数'] = (sell_trades['股数'] // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE

            # 计算总买入和总卖出股数
            total_buy = buy_trades['调整后股数'].sum()
            total_sell = sell_trades['调整后股数'].sum()

            # 如果没有有效交易，跳过
            if total_buy == 0:
                continue

            current_shares = total_buy - total_sell

            # 确保持仓数量是正的，且按照A股规则显示
            if current_shares > 0:
                # 获取最后一次有效买入的价格作为参考买入价格
                # 过滤掉调整后股数为0的买入记录
                valid_buy_trades = buy_trades[buy_trades['调整后股数'] > 0]
                if not valid_buy_trades.empty:
                    last_buy_price = valid_buy_trades['价格'].iloc[-1]
                    # 应用价格最小变动单位调整
                    last_buy_price = adjust_price_by_tick(last_buy_price, stock_code)
                else:
                    last_buy_price = 0
                # 获取用户选择日期的收盘价作为当前价格
                current_price = None
                
                # 确定用户选择的日期
                if selected_date:
                    selected_datetime = pd.to_datetime(selected_date)
                    
                    # 尝试从选股池数据中获取该日期的收盘价
                    if stock_code in stock_price_data and not stock_price_data[stock_code].empty:
                        price_data = stock_price_data[stock_code]
                        
                        # 查找指定日期的收盘价
                        if selected_datetime in price_data.index:
                            current_price = price_data.loc[selected_datetime, '收盘价']
                        else:
                            # 如果没有找到指定日期，查找最接近的前一个交易日的收盘价
                            # 获取所有在指定日期之前的价格数据
                            before_selected = price_data[price_data.index <= selected_datetime]
                            if not before_selected.empty:
                                # 取最后一个（最接近指定日期）的收盘价
                                current_price = before_selected.iloc[-1]['收盘价']
                
                # 如果没有找到合适的收盘价，使用最后一次交易的价格作为备选
                if current_price is None:
                    current_price = stock_trades['价格'].iloc[-1]
                
                # 应用价格最小变动单位调整
                current_price = adjust_price_by_tick(current_price, stock_code)

                # 计算收益率
                if last_buy_price > 0:
                    return_rate = (current_price - last_buy_price) / last_buy_price * 100
                else:
                    return_rate = 0

                # 使用股票名称映射获取实际股票名称，如果没有找到则使用股票代码
                stock_name = stock_name_map.get(stock_code, stock_code)

                # 计算持仓手数（100股为1手）
                lot_count = current_shares // A_SHARE_LOT_SIZE

                positions.append({
                    '股票代码': stock_code,
                    '名称': stock_name,
                    '持仓数量': current_shares,
                    '持仓手数': lot_count,
                    '买入价格': last_buy_price,
                    '当前价格': current_price,
                    '收益率(%)': round(return_rate, 2)
                })

    # 应用A股交易规则调整交易记录
    # A股交易规则：
    # 主板：100股一手，买卖数量应为1手或其整数倍
    # 科创板/创业板：200股一手，买卖数量应为1手或其整数倍
    # 卖出时，不足1手的部分（零股）需一次性申报卖出

    # 由于交易记录中没有板块信息，统一按照主板规则（100股一手）调整
    A_SHARE_LOT_SIZE = 100

    # 先按日期排序，以便按时间顺序处理交易
    trades_sorted = processing_trades.sort_values('日期', ascending=True).copy()

    # 创建一个字典来跟踪每个股票的持仓情况
    position_tracker = {}

    # 对每条交易记录应用A股交易规则
    for index, trade in trades_sorted.iterrows():
        stock_code = trade['股票代码']
        trade_type = trade['类型']
        original_shares = trade['股数']
        price = trade['价格']

        # 应用价格最小变动单位调整
        adjusted_price = adjust_price_by_tick(price, stock_code)

        # 初始化该股票的持仓记录
        if stock_code not in position_tracker:
            position_tracker[stock_code] = {
                'current_shares': 0,  # 当前持仓股数
                'adjusted_buys': [],  # 调整后的买入记录
                'adjusted_sells': []  # 调整后的卖出记录
            }

        current_position = position_tracker[stock_code]['current_shares']
        adjusted_shares = original_shares

        if trade_type == '买入':
            # 买入交易：股数向下取整为100的倍数
            adjusted_shares = (original_shares // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE
        elif trade_type == '卖出':
            # 卖出交易：先计算可卖出的整手数量
            # 零股部分必须一次性卖出
            if current_position <= 0:
                adjusted_shares = 0  # 没有持仓，无法卖出
            else:
                # 如果卖出数量超过持仓，只能卖出持仓数量
                if original_shares > current_position:
                    adjusted_shares = current_position
                else:
                    adjusted_shares = original_shares

                # 零股处理：如果剩余持仓是零股，必须一次性卖出
                # 这里简化处理，直接按100的倍数处理
                # 更准确的处理需要跟踪每笔买入的具体数量，这里为了简化，使用向下取整
                adjusted_shares = (adjusted_shares // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE

                # 如果调整后为0，但有持仓，可能是零股情况
                if adjusted_shares == 0 and current_position > 0:
                    # 卖出所有持仓（包括零股）
                    adjusted_shares = current_position

        # 更新持仓
        if trade_type == '买入' and adjusted_shares > 0:
            position_tracker[stock_code]['current_shares'] += adjusted_shares
        elif trade_type == '卖出' and adjusted_shares > 0:
            position_tracker[stock_code]['current_shares'] -= adjusted_shares

        # 修复：如果调整后股数为0，但原始股数不为0，使用原始股数
        # 这确保即使交易记录不完整，也能显示正确的股数和金额
        if adjusted_shares == 0 and original_shares > 0:
            adjusted_shares = original_shares

        # 更新交易记录中的调整后股数和调整后价格
        trades_sorted.loc[index, '调整后股数'] = adjusted_shares
        trades_sorted.loc[index, '调整后价格'] = adjusted_price

        # 重新计算调整后的金额（调整后股数 * 调整后价格）
        adjusted_amount = adjusted_shares * adjusted_price
        trades_sorted.loc[index, '调整后金额'] = adjusted_amount

    # === 新增：直接从 daily_values_*.csv 读取总资产 ===
    daily_values_path = reports_dir / f'daily_values_mixed_{selected_strategy}.csv'

    latest_total_value = 0  # 默认值
    if daily_values_path.exists():
        try:
            daily_values_df = pd.read_csv(daily_values_path, encoding='utf-8-sig')
            # 确保日期列是 datetime 类型
            if '日期' in daily_values_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(daily_values_df['日期']):
                    daily_values_df['日期'] = pd.to_datetime(daily_values_df['日期'], errors='coerce')
                
                # 转换用户选择的日期为 datetime
                if selected_date:
                    selected_date_dt = pd.to_datetime(selected_date)
                    # 查找 <= selected_date 的最新记录（因为可能不是每个日历日都有数据）
                    filtered_values = daily_values_df[daily_values_df['日期'] <= selected_date_dt]
                    if not filtered_values.empty:
                        latest_total_value = filtered_values.iloc[-1]['总价值']
                    else:
                        latest_total_value = 0
                else:
                    # 如果没有选择日期，使用最后一条记录
                    if not daily_values_df.empty:
                        latest_total_value = daily_values_df.iloc[-1]['总价值']
        except Exception as e:
            logger.error(f"读取 daily_values 失败: {e}")
            latest_total_value = 0
    else:
        logger.error(f"daily_values 文件不存在: {daily_values_path}")
        latest_total_value = 0
    # ==============================================

    # 准备交易记录数据，根据选择的日期筛选
    if selected_date:
        if pd.api.types.is_datetime64_any_dtype(trades_sorted['日期']):
            selected_datetime = pd.to_datetime(selected_date)
            trades_filtered = trades_sorted[trades_sorted['日期'] == selected_datetime]
        else:
            trades_filtered = trades_sorted[trades_sorted['日期'] == selected_date]
        trades_records = trades_filtered.sort_values('日期', ascending=False).to_dict('records')
    else:
        trades_records = trades_sorted.sort_values('日期', ascending=False).to_dict('records')

    # 处理交易记录中的日期格式，确保只显示YYYY-MM-DD
    for record in trades_records:
        if '日期' in record and record['日期'] is not None:
            # 处理datetime对象和字符串类型的日期
            if isinstance(record['日期'], pd.Timestamp) or isinstance(record['日期'], str):
                # 如果是Timestamp对象，格式化
                if isinstance(record['日期'], pd.Timestamp):
                    record['日期'] = record['日期'].strftime('%Y-%m-%d')
                # 如果是字符串，确保格式正确
                elif isinstance(record['日期'], str):
                    # 如果字符串包含时间部分，去掉
                    if ' ' in record['日期']:
                        record['日期'] = record['日期'].split(' ')[0]

    # 处理交易记录中的NaN值，将收益和收益率的NaN替换为0或空字符串
    for record in trades_records:
        # 处理收益字段
        if pd.isna(record['收益']):
            record['收益'] = 0
        else:
            record['收益'] = round(record['收益'], 2)

        # 处理收益率字段
        if pd.isna(record['收益率']):
            record['收益率'] = 0
        else:
            record['收益率'] = round(record['收益率'], 2)

        # 如果没有调整后股数，使用原始股数
        if '调整后股数' not in record or pd.isna(record['调整后股数']):
            record['调整后股数'] = record['股数']

        # 如果没有调整后金额，使用原始金额
        if '调整后金额' not in record or pd.isna(record['调整后金额']):
            record['调整后金额'] = record['金额']

        # 如果没有调整后剩余资金，使用原始剩余资金
        if '调整后剩余资金' not in record or pd.isna(record['调整后剩余资金']):
            record['调整后剩余资金'] = record['剩余资金']
    
    from flask import make_response
    
    response = make_response(render_template('trading_status.html', positions=positions, latest_total_value=latest_total_value, trades=trades_records, selected_strategy=selected_strategy, selected_date=selected_date))
    # 添加响应头，防止浏览器缓存
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/stock_detail/<stock_code>')
@login_required
def stock_detail(stock_code):
    """股票详情页面"""
    try:
        from pathlib import Path
        import datetime
        import pandas as pd
        import plotly.graph_objects as go
        import logging

        # 获取用户选择的日期
        selected_date = request.args.get('selected_date', '')
        logger.info(f"股票详情页面请求，股票代码: {stock_code}, 选择日期: {selected_date}")

        # 获取当前文件的绝对路径
        current_file = Path(__file__).resolve()

        # 计算项目根目录
        base_dir = current_file.parents[2]

        # 构建报告目录的绝对路径
        reports_dir = base_dir / 'reports'

        # 加载选股池数据
        pools_path = reports_dir / 'daily_pools.csv'
        daily_pools = None
        stock_data = None
        latest_stock_data = {}

        if pools_path.exists():
            daily_pools = load_csv_with_cache(str(pools_path), date_column='日期')

            # 根据股票代码筛选数据
            stock_data = daily_pools[daily_pools['股票代码'] == int(stock_code)]

            if not stock_data.empty:
                # 如果用户选择了日期，获取该日期的数据
                if selected_date:
                    selected_datetime = pd.to_datetime(selected_date)
                    # 查找指定日期的数据
                    date_stock_data = stock_data[stock_data['日期'] == selected_datetime]
                    if not date_stock_data.empty:
                        latest_stock_data = date_stock_data.iloc[0].to_dict()
                        logger.info(f"找到指定日期 {selected_date} 的股票数据")
                    else:
                        # 如果没有找到指定日期，查找最接近的前一个交易日的数据
                        before_selected = stock_data[stock_data['日期'] <= selected_datetime]
                        if not before_selected.empty:
                            latest_stock_data = before_selected.iloc[-1].to_dict()
                            logger.info(f"未找到指定日期 {selected_date} 的数据，使用最接近的前一个交易日数据")
                        else:
                            # 如果没有找到，使用最新数据
                            latest_stock_data = stock_data.sort_values('日期', ascending=False).iloc[0].to_dict()
                            logger.warning(f"未找到指定日期 {selected_date} 的数据，使用最新数据")
                else:
                    # 获取最新的一条数据
                    latest_stock_data = stock_data.sort_values('日期', ascending=False).iloc[0].to_dict()
                    logger.info("未选择日期，使用最新数据")
            else:
                logger.warning(f"未找到股票代码 {stock_code} 的选股池数据，将仅显示K线图")
        else:
            logger.warning(f"选股池文件不存在: {pools_path}，将仅显示K线图")

        # 生成K线图
        fig = go.Figure()
        kline_generated = False
        stock_df_for_indicators = None  # 用于计算技术指标的原始数据

        # 从原始股票数据文件中加载完整的历史数据
        try:
            from pathlib import Path
            
            # 动态计算项目根目录
            current_file = Path(__file__).resolve()
            base_dir = current_file.parents[2]  # app.py位于src/website/, 所以向上两级
            logger.info(f"使用的base_dir: {base_dir}")
            
            # 检查目录是否存在
            logger.info(f"1-500目录是否存在: {(base_dir / '1-500').exists()}")
            logger.info(f"501-1000目录是否存在: {(base_dir / '501-1000').exists()}")
            
            # 查找对应的股票数据文件
            # 确保股票代码为6位数字格式（补前导零）
            stock_code_str = str(stock_code).zfill(6)
            stock_file = None
            file_path_1 = base_dir / '1-500' / f'{stock_code_str}.csv'
            file_path_2 = base_dir / '501-1000' / f'{stock_code_str}.csv'
            
            logger.info(f"检查文件: {file_path_1}, 存在: {file_path_1.exists()}")
            logger.info(f"检查文件: {file_path_2}, 存在: {file_path_2.exists()}")
            
            # 检查1-500目录
            if file_path_1.exists():
                stock_file = str(file_path_1)
            else:
                # 检查501-1000目录
                if file_path_2.exists():
                    stock_file = str(file_path_2)
            
            if stock_file:
                logger.info(f"找到股票数据文件: {stock_file}")
            else:
                logger.error(f"未找到股票数据文件: {stock_code_str} 在 {base_dir / '1-500'} 和 {base_dir / '501-1000'}")
                # 文件不存在，显示提示信息
                plot_html = '<div style="display: flex; justify-content: center; align-items: center; height: 600px; font-size: 18px; color: #666;">未找到该股票的历史数据文件，无法显示K线图</div>'
                return render_template('stock_detail.html', stock_code=stock_code, stock_data=latest_stock_data, plot_html=plot_html)
            
            if stock_file:
                logger.info(f"开始加载股票数据文件: {stock_file}")
                # 加载原始数据，尝试不同的编码
                encodings = ['utf-8-sig', 'gbk', 'gb2312', 'utf-16']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(stock_file, encoding=encoding)
                        logger.info(f"成功加载文件: {stock_file}, 编码: {encoding}, 列名: {list(df.columns)}, 行数: {len(df)}")
                        break
                    except Exception as e:
                        logger.warning(f"使用编码 {encoding} 加载文件失败: {stock_file}, 错误: {e}")
                        continue
                
                if df is None:
                    logger.error(f"无法使用任何编码加载文件: {stock_file}")
                    # 数据加载失败，显示提示信息
                    plot_html = '<div style="display: flex; justify-content: center; align-items: center; height: 600px; font-size: 18px; color: #666;">无法加载股票数据文件，请稍后重试</div>'
                    return render_template('stock_detail.html', stock_code=stock_code, stock_data=latest_stock_data, plot_html=plot_html)
                
                logger.info(f"数据加载成功，列名: {list(df.columns)}")
                
                # 保存原始数据用于计算技术指标
                stock_df_for_indicators = df.copy()
                
                # 处理日期格式
                date_columns = ['交易日期', '日期']
                date_column = None
                
                for col in date_columns:
                    if col in df.columns:
                        date_column = col
                        logger.info(f"找到日期列: {date_column}")
                        break
                
                if date_column:
                    # 尝试不同的日期格式
                    date_formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']
                    parsed = False
                    
                    for fmt in date_formats:
                        try:
                            df[date_column] = pd.to_datetime(df[date_column], format=fmt)
                            logger.info(f"成功解析日期格式: {fmt}")
                            parsed = True
                            break
                        except Exception:
                            continue
                    
                    if not parsed:
                        # 尝试自动解析
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                        logger.info(f"使用自动解析日期")
                    
                    df = df.sort_values(date_column)
                    logger.info(f"数据已按日期排序，最新日期: {df[date_column].max()}")
                else:
                    logger.warning("未找到日期列")
                
                # 确定K线图显示的日期范围
                if selected_date:
                    # 如果用户选择了日期，K线图显示该日期前30天的数据
                    selected_datetime = pd.to_datetime(selected_date)
                    # 计算起始日期（30个交易日之前）
                    # 使用工作日计算，排除周末和节假日
                    start_date = selected_datetime - pd.Timedelta(days=45)  # 多计算一些天数以确保有30个交易日
                    
                    # 筛选指定日期范围内的数据
                    stock_history = df[(df[date_column] >= start_date) & (df[date_column] <= selected_datetime)]
                    
                    # 如果数据超过30天，只取最后30天
                    if len(stock_history) > 30:
                        stock_history = stock_history.tail(30)
                    
                    logger.info(f"获取指定日期 {selected_date} 前30天的数据，行数: {len(stock_history)}")
                else:
                    # 如果用户没有选择日期，获取最近30天的数据
                    stock_history = df.tail(30)
                    logger.info(f"获取最近30天数据，行数: {len(stock_history)}")
                
                if len(stock_history) > 0:
                    # 确保数据包含K线所需的列
                    price_columns = ['开盘价', '最高价', '最低价', '收盘价']
                    logger.info(f"检查价格列: {price_columns}")
                    logger.info(f"数据包含的列: {list(stock_history.columns)}")
                    
                    if all(col in stock_history.columns for col in price_columns) and date_column:
                        logger.info("所有必要列都存在，开始生成K线图")
                        fig.add_trace(go.Candlestick(
                            x=stock_history[date_column],
                            open=stock_history['开盘价'],
                            high=stock_history['最高价'],
                            low=stock_history['最低价'],
                            close=stock_history['收盘价'],
                            name=f'{stock_code} K线'
                        ))

                        # 更新K线图标题，显示正确的日期范围
                        if selected_date:
                            kline_title = f'{stock_code} {selected_date} 前30天K线图'
                        else:
                            kline_title = f'{stock_code} 最近30天K线图'
                        
                        fig.update_layout(
                            title=kline_title,
                            xaxis_title='日期',
                            yaxis_title='价格',
                            hovermode='x unified',
                            height=600  # 设置图表高度
                        )
                        logger.info("K线图生成成功")
                        kline_generated = True
                    else:
                        logger.error(f"股票数据缺少必要的K线图列: 价格列={price_columns}, 日期列={date_column}")
                        logger.error(f"数据实际包含的列: {list(stock_history.columns)}")
                else:
                    logger.error(f"股票数据不足30天: {stock_file}")
        except Exception as e:
            logger.error(f"生成K线图失败: {e}", exc_info=True)
        
        # 只有在成功生成K线图时才返回图表，否则返回空字符串
        if kline_generated:
            plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            logger.info("成功生成K线图HTML")
        else:
            plot_html = ''
            logger.info("未生成K线图，返回空字符串")

        # 如果选股池数据中没有找到用户选择日期的数据，尝试从原始数据中获取并计算
        if not latest_stock_data and stock_df_for_indicators is not None and selected_date:
            logger.info(f"选股池中未找到指定日期 {selected_date} 的数据，尝试从原始数据中获取")
            
            try:
                # 确保日期列已转换为datetime类型
                date_columns = ['交易日期', '日期']
                date_column = None
                
                for col in date_columns:
                    if col in stock_df_for_indicators.columns:
                        date_column = col
                        break
                
                if date_column:
                    # 转换日期格式
                    if not pd.api.types.is_datetime64_any_dtype(stock_df_for_indicators[date_column]):
                        date_formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']
                        parsed = False
                        
                        for fmt in date_formats:
                            try:
                                stock_df_for_indicators[date_column] = pd.to_datetime(stock_df_for_indicators[date_column], format=fmt)
                                parsed = True
                                break
                            except Exception:
                                continue
                        
                        if not parsed:
                            stock_df_for_indicators[date_column] = pd.to_datetime(stock_df_for_indicators[date_column], errors='coerce')
                    
                    # 查找指定日期的数据
                    selected_datetime = pd.to_datetime(selected_date)
                    date_stock_data = stock_df_for_indicators[stock_df_for_indicators[date_column] == selected_datetime]
                    
                    if not date_stock_data.empty:
                        # 找到指定日期的数据，计算技术指标
                        stock_data_at_date = date_stock_data.iloc[0].to_dict()
                        
                        # 获取该日期之前的数据用于计算技术指标
                        historical_data = stock_df_for_indicators[stock_df_for_indicators[date_column] <= selected_datetime].copy()
                        historical_data = historical_data.sort_values(date_column)
                        
                        # 计算移动平均线
                        if len(historical_data) >= 5:
                            historical_data['MA5'] = historical_data['收盘价'].rolling(window=5).mean()
                        if len(historical_data) >= 10:
                            historical_data['MA10'] = historical_data['收盘价'].rolling(window=10).mean()
                        if len(historical_data) >= 20:
                            historical_data['MA20'] = historical_data['收盘价'].rolling(window=20).mean()
                        
                        # 计算RSI14
                        if len(historical_data) >= 15:
                            delta = historical_data['收盘价'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            historical_data['RSI14'] = 100 - (100 / (1 + rs))
                        
                        # 计算BIAS10
                        if len(historical_data) >= 11:
                            historical_data['BIAS10'] = ((historical_data['收盘价'] - historical_data['收盘价'].rolling(window=10).mean()) / historical_data['收盘价'].rolling(window=10).mean()) * 100
                        
                        # 计算成交量变化率
                        if len(historical_data) >= 2:
                            historical_data['成交量变化率'] = historical_data['成交量(手)'].pct_change() * 100
                        
                        # 计算涨跌幅
                        if len(historical_data) >= 2:
                            historical_data['涨跌幅(%)'] = historical_data['收盘价'].pct_change() * 100
                        
                        # 计算波动率
                        if len(historical_data) >= 6:
                            historical_data['5日波动率'] = historical_data['收盘价'].pct_change().rolling(window=5).std() * 100
                        
                        # 获取指定日期的计算结果
                        if len(historical_data) > 0:
                            latest_row = historical_data.iloc[-1]
                            
                            # 合并基本面数据和技术指标
                            for col in ['MA5', 'MA10', 'MA20', 'RSI14', 'BIAS10', '成交量变化率', '5日波动率']:
                                if col in latest_row and pd.notna(latest_row[col]):
                                    stock_data_at_date[col] = latest_row[col]
                            
                            # 计算涨跌幅（如果原始数据中没有）
                            if '涨跌幅(%)' not in stock_data_at_date or pd.isna(stock_data_at_date['涨跌幅(%)']):
                                if len(historical_data) >= 2:
                                    stock_data_at_date['涨跌幅(%)'] = latest_row['涨跌幅(%)']
                            
                            latest_stock_data = stock_data_at_date
                            logger.info(f"从原始数据中获取并计算了指定日期 {selected_date} 的股票数据")
                    else:
                        logger.warning(f"原始数据中也未找到指定日期 {selected_date} 的数据")
            except Exception as e:
                logger.error(f"从原始数据中获取数据失败: {e}", exc_info=True)

        return render_template('stock_detail.html',
                               stock_code=stock_code,
                               stock_data=latest_stock_data,
                               plot_html=plot_html)
    except Exception as e:
        logger.error(f"加载股票详情失败: {e}", exc_info=True)
        return render_template('stock_detail.html', stock_code=stock_code, stock_data={}, plot_html='')


@app.route('/trading_snapshot/<date_str>/<strategy>')
@login_required
def trading_snapshot(date_str, strategy):
    """查询指定日期和策略的交易状态快照
    
    Args:
        date_str: 日期字符串（格式：YYYYMMDD）
        strategy: 策略名称（英文或中文）
    """
    try:
        from pathlib import Path
        import json
        
        # 获取当前文件的绝对路径
        current_file = Path(__file__).resolve()
        
        # 计算项目根目录
        base_dir = current_file.parents[2]
        
        # 构建报告目录的绝对路径
        reports_dir = base_dir / 'reports'
        
        # 映射策略名
        sell_key = STRATEGY_NAME_MAP.get(strategy, strategy)
        
        # 构建快照文件路径
        snapshot_file = reports_dir / f"snapshot_mixed_{sell_key}_{date_str}.json"
        
        if not snapshot_file.exists():
            logger.warning(f"快照文件不存在: {snapshot_file}")
            return {"error": "No data for this date", "message": f"快照文件不存在: {snapshot_file}"}, 404
        
        # 读取快照数据
        with open(snapshot_file, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
        
        # 获取当日选股池（从 daily_pools.csv 过滤）
        pools_df = None
        pools_path = reports_dir / "daily_pools.csv"
        if pools_path.exists():
            pools_df = load_csv_with_cache(str(pools_path))
            if not pools_df.empty and '日期' in pools_df.columns:
                # 转换日期格式
                if not pd.api.types.is_datetime64_any_dtype(pools_df['日期']):
                    pools_df['日期'] = pd.to_datetime(pools_df['日期'], errors='coerce')
                # 筛选当日选股池
                date_dt = pd.to_datetime(date_str, format='%Y%m%d')
                today_pools = pools_df[pools_df['日期'] == date_dt]
                snapshot['pools'] = today_pools.to_dict('records') if not today_pools.empty else []
            else:
                snapshot['pools'] = []
        else:
            snapshot['pools'] = []
        
        # 获取当日交易记录
        trades_df = None
        trades_path = reports_dir / f"trades_mixed_{sell_key}.csv"
        if trades_path.exists():
            trades_df = load_csv_with_cache(str(trades_path), date_column='日期')
            if not trades_df.empty:
                # 筛选当日交易记录
                date_dt = pd.to_datetime(date_str, format='%Y%m%d')
                today_trades = trades_df[trades_df['日期'] == date_dt]
                snapshot['trades'] = today_trades.to_dict('records') if not today_trades.empty else []
            else:
                snapshot['trades'] = []
        else:
            snapshot['trades'] = []
        
        return snapshot
        
    except Exception as e:
        logger.error(f"查询交易状态快照失败: {e}", exc_info=True)
        return {"error": "Internal server error", "message": str(e)}, 500


@app.route('/kline/<stock_code>/<end_date>')
@login_required
def get_kline(stock_code, end_date):
    """获取股票K线图数据
    
    Args:
        stock_code: 股票代码
        end_date: 结束日期（格式：YYYYMMDD）
    """
    try:
        from pathlib import Path
        
        # 获取当前文件的绝对路径
        current_file = Path(__file__).resolve()
        
        # 计算项目根目录
        base_dir = current_file.parents[2]
        
        # 查找股票数据文件
        stock_code_str = str(stock_code).zfill(6)
        stock_file = None
        
        # 检查1-500目录
        file_path_1 = base_dir / '1-500' / f'{stock_code_str}.csv'
        if file_path_1.exists():
            stock_file = str(file_path_1)
        else:
            # 检查501-1000目录
            file_path_2 = base_dir / '501-1000' / f'{stock_code_str}.csv'
            if file_path_2.exists():
                stock_file = str(file_path_2)
        
        if not stock_file:
            logger.error(f"未找到股票数据文件: {stock_code_str}")
            return {"error": "Stock data not found", "message": f"未找到股票数据文件: {stock_code_str}"}, 404
        
        # 加载股票数据
        encodings = ['utf-8-sig', 'gbk', 'gb2312', 'utf-16']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(stock_file, encoding=encoding)
                break
            except Exception:
                continue
        
        if df is None:
            logger.error(f"无法加载股票数据文件: {stock_file}")
            return {"error": "Failed to load stock data", "message": f"无法加载股票数据文件: {stock_file}"}, 500
        
        # 处理日期格式
        date_columns = ['交易日期', '日期']
        date_column = None
        
        for col in date_columns:
            if col in df.columns:
                date_column = col
                break
        
        if not date_column:
            logger.error(f"股票数据中没有日期列: {list(df.columns)}")
            return {"error": "No date column", "message": "股票数据中没有日期列"}, 500
        
        # 尝试不同的日期格式
        date_formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']
        parsed = False
        
        for fmt in date_formats:
            try:
                df[date_column] = pd.to_datetime(df[date_column], format=fmt)
                parsed = True
                break
            except Exception:
                continue
        
        if not parsed:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # 筛选指定日期之前的数据，取最后30条
        end_date_dt = pd.to_datetime(end_date, format='%Y%m%d')
        df = df[df[date_column] <= end_date_dt].tail(30)
        
        # 确保数据包含K线所需的列
        price_columns = ['开盘价', '最高价', '最低价', '收盘价']
        if not all(col in df.columns for col in price_columns):
            logger.error(f"股票数据缺少必要的K线图列: {price_columns}")
            return {"error": "Missing price columns", "message": f"股票数据缺少必要的K线图列"}, 500
        
        # 准备返回数据
        kline_data = df[[date_column] + price_columns + ['成交量(手)']].copy()
        kline_data.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
        kline_data['日期'] = kline_data['日期'].dt.strftime('%Y-%m-%d')
        
        return kline_data.to_dict('records')
        
    except Exception as e:
        logger.error(f"获取K线图数据失败: {e}", exc_info=True)
        return {"error": "Internal server error", "message": str(e)}, 500


@app.route('/backtest_strategy_comparison')
@login_required
def backtest_strategy_comparison():
    """三种卖出策略的回测对比页面"""
    # 使用pathlib处理路径，更现代、更可靠
    from pathlib import Path
    import yaml
    import json
    import pandas as pd

    # 获取当前文件的绝对路径
    current_file = Path(__file__).resolve()

    # 计算项目根目录
    # app.py位于src/website/目录下，所以需要向上走2级目录
    base_dir = current_file.parents[2]

    # 读取配置文件
    config_path = base_dir / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取配置文件中的回测日期范围
    date_start = pd.to_datetime(config['backtest']['start_date'])
    date_end = pd.to_datetime(config['backtest']['end_date'])

    try:
        # 导入所需模块
        from trading.trading_system import TradingSystem
        from strategy.stock_selector import StockSelector
        from data_processing.data_loader import StockDataLoader
        from data_processing.feature_engineering import FeatureEngineer

        # 检查是否存在已保存的策略对比结果文件
        reports_dir = base_dir / 'reports'
        daily_values_file = reports_dir / 'strategy_comparison_daily_values.csv'
        metrics_file = reports_dir / 'strategy_comparison_metrics.csv'
        
        if daily_values_file.exists() and metrics_file.exists():
            logger.info("发现已保存的策略对比结果文件，直接读取")
            
            # 读取每日价值数据
            combined_daily_values = pd.read_csv(daily_values_file, parse_dates=['日期'])
            
            # 读取性能指标数据
            metrics_comparison = pd.read_csv(metrics_file).to_dict('records')
            
            # 准备策略名称和对应的返回值 - 处理CSV文件中的列名问题
            # 检查列名并处理可能的空格
            if ' 策略名称' in combined_daily_values.columns:
                # 如果列名前面有空格，将其重命名为没有空格的列名
                combined_daily_values = combined_daily_values.rename(columns={' 策略名称': '策略名称'})
            
            ordered_strategy_names = sorted(combined_daily_values['策略名称'].unique().tolist())
            ordered_cumulative_returns = []
            
            # 添加调试日志
            logger.info(f"策略名称列表: {ordered_strategy_names}")
            logger.info(f"策略数量: {len(ordered_strategy_names)}")
            
            for strategy_name in ordered_strategy_names:
                strategy_daily_values = combined_daily_values[combined_daily_values['策略名称'] == strategy_name].copy()
                # 计算累计收益率 - 处理CSV文件中的列名问题
                if '累 计收益率' in strategy_daily_values.columns:
                    strategy_daily_values['累计收益率'] = strategy_daily_values['累 计收益率']
                else:
                    strategy_daily_values['累计收益率'] = (strategy_daily_values['总价值'] / 1000000 - 1) * 100
                ordered_cumulative_returns.append(strategy_daily_values)
                logger.info(f"{strategy_name}的记录数量: {len(strategy_daily_values)}")
        else:
            logger.info("未发现已保存的策略对比结果文件，执行回测")
            
            # 创建数据加载器和特征工程师
            data_loader = StockDataLoader(
                base_path=config['data_path']['base_path'],
                directories=config['data_path']['directories']
            )
            feature_engineer = FeatureEngineer()
            
            # 加载和处理股票数据
            stock_data_dict = data_loader.load_all_stocks()
            processed_stock_data = {}
            
            for stock_code, df in stock_data_dict.items():
                if df is not None and not df.empty:
                    processed_df = feature_engineer.process_stock_data(df)
                    if processed_df is not None and not processed_df.empty:
                        processed_stock_data[stock_code] = processed_df

            # 创建选股器
            stock_selector = StockSelector(config)

            # 创建交易系统
            trading_system = TradingSystem(config)

            # 执行三种卖出策略的对比回测
            comparison_results = trading_system.backtest_all_sell_strategies(processed_stock_data, stock_selector, date_start, date_end)

            # 准备策略性能指标对比数据
            metrics_comparison = []
            # 同时收集策略名称和对应的返回值，确保顺序一致
            ordered_strategy_names = []
            ordered_cumulative_returns = []
            
            for strategy_name, strategy_data in comparison_results.items():
                if 'result' in strategy_data and '每日价值' in strategy_data['result']:
                    ordered_strategy_names.append(strategy_name)
                    daily_values = pd.DataFrame(strategy_data['result']['每日价值'])
                    daily_values['累计收益率'] = (daily_values['总价值'] / 1000000 - 1) * 100
                    ordered_cumulative_returns.append(daily_values)
                    
                    final_value = daily_values['总价值'].iloc[-1]
                    total_return = (final_value / 1000000 - 1) * 100
                    
                    # 计算年化收益率
                    days = len(daily_values)
                    if days > 0:
                        annualized_return = ((final_value / 1000000) ** (365 / days) - 1) * 100
                    else:
                        annualized_return = 0
                    
                    # 计算最大回撤
                    daily_values['峰值'] = daily_values['总价值'].cummax()
                    daily_values['回撤'] = (daily_values['总价值'] - daily_values['峰值']) / daily_values['峰值'] * 100
                    max_drawdown = daily_values['回撤'].min()
                    
                    metrics_comparison.append({
                        '策略名称': strategy_name,
                        '总收益率(%)': round(total_return, 2),
                        '年化收益率(%)': round(annualized_return, 2),
                        '最大回撤(%)': round(max_drawdown, 2)
                    })

        # 创建累计收益率对比图
        fig = go.Figure()
        for i, (strategy_name, returns) in enumerate(zip(ordered_strategy_names, ordered_cumulative_returns)):
            fig.add_trace(go.Scatter(
                x=returns['日期'],
                y=returns['累计收益率'],
                mode='lines',
                name=strategy_name
            ))

        fig.update_layout(
            title='三种卖出策略累计收益率对比',
            xaxis_title='日期',
            yaxis_title='累计收益率(%)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        comparison_plot_html = fig.to_html(full_html=False)

        # 保存策略对比结果到CSV文件
        import pandas as pd
        import os
        
        # 构建报告目录的绝对路径
        reports_dir = base_dir / 'reports'
        
        # 1. 保存每日价值数据
        all_daily_values = []
        for strategy_name, returns in zip(ordered_strategy_names, ordered_cumulative_returns):
            daily_values = returns.copy()
            daily_values['策略名称'] = strategy_name
            all_daily_values.append(daily_values)
        
        combined_daily_values = pd.concat(all_daily_values, ignore_index=True)
        daily_values_file = reports_dir / 'strategy_comparison_daily_values.csv'
        combined_daily_values.to_csv(daily_values_file, index=False, encoding='utf-8-sig')
        logger.info(f"成功保存策略对比每日价值数据到: {daily_values_file}")
        
        # 2. 保存策略性能指标
        metrics_df = pd.DataFrame(metrics_comparison)
        metrics_file = reports_dir / 'strategy_comparison_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False, encoding='utf-8-sig')
        logger.info(f"成功保存策略对比性能指标到: {metrics_file}")
        
        return render_template('strategy_comparison.html',
                               comparison_plot_html=comparison_plot_html,
                               metrics_comparison=metrics_comparison,
                               strategies=json.dumps(ordered_strategy_names))
    except Exception as e:
        logger.error(f"执行策略对比回测时出错: {e}", exc_info=True)
        return render_template('strategy_comparison.html',
                               error_message=f"执行策略对比回测时出错: {str(e)}")


@app.route('/backtest_results')
def backtest_results():
    """回测结果页面"""
    # 获取用户选择的策略，如果没有则使用选股混合策略
    selected_strategy = request.args.get('strategy', 'mixed')

    # 使用pathlib处理路径，更现代、更可靠
    from pathlib import Path
    import yaml

    # 获取当前文件的绝对路径
    current_file = Path(__file__).resolve()

    # 计算项目根目录
    # app.py位于src/website/目录下，所以需要向上走2级目录
    base_dir = current_file.parents[2]

    # 读取配置文件
    config_path = base_dir / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取配置文件中的回测日期范围
    date_start = pd.to_datetime(config['backtest']['start_date'])
    date_end = pd.to_datetime(config['backtest']['end_date'])

    # 构建报告目录的绝对路径
    reports_dir = base_dir / 'reports'

    # 根据选择的策略加载不同的回测结果数据
    if selected_strategy == 'all':
        daily_values_path = reports_dir / 'daily_values.csv'
        trades_path = reports_dir / 'trades.csv'
    else:
        # 处理交易策略的文件名构建：选股策略为 mixed + 交易策略名
        if selected_strategy in ['止损止盈策略', '时间基础策略', 'RSI超买策略']:
            actual_strategy_name = f"mixed_{selected_strategy}"
        elif selected_strategy == '选股混合策略':
            actual_strategy_name = 'mixed'
        else:
            actual_strategy_name = selected_strategy
        
        daily_values_path = reports_dir / f'daily_values_{actual_strategy_name}.csv'
        trades_path = reports_dir / f'trades_{actual_strategy_name}.csv'

    # 添加调试信息
    print(f"selected_strategy: {selected_strategy}")
    print(f"actual_strategy_name: {actual_strategy_name}")
    print(f"base_dir: {base_dir}")
    print(f"reports_dir: {reports_dir}")
    print(f"daily_values_path: {daily_values_path}")
    print(f"File exists: {daily_values_path.exists()}")

    # 确保文件存在
    if not daily_values_path.exists():
        logger.error(f"每日价值文件不存在: {daily_values_path}")
        # 创建空数据以避免模板渲染错误
        daily_values = pd.DataFrame(columns=['日期', '总价值', '累计收益率', '已实现收益率', '回撤'])

        # 回测指标默认值
        backtest_metrics = {
            '总收益率': 0,
            '年化收益率': 0,
            '最大回撤': 0,
            '夏普比率': 0,
            '胜率': 0
        }

        # 空图表
        fig = go.Figure()
        fig.update_layout(title=f'{selected_strategy}策略数据',
                          xaxis_title='日期',
                          yaxis_title='数据',
                          annotations=[dict(text='暂无数据', x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False,
                                            font=dict(size=20))])
        empty_plot_html = fig.to_html(full_html=False)

        return render_template('backtest_results.html',
                               plot_html_cumulative=empty_plot_html,
                               plot_html_realized=empty_plot_html,
                               plot_html_returns_dist=empty_plot_html,
                               plot_html_capital=empty_plot_html,
                               plot_html_drawdown=empty_plot_html,
                               daily_values=[],
                               metrics=backtest_metrics,
                               selected_strategy=selected_strategy)

    daily_values = load_csv_with_cache(str(daily_values_path), date_column='日期')

    # # 根据配置文件中的日期范围过滤数据（暂时注释，因为测试数据日期是2023年）
    # daily_values = daily_values[(daily_values['日期'] >= date_start) & (daily_values['日期'] <= date_end)]

    daily_values['累计收益率'] = (daily_values['总价值'] / 1000000 - 1) * 100

    # 计算已实现收益率
    realized_return = 0
    realized_returns = []

    if trades_path.exists(): 
        trades = load_csv_with_cache(str(trades_path), date_column='日期')

        # 安全检查：确保trades DataFrame非空且包含必要的列
        if not trades.empty and '日期' in trades.columns:
            # 根据配置文件中的日期范围过滤交易数据
            trades = trades[(trades['日期'] >= date_start) & (trades['日期'] <= date_end)]

            # 筛选出所有卖出交易并按日期排序
            if '类型' in trades.columns:
                sell_trades = trades[trades['类型'] == '卖出'].sort_values('日期')

                # 创建一个字典用于存储每天的已实现收益
                daily_realized = {}
                for _, trade in sell_trades.iterrows():
                    date_str = trade['日期'].strftime('%Y-%m-%d')
                    if date_str not in daily_realized:
                        daily_realized[date_str] = 0
                    # 累加每日的已实现收益
                    if '收益' in trade:
                        daily_realized[date_str] += trade['收益']

                # 为每一天计算累计已实现收益率
                for _, row in daily_values.iterrows():
                    date_str = row['日期'].strftime('%Y-%m-%d')
                    if date_str in daily_realized:
                        realized_return += daily_realized[date_str]
                    # 计算已实现收益率（相对于初始资金）
                    realized_return_rate = (realized_return / 1000000) * 100
                    realized_returns.append(realized_return_rate)
            else:
                # 如果没有类型列，已实现收益率为0
                realized_returns = [0] * len(daily_values)
        else:
            # 如果trades是空的或没有日期列，已实现收益率为0
            realized_returns = [0] * len(daily_values)
    else:
        # 如果没有交易记录，已实现收益率为0
        realized_returns = [0] * len(daily_values)

    # 将已实现收益率添加到每日价值数据中
    daily_values['已实现收益率'] = realized_returns

    # 计算关键回测指标
    total_return = daily_values['累计收益率'].iloc[-1] if len(daily_values) > 0 else 0

    # 计算年化收益率（假设一年252个交易日）
    if len(daily_values) > 0:
        days = (daily_values['日期'].iloc[-1] - daily_values['日期'].iloc[0]).days
        annual_return = ((daily_values['总价值'].iloc[-1] / daily_values['总价值'].iloc[0]) ** (365 / days) - 1) * 100
    else:
        annual_return = 0

    # 计算最大回撤
    if len(daily_values) > 0:
        daily_values['峰值'] = daily_values['总价值'].cummax()
        daily_values['回撤'] = (daily_values['总价值'] - daily_values['峰值']) / daily_values['峰值'] * 100
        max_drawdown = daily_values['回撤'].min()
    else:
        max_drawdown = 0

    # 计算日收益率
    daily_values['日收益率'] = daily_values['总价值'].pct_change() * 100
    daily_returns = daily_values['日收益率'].dropna()

    # 计算夏普比率（假设无风险利率为3%）
    if len(daily_returns) > 0:
        risk_free_rate = 3 / 252  # 日无风险利率
        sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * (252 ** 0.5)
    else:
        sharpe_ratio = 0

    # 计算胜率（正收益天数占比）
    if len(daily_returns) > 0:
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
    else:
        win_rate = 0

    # 生成累计收益率曲线
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=daily_values['日期'], y=daily_values['累计收益率'],
                                        mode='lines+markers', name='累计收益率(%)',
                                        hovertemplate='日期: %{x}<br>累计收益率: %{y:.2f}%<br>总价值: %{customdata:.2f}',
                                        customdata=daily_values['总价值']))

    # 优化累计收益率图表布局
    min_y_cumulative = daily_values['累计收益率'].min() * 1.1
    max_y_cumulative = daily_values['累计收益率'].max() * 1.1

    fig_cumulative.update_layout(
        title=f'{selected_strategy}策略累计收益率曲线',
        xaxis_title='日期',
        yaxis_title='收益率(%)',
        yaxis=dict(
            range=[min_y_cumulative, max_y_cumulative],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    # 生成已实现收益率曲线
    fig_realized = go.Figure()
    fig_realized.add_trace(go.Scatter(x=daily_values['日期'], y=daily_values['已实现收益率'],
                                      mode='lines+markers', name='已实现收益率(%)',
                                      line=dict(dash='dash', color='red'),
                                      hovertemplate='日期: %{x}<br>已实现收益率: %{y:.2f}%'))

    # 优化已实现收益率图表布局
    min_y_realized = daily_values['已实现收益率'].min() * 1.1
    max_y_realized = daily_values['已实现收益率'].max() * 1.1

    fig_realized.update_layout(
        title=f'{selected_strategy}策略已实现收益率曲线',
        xaxis_title='日期',
        yaxis_title='收益率(%)',
        yaxis=dict(
            range=[min_y_realized, max_y_realized],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    # 转换为HTML
    plot_html_cumulative = fig_cumulative.to_html(full_html=False)
    plot_html_realized = fig_realized.to_html(full_html=False)

    # 准备回测详情数据
    backtest_details = daily_values.sort_values('日期').to_dict('records')

    # 生成收益分布直方图
    fig_returns_dist = go.Figure()
    fig_returns_dist.add_trace(go.Histogram(x=daily_values['日收益率'].dropna(),
                                            nbinsx=30,
                                            name='日收益率分布',
                                            hovertemplate='收益率区间: %{x:.2f}%<br>天数: %{y}<br>频率: %{y}/%{xbin.count}'))

    fig_returns_dist.update_layout(
        title=f'{selected_strategy}策略日收益率分布',
        xaxis_title='日收益率(%)',
        yaxis_title='天数',
        hovermode='closest',
        bargap=0.1
    )

    # 生成资金曲线
    fig_capital = go.Figure()
    fig_capital.add_trace(go.Scatter(x=daily_values['日期'], y=daily_values['总价值'],
                                     mode='lines+markers', name='总资金(元)',
                                     hovertemplate='日期: %{x}<br>总资金: %{y:.2f}元<extra></extra>'))

    fig_capital.update_layout(
        title=f'{selected_strategy}策略资金曲线',
        xaxis_title='日期',
        yaxis_title='总资金(元)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        hovermode='x unified'
    )

    # 生成回撤曲线
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(x=daily_values['日期'], y=daily_values['回撤'],
                                      mode='lines+markers', name='每日回撤(%)',
                                      hovertemplate='日期: %{x}<br>回撤: %{y:.2f}%<extra></extra>'))

    fig_drawdown.update_layout(
        title=f'{selected_strategy}策略回撤曲线',
        xaxis_title='日期',
        yaxis_title='回撤(%)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        hovermode='x unified'
    )

    # 转换所有图表为HTML
    plot_html_cumulative = fig_cumulative.to_html(full_html=False)
    plot_html_realized = fig_realized.to_html(full_html=False)
    plot_html_returns_dist = fig_returns_dist.to_html(full_html=False)
    plot_html_capital = fig_capital.to_html(full_html=False)
    plot_html_drawdown = fig_drawdown.to_html(full_html=False)

    # 回测指标
    backtest_metrics = {
        '总收益率': round(total_return, 2),
        '年化收益率': round(annual_return, 2),
        '最大回撤': round(max_drawdown, 2),
        '夏普比率': round(sharpe_ratio, 2),
        '胜率': round(win_rate, 2)
    }

    return render_template('backtest_results.html',
                           plot_html_cumulative=plot_html_cumulative,
                           plot_html_realized=plot_html_realized,
                           plot_html_returns_dist=plot_html_returns_dist,
                           plot_html_capital=plot_html_capital,
                           plot_html_drawdown=plot_html_drawdown,
                           daily_values=backtest_details,
                           metrics=backtest_metrics,
                           selected_strategy=selected_strategy)





@app.route('/export_backtest_results', methods=['GET'])
def export_backtest_results():
    """导出回测结果为CSV文件"""
    import io
    from pathlib import Path

    # 获取用户选择的策略
    selected_strategy = request.args.get('strategy', 'mixed')

    # 获取当前文件的绝对路径
    current_file = Path(__file__).resolve()
    # 计算项目根目录
    base_dir = current_file.parents[2]
    # 构建报告目录的绝对路径
    reports_dir = base_dir / 'reports'

    # 根据选择的策略加载不同的回测结果数据
    if selected_strategy == 'all':
        daily_values_path = reports_dir / 'daily_values.csv'
        trades_path = reports_dir / 'trades.csv'
    else:
        # 处理交易策略的文件名构建：选股策略为 mixed + 交易策略名
        if selected_strategy in ['止损止盈策略', '时间基础策略', 'RSI超买策略']:
            actual_strategy_name = f"mixed_{selected_strategy}"
        else:
            actual_strategy_name = selected_strategy
        
        daily_values_path = reports_dir / f'daily_values_{actual_strategy_name}.csv'
        trades_path = reports_dir / f'trades_{actual_strategy_name}.csv'

    # 确保文件存在
    if not daily_values_path.exists():
        raise FileNotFoundError(f"每日价值文件不存在: {daily_values_path}")

    daily_values = pd.read_csv(daily_values_path)
    daily_values['日期'] = pd.to_datetime(daily_values['日期'])
    daily_values['累计收益率'] = (daily_values['总价值'] / 1000000 - 1) * 100

    # 计算回撤
    if not daily_values.empty:
        daily_values['峰值'] = daily_values['总价值'].cummax()
        daily_values['回撤'] = (daily_values['总价值'] - daily_values['峰值']) / daily_values['峰值'] * 100

    # 计算日收益率
    daily_values['日收益率'] = daily_values['总价值'].pct_change() * 100

    # 创建CSV文件的内存缓冲区
    output = io.StringIO()

    # 写入回测结果数据
    output.write("# 回测结果数据\n")
    daily_values.to_csv(output, index=False, encoding='utf-8')

    # 如果有交易记录，也添加到CSV文件中
    if trades_path.exists():
        output.write("\n# 交易记录\n")
        trades = pd.read_csv(trades_path)
        trades.to_csv(output, index=False, encoding='utf-8')

    # 重置缓冲区位置
    output.seek(0)

    # 生成文件名
    filename = f"backtest_results_{selected_strategy}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # 返回CSV文件
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )


def run_website(config):
    """运行网站应用

    Args:
        config: 配置字典
    """
    host = config['website']['host']
    port = config['website']['port']
    # 启用调试模式，便于调试错误
    debug = True

    # 设置secret key
    app.config['SECRET_KEY'] = config['website']['secret_key']

    logger.info(f"启动网站应用，地址: http://{host}:{port}")
    logger.info(f"调试模式: {debug}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    from pathlib import Path
    import yaml

    # 加载配置
    config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    run_website(config)
