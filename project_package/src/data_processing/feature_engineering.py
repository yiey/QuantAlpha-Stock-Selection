import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        # 定义需要使用的基础特征
        self.base_features = [
            '市盈率', '市净率', '市销率', '股息率(%)',  # 基本面指标
            '涨跌幅(%)', '换手率(%)', '量比'  # 技术面指标
        ]
        
    def clean_data(self, df):
        """数据清洗：处理缺失值和异常值"""
        df_clean = df.copy()
        
        # 1. 处理缺失值
        # 对数值型特征使用均值填充或滚动均值填充
        numeric_features = df_clean.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if feature in df_clean.columns:
                # 如果缺失值比例小于20%，使用滚动均值填充（需要按股票代码分组）
                if df_clean[feature].isnull().sum() / len(df_clean) < 0.2:
                    df_clean[feature] = df_clean.groupby('股票代码')[feature].transform(
                        lambda x: x.fillna(x.rolling(5, min_periods=1).mean())
                    )
                # 否则使用均值填充
                df_clean[feature] = df_clean[feature].fillna(df_clean[feature].mean())
        
        # 2. 处理异常值 - 使用3σ法则去除极端值
        for feature in numeric_features:
            if feature in df_clean.columns:
                mean = df_clean[feature].mean()
                std = df_clean[feature].std()
                # 将超过3σ的值替换为均值
                df_clean[feature] = np.where(
                    (df_clean[feature] > mean + 3 * std) | (df_clean[feature] < mean - 3 * std),
                    mean,
                    df_clean[feature]
                )
        
        return df_clean
    
    def generate_features(self, df):
        """为单只股票生成特征"""
        try:
            df_enhanced = df.copy()
            
            # 计算移动平均线（5日、10日、20日）
            for period in [5, 10, 20]:
                df_enhanced[f'MA{period}'] = df_enhanced['收盘价'].rolling(period).mean()
            
            # 计算价格动量（5日涨跌幅）
            df_enhanced['5日涨跌幅'] = df_enhanced['涨跌幅(%)'].rolling(5).mean()
            
            # 计算成交量变化率
            df_enhanced['成交量变化率'] = df_enhanced['成交量(手)'].pct_change(fill_method=None)
            
            # 计算相对强弱指标（RSI）
            df_enhanced['RSI14'] = self.calculate_rsi(df_enhanced['收盘价'], 14)
            df_enhanced['rsi_10'] = self.calculate_rsi(df_enhanced['收盘价'], 10)
            
            # 计算乖离率（BIAS）
            df_enhanced['BIAS10'] = (df_enhanced['收盘价'] - df_enhanced['MA10']) / df_enhanced['MA10'] * 100
            
            # 计算波动率（5日收益率标准差）
            df_enhanced['5日波动率'] = df_enhanced['涨跌幅(%)'].rolling(5).std()
            
            # 计算收益率（returns_5d, returns_10d）
            df_enhanced['returns_5d'] = df_enhanced['收盘价'].pct_change(periods=5, fill_method=None)
            df_enhanced['returns_10d'] = df_enhanced['收盘价'].pct_change(periods=10, fill_method=None)
            
            # 计算动量（momentum_5d, momentum_10d）
            df_enhanced['momentum_5d'] = df_enhanced['收盘价'] / df_enhanced['收盘价'].shift(5) - 1
            df_enhanced['momentum_10d'] = df_enhanced['收盘价'] / df_enhanced['收盘价'].shift(10) - 1
            
            # 计算MA比率（ma5_ma10_ratio, ma10_ratio, ma20_ratio）
            df_enhanced['ma5_ma10_ratio'] = df_enhanced['MA5'] / df_enhanced['MA10']
            df_enhanced['ma10_ratio'] = df_enhanced['收盘价'] / df_enhanced['MA10']
            df_enhanced['ma20_ratio'] = df_enhanced['收盘价'] / df_enhanced['MA20']
            
            # 计算成交量比率（volume_ratio_10d, volume_ratio_20d）
            df_enhanced['volume_ratio_10d'] = df_enhanced['成交量(手)'] / df_enhanced['成交量(手)'].rolling(10).mean()
            df_enhanced['volume_ratio_20d'] = df_enhanced['成交量(手)'] / df_enhanced['成交量(手)'].rolling(20).mean()
            
            # 计算换手率相关特征（turnover_ratio_10d, turnover_ratio_20d, turnover_ma5）
            df_enhanced['turnover_ratio_10d'] = df_enhanced['换手率(%)'] / df_enhanced['换手率(%)'].rolling(10).mean()
            df_enhanced['turnover_ratio_20d'] = df_enhanced['换手率(%)'] / df_enhanced['换手率(%)'].rolling(20).mean()
            df_enhanced['turnover_ma5'] = df_enhanced['换手率(%)'].rolling(5).mean()
            
            # 计算股息率移动平均（dividend_ma5）
            df_enhanced['dividend_ma5'] = df_enhanced['股息率(%)'].rolling(5).mean()
            
            # 计算MACD指标
            df_enhanced['macd_hist'] = self.calculate_macd(df_enhanced['收盘价'])
            
            # 计算布林带位置（bollinger_position）
            df_enhanced['bollinger_position'] = self.calculate_bollinger_position(df_enhanced['收盘价'], 20)
            
            return df_enhanced
        except Exception as e:
            logger.error(f"为股票生成特征时出错: {str(e)}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """计算相对强弱指标（RSI）"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD柱状图"""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return macd_hist
    
    def calculate_bollinger_position(self, prices, period=20, num_std=2):
        """计算布林带位置（价格在布林带中的位置）"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bollinger_position = (prices - lower_band) / (upper_band - lower_band)
        return bollinger_position
    
    def select_features(self, df, target_feature):
        """特征选择：基于相关性分析选择重要特征"""
        # 计算特征相关性矩阵
        corr_matrix = df.corr()
        
        # 获取与目标特征相关性较高的特征
        target_corr = corr_matrix[target_feature].abs().sort_values(ascending=False)
        
        # 选择相关性大于0.1的特征
        selected_features = target_corr[target_corr > 0.1].index.tolist()
        
        # 移除目标特征本身
        if target_feature in selected_features:
            selected_features.remove(target_feature)
        
        logger.info(f"选择了 {len(selected_features)} 个特征: {selected_features}")
        return selected_features
    
    def normalize_features(self, df, features):
        """特征归一化：使用Min-Max归一化"""
        df_normalized = df.copy()
        
        for feature in features:
            if feature in df_normalized.columns:
                min_val = df_normalized[feature].min()
                max_val = df_normalized[feature].max()
                
                # 避免除以0
                if max_val - min_val != 0:
                    df_normalized[feature] = (df_normalized[feature] - min_val) / (max_val - min_val)
        
        return df_normalized
    
    def process_stock_data(self, df):
        """处理股票数据：清洗数据并生成特征"""
        try:
            # 清洗数据
            df_clean = self.clean_data(df)
            
            # 生成特征
            df_enhanced = self.generate_features(df_clean)
            
            return df_enhanced
        except Exception as e:
            logger.error(f"处理股票数据时出错: {str(e)}")
            return None
