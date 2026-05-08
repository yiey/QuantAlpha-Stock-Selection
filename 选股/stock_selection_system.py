import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, data_dir: str, start_date: str = '20250501'):
        self.data_dir = Path(data_dir)
        self.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        self.stock_data = {}
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        for folder in ['1-500', '501-1000']:
            folder_path = self.data_dir / folder
            if not folder_path.exists():
                continue
                
            for csv_file in folder_path.glob('*.csv'):
                stock_code = csv_file.stem
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8')
                    df = self._preprocess(df)
                    if len(df) > 0:
                        self.stock_data[stock_code] = df
                except Exception as e:
                    print(f"Error loading {stock_code}: {e}")
                    continue
        
        print(f"Loaded {len(self.stock_data)} stocks")
        return self.stock_data
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y%m%d')
        df = df[df['交易日期'] >= self.start_date].copy()
        df = df.sort_values('交易日期').reset_index(drop=True)
        
        numeric_cols = ['开盘价', '最高价', '最低价', '收盘价', '前收盘价',
                       '涨跌额', '涨跌幅(%)', '成交量(手)', '成交额(千元)',
                       '换手率(%)', '换手率(自由流通股)', '量比', '市盈率',
                       '市盈率(TTM,亏损的PE为空)', '市净率', '市销率', '市销率(TTM)',
                       '股息率(%)', '股息率(TTM)(%)', '总股本(万股)', '流通股本(万股)',
                       '自由流通股本(万股)', '总市值(万元)', '流通市值(万元)',
                       '今日涨停价', '今日跌停价']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_all_data(self) -> pd.DataFrame:
        all_data = []
        for stock_code, df in self.stock_data.items():
            df_copy = df.copy()
            df_copy['股票代码'] = stock_code
            all_data.append(df_copy)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result
        return pd.DataFrame()


class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)
        
        df = self._price_features(df)
        df = self._volume_features(df)
        df = self._momentum_features(df)
        df = self._volatility_features(df)
        df = self._fundamental_features(df)
        
        df = df.dropna()
        return df
    
    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['收盘价']
        high = df['最高价']
        low = df['最低价']
        open_price = df['开盘价']
        volume = df['成交量(手)']
        
        df['returns_1d'] = close.pct_change(1)
        df['returns_5d'] = close.pct_change(5)
        df['returns_10d'] = close.pct_change(10)
        df['returns_20d'] = close.pct_change(20)
        
        df['high_low_ratio'] = high / low
        df['close_open_ratio'] = close / open_price
        df['close_high_ratio'] = close / high
        df['close_low_ratio'] = close / low
        
        df['ma5'] = close.rolling(5).mean()
        df['ma10'] = close.rolling(10).mean()
        df['ma20'] = close.rolling(20).mean()
        df['ma60'] = close.rolling(60).mean()
        
        df['ma5_ratio'] = close / df['ma5']
        df['ma10_ratio'] = close / df['ma10']
        df['ma20_ratio'] = close / df['ma20']
        df['ma60_ratio'] = close / df['ma60']
        
        df['ma5_ma10_ratio'] = df['ma5'] / df['ma10']
        df['ma10_ma20_ratio'] = df['ma10'] / df['ma20']
        df['ma20_ma60_ratio'] = df['ma20'] / df['ma60']
        
        df['bollinger_upper'] = df['ma20'] + 2 * close.rolling(20).std()
        df['bollinger_lower'] = df['ma20'] - 2 * close.rolling(20).std()
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['ma20']
        df['bollinger_position'] = (close - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        price_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
                      'high_low_ratio', 'close_open_ratio', 'close_high_ratio', 'close_low_ratio',
                      'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 'ma60_ratio',
                      'ma5_ma10_ratio', 'ma10_ma20_ratio', 'ma20_ma60_ratio',
                      'bollinger_width', 'bollinger_position']
        
        for col in price_cols:
            df[col] = df.groupby('股票代码')[col].shift(1)
        
        return df
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        volume = df['成交量(手)']
        turnover = df['换手率(%)']
        amount = df['成交额(千元)']
        
        df['volume_ma5'] = volume.rolling(5).mean()
        df['volume_ma10'] = volume.rolling(10).mean()
        df['volume_ma20'] = volume.rolling(20).mean()
        
        df['volume_ratio_5d'] = volume / df['volume_ma5']
        df['volume_ratio_10d'] = volume / df['volume_ma10']
        df['volume_ratio_20d'] = volume / df['volume_ma20']
        
        df['turnover_ma5'] = turnover.rolling(5).mean()
        df['turnover_ma10'] = turnover.rolling(10).mean()
        df['turnover_ma20'] = turnover.rolling(20).mean()
        
        df['turnover_ratio_5d'] = turnover / df['turnover_ma5']
        df['turnover_ratio_10d'] = turnover / df['turnover_ma10']
        df['turnover_ratio_20d'] = turnover / df['turnover_ma20']
        
        df['amount_ma5'] = amount.rolling(5).mean()
        df['amount_ma10'] = amount.rolling(10).mean()
        df['amount_ratio_5d'] = amount / df['amount_ma5']
        
        df['volume_price_trend'] = (volume * df['收盘价']).rolling(5).sum() / volume.rolling(5).sum()
        
        volume_cols = ['volume_ratio_5d', 'volume_ratio_10d', 'volume_ratio_20d',
                       'turnover_ratio_5d', 'turnover_ratio_10d', 'turnover_ratio_20d',
                       'amount_ratio_5d', 'volume_price_trend']
        
        for col in volume_cols:
            df[col] = df.groupby('股票代码')[col].shift(1)
        
        return df
    
    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['收盘价']
        
        df['momentum_5d'] = close / close.shift(5) - 1
        df['momentum_10d'] = close / close.shift(10) - 1
        df['momentum_20d'] = close / close.shift(20) - 1
        df['momentum_60d'] = close / close.shift(60) - 1
        
        df['rsi_5'] = self._calculate_rsi(close, 5)
        df['rsi_10'] = self._calculate_rsi(close, 10)
        df['rsi_20'] = self._calculate_rsi(close, 20)
        
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(close)
        
        momentum_cols = ['momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
                         'rsi_5', 'rsi_10', 'rsi_20', 'macd', 'macd_signal', 'macd_hist']
        
        for col in momentum_cols:
            df[col] = df.groupby('股票代码')[col].shift(1)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['收盘价']
        returns = close.pct_change()
        
        df['volatility_5d'] = returns.rolling(5).std()
        df['volatility_10d'] = returns.rolling(10).std()
        df['volatility_20d'] = returns.rolling(20).std()
        
        df['atr_14'] = self._calculate_atr(df, 14)
        
        df['high_low_std_5d'] = (df['最高价'] - df['最低价']).rolling(5).std()
        df['high_low_std_10d'] = (df['最高价'] - df['最低价']).rolling(10).std()
        
        volatility_cols = ['volatility_5d', 'volatility_10d', 'volatility_20d',
                           'atr_14', 'high_low_std_5d', 'high_low_std_10d']
        
        for col in volatility_cols:
            df[col] = df.groupby('股票代码')[col].shift(1)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high = df['最高价']
        low = df['最低价']
        close = df['收盘价']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pe = df['市盈率']
        pb = df['市净率']
        ps = df['市销率']
        pe_ttm = df['市盈率(TTM,亏损的PE为空)']
        ps_ttm = df['市销率(TTM)']
        dividend = df['股息率(%)']
        dividend_ttm = df['股息率(TTM)(%)']
        market_cap = df['总市值(万元)']
        
        df['pe_ma5'] = pe.rolling(5).mean()
        df['pe_ratio_ma5'] = pe / df['pe_ma5']
        
        df['pb_ma5'] = pb.rolling(5).mean()
        df['pb_ratio_ma5'] = pb / df['pb_ma5']
        
        df['ps_ma5'] = ps.rolling(5).mean()
        df['ps_ratio_ma5'] = ps / df['ps_ma5']
        
        df['market_cap_log'] = np.log(market_cap)
        df['market_cap_rank'] = df.groupby('交易日期')['market_cap_log'].rank(pct=True)
        
        df['dividend_ma5'] = dividend.rolling(5).mean()
        df['dividend_ttm_ma5'] = dividend_ttm.rolling(5).mean()
        
        fundamental_cols = ['pe_ratio_ma5', 'pb_ratio_ma5', 'ps_ratio_ma5',
                           'market_cap_log', 'market_cap_rank', 'dividend_ma5', 'dividend_ttm_ma5']
        
        for col in fundamental_cols:
            df[col] = df.groupby('股票代码')[col].shift(1)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude_cols = ['股票代码', '名称', '所属行业', '地域', '上市日期', 'TS代码', '交易日期',
                       '开盘价', '最高价', '最低价', '收盘价', '前收盘价', '涨跌额', '涨跌幅(%)',
                       '成交量(手)', '成交额(千元)', '换手率(%)', '换手率(自由流通股)', '量比',
                       '市盈率', '市盈率(TTM,亏损的PE为空)', '市净率', '市销率', '市销率(TTM)',
                       '股息率(%)', '股息率(TTM)(%)', '总股本(万股)', '流通股本(万股)',
                       '自由流通股本(万股)', '总市值(万元)', '流通市值(万元)',
                       '今日涨停价', '今日跌停价', '复权因子']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols


class FeatureSelector:
    def __init__(self, ic_threshold: float = 0.02, ir_threshold: float = 0.5):
        self.ic_threshold = ic_threshold
        self.ir_threshold = ir_threshold
        self.ic_results = {}
        self.selected_features = []
    
    def calculate_ic(self, df: pd.DataFrame, feature_cols: List[str], 
                     target_col: str = 'future_return') -> pd.DataFrame:
        ic_results = []
        
        for date in df['交易日期'].unique():
            date_data = df[df['交易日期'] == date]
            if len(date_data) < 10:
                continue
            
            for feature in feature_cols:
                if feature not in date_data.columns:
                    continue
                
                valid_data = date_data[[feature, target_col]].dropna()
                if len(valid_data) < 5:
                    continue
                
                try:
                    ic = valid_data[feature].corr(valid_data[target_col])
                    if pd.isna(ic):
                        continue
                    ic_results.append({
                        '交易日期': date,
                        'feature': feature,
                        'ic': ic
                    })
                except:
                    continue
        
        ic_df = pd.DataFrame(ic_results)
        return ic_df
    
    def calculate_ir(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        ic_summary = ic_df.groupby('feature')['ic'].agg(['mean', 'std', 'count'])
        ic_summary['ir'] = ic_summary['mean'] / ic_summary['std']
        ic_summary['ic_mean_abs'] = ic_summary['mean'].abs()
        
        return ic_summary
    
    def select_features(self, df: pd.DataFrame, feature_cols: List[str],
                        target_col: str = 'future_return') -> List[str]:
        ic_df = self.calculate_ic(df, feature_cols, target_col)
        ic_summary = self.calculate_ir(ic_df)
        
        self.ic_results = ic_summary
        
        selected = ic_summary[
            (ic_summary['ic_mean_abs'] >= self.ic_threshold) & 
            (ic_summary['ir'].abs() >= self.ir_threshold)
        ].index.tolist()
        
        self.selected_features = selected
        print(f"Selected {len(selected)} features out of {len(feature_cols)}")
        print(f"IC threshold: {self.ic_threshold}, IR threshold: {self.ir_threshold}")
        
        return selected
    
    def get_ic_report(self) -> pd.DataFrame:
        return self.ic_results.sort_values('ic_mean_abs', ascending=False)


class StockModel:
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = {}
        
    def prepare_target(self, df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(['股票代码', '交易日期'])
        
        df['future_return'] = df.groupby('股票代码')['收盘价'].pct_change(forward_days).shift(-forward_days)
        
        return df
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.model.fit(X_train, y_train)
            
            self.feature_importance = dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
        
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            self.feature_importance = dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
        
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            self.feature_importance = dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
        
        print(f"Model trained: {self.model_type}")
    
    def predict_daily(self, df: pd.DataFrame, selected_features: List[str], 
                    forward_days: int = 5) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(['股票代码', '交易日期']).reset_index(drop=True)
        
        unique_dates = sorted(df['交易日期'].unique())
        predictions = []
        
        for i, date in enumerate(unique_dates):
            if i < 60:
                continue
            
            train_data = df[df['交易日期'] < date].copy()
            test_data = df[df['交易日期'] == date].copy()
            
            if len(train_data) < 100 or len(test_data) < 10:
                continue
            
            train_data = train_data.dropna(subset=['future_return'])
            
            X_train = train_data[selected_features]
            y_train = train_data['future_return']
            
            X_test = test_data[selected_features]
            
            self.train(X_train, y_train)
            pred = self.predict(X_test)
            
            test_data['predicted_return'] = pred
            predictions.append(test_data[['股票代码', '交易日期', '名称', '收盘价', 
                                        'predicted_return', '市盈率', '市净率', 
                                        '换手率(%)', '总市值(万元)']])
        
        if predictions:
            result = pd.concat(predictions, ignore_index=True)
            return result
        return pd.DataFrame()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance.keys()),
            'importance': list(self.feature_importance.values())
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df


class ReportGenerator:
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_ic_report(self, ic_results: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        ic_report = ic_results.copy()
        ic_report['ic_rank'] = ic_report['ic_mean_abs'].rank(ascending=False)
        ic_report = ic_report.sort_values('ic_mean_abs', ascending=False)
        
        if save_path:
            ic_report.to_csv(save_path, index=True, encoding='utf-8-sig')
            print(f"IC report saved to {save_path}")
        
        return ic_report
    
    def generate_feature_importance_report(self, importance_df: pd.DataFrame, 
                                             save_path: str = None) -> pd.DataFrame:
        importance_report = importance_df.copy()
        importance_report['rank'] = range(1, len(importance_report) + 1)
        importance_report['importance_pct'] = importance_report['importance'] / importance_report['importance'].sum() * 100
        
        if save_path:
            importance_report.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"Feature importance report saved to {save_path}")
        
        return importance_report
    
    def generate_daily_selection_pool(self, predictions_df: pd.DataFrame, 
                                   top_n: int = 50, save_path: str = None) -> pd.DataFrame:
        if predictions_df.empty:
            print("No predictions to generate selection pool")
            return pd.DataFrame()
        
        selection_pools = []
        
        for date in sorted(predictions_df['交易日期'].unique()):
            date_data = predictions_df[predictions_df['交易日期'] == date].copy()
            top_stocks = date_data.nlargest(top_n, 'predicted_return')
            top_stocks['rank'] = range(1, len(top_stocks) + 1)
            selection_pools.append(top_stocks)
        
        if selection_pools:
            result = pd.concat(selection_pools, ignore_index=True)
            result = result[['rank', '交易日期', '股票代码', '名称', '收盘价', 
                           'predicted_return', '市盈率', '市净率', '换手率(%)', '总市值(万元)']]
            
            if save_path:
                result.to_csv(save_path, index=False, encoding='utf-8-sig')
                print(f"Daily selection pool saved to {save_path}")
            
            return result
        return pd.DataFrame()
    
    def generate_selection_pool(self, df: pd.DataFrame, predictions: np.ndarray,
                                top_n: int = 50, save_path: str = None) -> pd.DataFrame:
        df_copy = df.copy()
        
        latest_date = df_copy['交易日期'].max()
        latest_data = df_copy[df_copy['交易日期'] == latest_date].copy()
        latest_data['predicted_return'] = predictions
        
        selection_pool = latest_data.nlargest(top_n, 'predicted_return')
        
        selection_pool = selection_pool[['股票代码', '名称', '收盘价', 'predicted_return', 
                                          '市盈率', '市净率', '换手率(%)', '总市值(万元)']]
        
        if save_path:
            selection_pool.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"Selection pool saved to {save_path}")
        
        return selection_pool
    
    def visualize_ic(self, ic_results: pd.DataFrame, top_n: int = 20, save_path: str = None):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        top_features = ic_results.nlargest(top_n, 'ic_mean_abs')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' for x in top_features['ic_mean_abs']]
        ax.barh(range(len(top_features)), top_features['ic_mean_abs'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.set_xlabel('IC Mean', fontsize=12)
        ax.set_title(f'Top {top_n} Features by IC Mean', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"IC visualization saved to {save_path}")
        plt.show()
        plt.close()
    
    def visualize_feature_importance(self, importance_df: pd.DataFrame, 
                                     top_n: int = 20, save_path: str = None):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance visualization saved to {save_path}")
        plt.show()
        plt.close()
    
    def visualize_ir(self, ic_results: pd.DataFrame, top_n: int = 20, save_path: str = None):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        top_features = ic_results.nlargest(top_n, 'ir')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' for x in top_features['ir']]
        ax.barh(range(len(top_features)), top_features['ir'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.set_xlabel('IR (Information Ratio)', fontsize=12)
        ax.set_title(f'Top {top_n} Features by IR', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"IR visualization saved to {save_path}")
        plt.show()
        plt.close()


class StockSelectionSystem:
    def __init__(self, data_dir: str, start_date: str = '20250501'):
        self.data_loader = DataLoader(data_dir, start_date)
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.model = StockModel()
        self.report_generator = ReportGenerator()
        self.data = None
        self.feature_data = None
    
    def run(self, ic_threshold: float = 0.02, ir_threshold: float = 0.5,
            model_type: str = 'lightgbm', top_n: int = 50):
        print("=" * 60)
        print("股票选股系统启动")
        print("=" * 60)
        
        print("\n[1/6] 加载数据...")
        self.data_loader.load_all_stocks()
        self.data = self.data_loader.get_all_data()
        print(f"Total records: {len(self.data)}")
        
        print("\n[2/6] 构建特征...")
        self.feature_data = self.feature_engineer.create_features(self.data)
        print(f"Feature data shape: {self.feature_data.shape}")
        
        print("\n[3/6] 准备目标变量...")
        self.feature_data = self.model.prepare_target(self.feature_data, forward_days=5)
        self.feature_data = self.feature_data.dropna(subset=['future_return'])
        print(f"Data after target preparation: {len(self.feature_data)}")
        
        print("\n[4/6] 特征选择 (IC/IR筛选)...")
        feature_cols = self.feature_engineer.get_feature_columns(self.feature_data)
        self.feature_selector.ic_threshold = ic_threshold
        self.feature_selector.ir_threshold = ir_threshold
        selected_features = self.feature_selector.select_features(
            self.feature_data, feature_cols, 'future_return'
        )
        
        if len(selected_features) == 0:
            print("No features selected. Lowering thresholds...")
            self.feature_selector.ic_threshold = 0.01
            self.feature_selector.ir_threshold = 0.3
            selected_features = self.feature_selector.select_features(
                self.feature_data, feature_cols, 'future_return'
            )
        
        print(f"Selected features: {selected_features}")
        
        print("\n[5/6] 训练模型...")
        train_data = self.feature_data[self.feature_data['交易日期'] < self.feature_data['交易日期'].max()]
        
        X_train = train_data[selected_features]
        y_train = train_data['future_return']
        
        self.model.model_type = model_type
        self.model.train(X_train, y_train)
        
        print("\n[6/6] 生成报告和选股池...")
        latest_data = self.feature_data[self.feature_data['交易日期'] == self.feature_data['交易日期'].max()]
        X_latest = latest_data[selected_features]
        predictions = self.model.predict(X_latest)
        
        ic_report = self.feature_selector.get_ic_report()
        importance_report = self.model.get_feature_importance()
        
        self.report_generator.generate_ic_report(
            ic_report, 
            self.report_generator.output_dir / 'ic_report.csv'
        )
        self.report_generator.generate_feature_importance_report(
            importance_report,
            self.report_generator.output_dir / 'feature_importance_report.csv'
        )
        self.report_generator.generate_selection_pool(
            self.feature_data,
            predictions,
            top_n,
            self.report_generator.output_dir / 'selection_pool.csv'
        )
        
        self.report_generator.visualize_ic(
            ic_report,
            top_n=20,
            save_path=self.report_generator.output_dir / 'ic_visualization.png'
        )
        self.report_generator.visualize_feature_importance(
            importance_report,
            top_n=20,
            save_path=self.report_generator.output_dir / 'feature_importance_visualization.png'
        )
        self.report_generator.visualize_ir(
            ic_report,
            top_n=20,
            save_path=self.report_generator.output_dir / 'ir_visualization.png'
        )
        
        print("\n" + "=" * 60)
        print("选股系统运行完成！")
        print(f"报告已保存至: {self.report_generator.output_dir}")
        print("=" * 60)
        
        return ic_report, importance_report
