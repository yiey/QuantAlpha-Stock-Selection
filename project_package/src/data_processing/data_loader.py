import os
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataLoader:
    def __init__(self, base_path, directories=None):
        self.base_path = base_path
        self.directories = directories if directories else ['1-500', '501-1000']  # 默认加载前两个目录
        self.stock_directories = []
        
        # 验证目录是否存在
        for directory in self.directories:
            dir_path = os.path.join(self.base_path, directory)
            if os.path.exists(dir_path):
                self.stock_directories.append(directory)
            else:
                logger.warning(f"目录 {dir_path} 不存在，将跳过")
    
    def load_single_stock(self, file_path):
        """加载单个股票数据文件"""
        try:
            df = pd.read_csv(file_path)
            
            # 处理日期格式
            if '交易日期' in df.columns:
                df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y%m%d')
            
            # 按交易日期排序
            if '交易日期' in df.columns:
                df = df.sort_values('交易日期')
            
            return df
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            return None
    
    def load_all_stocks(self):
        """加载所有股票数据"""
        stock_data = {}
        
        for directory in self.stock_directories:
            directory_path = os.path.join(self.base_path, directory)
            stock_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
            
            for stock_file in tqdm(stock_files, desc=f"加载 {directory} 目录下的股票数据"):
                stock_file_path = os.path.join(directory_path, stock_file)
                stock_code = stock_file.split('.')[0]  # 从文件名提取股票代码
                
                df = self.load_single_stock(stock_file_path)
                if df is not None:
                    stock_data[stock_code] = df
        
        logger.info(f"成功加载 {len(stock_data)} 支股票的数据")
        return stock_data
    
    def get_stock_data_by_date(self, stock_data, date):
        """获取指定日期的所有股票数据"""
        date_data = []
        
        for stock_code, df in stock_data.items():
            try:
                # 获取指定日期的数据
                date_df = df[df['交易日期'] == date].copy()
                if not date_df.empty:
                    date_df['股票代码'] = stock_code
                    date_data.append(date_df)
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 的 {date} 数据时出错: {str(e)}")
        
        if date_data:
            return pd.concat(date_data, ignore_index=True)
        else:
            logger.warning(f"日期 {date} 没有股票数据")
            return pd.DataFrame()
    
    def get_all_trading_dates(self, stock_data):
        """获取所有股票的交易日期并去重排序"""
        all_dates = set()
        
        for df in stock_data.values():
            if '交易日期' in df.columns:
                all_dates.update(df['交易日期'].tolist())
        
        return sorted(list(all_dates))
