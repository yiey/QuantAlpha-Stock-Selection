# QuantAlpha - 量化阿尔法选股系统

&gt; 一个专业的量化选股与交易系统，通过机器学习算法和多因子分析模型挖掘市场阿尔法收益。

## 📖 项目简介

**QuantAlpha** 是一个专业的量化投资研究平台，专注于通过多因子模型和机器学习算法挖掘市场阿尔法收益。系统整合了数据处理、因子工程、IC/IR因子分析、模型训练、策略回测等完整量化研究流程，并提供Web界面进行可视化展示。

## ✨ 核心特性

### 🔧 数据处理引擎
- 高效批量数据加载与清洗
- 多源数据整合与预处理
- 灵活的数据质量检测

### 📊 因子工程平台
构建65+个专业因子，涵盖：
- **价格因子**：均线系统、布林带、收益率矩阵
- **成交量因子**：量价关系、换手率分析
- **动量因子**：动量指标、RSI、MACD
- **波动率因子**：ATR、历史波动率
- **基本面因子**：估值体系、市值因子

### 🎯 因子筛选系统
- **IC系数分析**：信息系数评估因子预测能力
- **IR比率检验**：信息比率衡量因子稳定性
- 自动化有效因子自动筛选

### 🤖 机器学习模型库
集成主流算法支持：
- LightGBM
- XGBoost
- Random Forest

### 💹 智能选股模块
- 每日动态选股池生成
- 基于预期收益的智能排序
- Top-N 股票推荐

### 📈 回测分析系统
- 多种卖出策略对比（止损止盈、时间基础、RSI超买）
- 专业绩效指标计算
  - 绝对收益分析
  - 风险调整收益（夏普比率）
  - 最大回撤分析
  - 胜率与盈亏比
- 策略归因分析

### 🌐 Web可视化界面
- 基于Flask的Web应用
- 实时交易状态监控
- 选股池查询
- 股票详情与K线图展示
- 用户权限管理

## 🏗️ 项目结构

```
QuantAlpha/
├── project_package/              # 核心项目包
│   ├── main.py                   # 系统主程序入口
│   ├── config/                   # 配置文件目录
│   │   └── config.yaml           # 系统配置文件
│   └── src/                      # 源代码目录
│       ├── data_processing/      # 数据处理模块
│       │   ├── data_loader.py    # 数据加载器
│       │   └── feature_engineering.py  # 特征工程
│       ├── strategy/             # 策略模块
│       │   └── stock_selector.py # 股票选择器
│       ├── trading/              # 交易系统模块
│       │   └── trading_system.py # 回测交易系统
│       └── website/              # Web界面模块
│           ├── app.py            # Flask Web应用
│           └── templates/        # HTML模板
├── 选股/                         # 快速选股工具
│   ├── stock_selection_system.py # 选股系统核心
│   ├── run.py                    # 快速选股入口
│   └── config.yaml               # 选股配置
├── requirements.txt              # Python依赖
├── .gitignore                    # Git忽略文件
└── README.md                     # 项目文档
```

## 🚀 快速开始

### 环境要求
- Python 3.7+
- 详见 requirements.txt

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

将股票数据按以下结构组织：
```
选股/
├── 1-500/
│   └── *.csv
└── 501-1000/
    └── *.csv
```

### 配置文件

在 `project_package/config/config.yaml` 中配置系统参数：
- 数据源路径
- 回测时间范围
- 模型超参数
- IC/IR 阈值
- 选股参数

### 运行方式

#### 方式一：快速选股系统

```bash
cd 选股
python run.py
```

#### 方式二：完整研究平台

```bash
cd project_package
python main.py
```

#### 方式三：Web可视化界面

```bash
cd project_package/src/website
python app.py
```
然后在浏览器中访问 http://localhost:5000

## 📚 使用指南

### 工作流程

1. 数据加载与预处理
2. 多因子特征构建
3. IC/IR 因子筛选
4. 预测模型训练
5. 每日选股池生成
6. 策略回测分析
7. 研究报告输出

### 输出成果

系统在 `reports/` 目录生成：
- `daily_pools_*.csv` - 每日选股池
- `daily_values_*.csv` - 每日资产价值
- `trades_*.csv` - 交易记录
- `strategy_comparison_*.csv` - 策略对比分析

## 🔬 核心方法论

### IC/IR 分析框架
- **IC（Information Coefficient）**：因子值与未来收益率的相关系数
- **IR（Information Ratio）**：= IC均值 / IC标准差，衡量因子稳定性

### 多因子体系
覆盖技术分析核心因子：
- 趋势追踪（均线系统）
- 动量效应（RSI、MACD）
- 波动率（布林带、ATR）
- 成交量（换手率、量比）
- 基本面（估值指标）

### 模型训练
采用滚动训练，严格避免未来信息

### A股交易规则
- 100股一手，买卖数量应为1手或其整数倍
- 科创板/创业板高价股(≥100元)使用0.05元价格最小变动单位
- 一般股票使用0.01元价格最小变动单位

## 📊 策略回测

支持多种卖出策略对比：
- **止损止盈策略**：基于预设止损止盈点位
- **时间基础策略**：固定持有周期
- **RSI超买策略**：基于RSI技术指标

## ⚠️ 风险提示

1. 历史表现不代表未来收益
2. 回测结果存在过拟合风险
3. 实盘交易需谨慎
4. 建议进行样本外测试验证

## 🛠️ 技术栈

- **数据科学**：Pandas, NumPy
- **机器学习**：Scikit-learn, LightGBM, XGBoost
- **数据可视化**：Matplotlib, Plotly
- **Web框架**：Flask
- **配置管理**：PyYAML

## 📄 许可证

本项目仅供学习交流使用

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**免责声明**：本系统仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
