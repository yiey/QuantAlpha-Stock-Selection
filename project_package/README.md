# 股票选股与交易系统

## 项目概述

本项目实现了一个完整的股票选股与交易系统，涵盖数据加载、特征工程、选股策略、交易模拟和结果展示等完整流程。

## 项目结构

```
├── config/             # 配置文件目录
│   └── config.yaml    # 系统配置文件
├── src/               # 核心代码目录
│   ├── data_processing/   # 数据处理模块
│   ├── strategy/      # 选股策略模块
│   ├── trading/       # 交易系统模块
│   └── website/       # 网页应用模块
├── main.py            # 系统主入口
└── 项目总结文档.md      # 项目详细文档
```

## 环境要求

- Python 3.11+
- pandas, numpy, scikit-learn, tqdm
- flask, flask-login, plotly
- pyyaml

## 安装依赖

```bash
pip install pandas numpy scikit-learn tqdm flask flask-login plotly pyyaml
```

## 数据准备

1. 将股票数据CSV文件放在1-500/目录下
2. 每个CSV文件命名为股票代码.csv（如：000001.csv）
3. CSV文件必须包含'交易日期'和'收盘价'等基本字段

## 运行系统

```bash
# 运行回测
python main.py

# 启动网页应用
python src/website/app.py
```

## 详细文档

请参考`项目总结文档.md`获取完整的项目说明、架构和使用指南。