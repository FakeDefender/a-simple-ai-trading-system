# AI驱动的量化交易策略系统

一个基于人工智能和大语言模型的智能量化交易策略系统，集成了多智能体协作、技术分析、风险管理和回测分析功能。

## 🚀 项目特性

### 核心功能
- **多智能体协作**：策略开发者、风险分析师、交易顾问三个AI智能体协同工作
- **LLM驱动策略生成**：使用OpenAI GPT模型动态生成交易策略
- **全面技术分析**：支持20+种技术指标（MA、RSI、MACD、布林带等）
- **智能风险管理**：动态止损、仓位管理、风险指标监控
- **完整回测系统**：历史数据回测、绩效分析、风险评估

### 技术特色
- **自适应策略调整**：根据股票特征自动调整策略参数
- **多数据源支持**：Stooq、Yahoo Finance等多个数据源
- **实时数据处理**：支持分钟级到日级多种时间周期
- **可视化分析**：权益曲线、回撤分析、交易记录图表
- **模块化设计**：高度解耦的组件架构，易于扩展

## 📊 系统架构

```
AI量化交易系统
├── 数据层 (Data Layer)
│   ├── 数据获取 (DataLoader)
│   ├── 数据处理 (DataProcessor)
│   └── 数据存储 (DataStorage)
├── 策略层 (Strategy Layer)
│   ├── ML策略智能体 (MLStrategyAgent)
│   ├── 多智能体协作
│   └── 信号生成
├── 风险层 (Risk Layer)
│   ├── 风险计算器 (RiskCalculator)
│   ├── 仓位管理
│   └── 止损止盈
└── 执行层 (Execution Layer)
    ├── 回测引擎
    ├── 绩效分析
    └── 结果输出
```

## 🛠️ 安装指南

### 环境要求
- Python 3.8+
- 8GB+ RAM
- 稳定的网络连接（用于数据获取和LLM调用）

### 快速安装

1. **克隆项目**
```bash
git clone <repository-url>
cd trading-ai-system
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置API密钥**
```bash
# 复制配置模板
cp src/config/api_keys.yaml.template src/config/api_keys.yaml

# 编辑配置文件，添加你的OpenAI API密钥
# openai:
#   api_key: "your-openai-api-key"
#   model: "gpt-3.5-turbo"
```

## 🎯 快速开始

### 基础使用

1. **配置交易标的**
编辑 `src/config/config.yaml`：
```yaml
data:
  symbol: 'aapl.us'  # 苹果股票
  interval: 'd'      # 日线数据
  start_date: '2023-01-01'
  end_date: '2024-12-31'
```

2. **运行回测**
```bash
python src/main.py
```

3. **查看结果**
结果将保存在 `results/` 目录下，包含：
- 交易记录 (`trades.csv`)
- 权益曲线 (`equity_curve.csv`)
- 绩效指标 (`metrics.json`)
- 可视化图表 (`backtest_results.png`)

### 高级配置

**策略参数调整**
```yaml
strategy:
  type: 'trend_following'
  rsi_period: 14
  rsi_oversold_threshold: 30
  rsi_overbought_threshold: 70
  
risk:
  stop_loss: 0.03      # 3%止损
  take_profit: 0.06    # 6%止盈
  max_position: 1.0    # 最大仓位
```

**回测设置**
```yaml
backtest:
  initial_capital: 1000000  # 初始资金100万
  commission: 0.001         # 手续费0.1%
  slippage: 0.001          # 滑点0.1%
```

## 📁 项目结构

```
trading-ai-system/
├── src/                          # 源代码目录
│   ├── agents/                   # AI智能体模块
│   │   ├── ml_strategy_agent.py  # 主策略智能体
│   │   └── __init__.py
│   ├── utils/                    # 工具模块
│   │   ├── data_loader.py        # 数据加载器
│   │   ├── data_processor.py     # 数据处理器
│   │   ├── data_storage.py       # 数据存储
│   │   ├── openai_client.py      # OpenAI客户端
│   │   ├── risk_calculator.py    # 风险计算器
│   │   ├── config_loader.py      # 配置加载器
│   │   └── __init__.py
│   ├── config/                   # 配置文件
│   │   ├── config.yaml           # 主配置文件
│   │   ├── api_keys.yaml         # API密钥配置
│   │   └── __init__.py
│   ├── main.py                   # 主程序入口
│   └── __init__.py
├── data/                         # 数据目录
├── results/                      # 结果输出目录
├── tests/                        # 测试文件
├── requirements.txt              # 依赖列表
├── setup.py                      # 安装配置
└── README.md                     # 项目说明
```

## 🔧 核心组件

### 1. MLStrategyAgent (主策略智能体)
- **功能**：协调多个AI智能体，生成交易信号
- **特性**：自适应参数调整、多策略支持
- **输入**：市场数据、技术指标
- **输出**：买入/卖出/持有信号

### 2. DataLoader (数据加载器)
- **功能**：从多个数据源获取股票数据
- **支持**：Stooq、Yahoo Finance
- **特性**：数据缓存、增量更新、错误重试

### 3. DataProcessor (数据处理器)
- **功能**：计算技术指标、数据清洗
- **指标**：MA、RSI、MACD、布林带、ATR等20+种
- **特性**：向量化计算、缺失值处理

### 4. RiskCalculator (风险计算器)
- **功能**：风险指标计算、仓位管理
- **指标**：VaR、最大回撤、夏普比率、波动率
- **特性**：实时风险监控、动态调整

## 📈 使用示例

### 示例1：苹果股票趋势跟踪策略
```python
from src.agents.ml_strategy_agent import MLStrategyAgent
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader

# 加载配置
config = load_config()
config['data']['symbol'] = 'aapl.us'

# 创建组件
data_loader = DataLoader(config)
strategy_agent = MLStrategyAgent(config, data_loader)

# 获取数据并生成信号
market_data = data_loader.load_data('aapl.us', 'd')
signals = strategy_agent.generate_signals(market_data)

# 运行回测
results = strategy_agent._backtest_strategy(market_data, signals)
```

### 示例2：多股票批量回测
```python
symbols = ['aapl.us', 'tsla.us', 'msft.us']
results = {}

for symbol in symbols:
    config['data']['symbol'] = symbol
    # ... 运行回测逻辑
    results[symbol] = backtest_results
```

## 📊 性能指标

系统支持以下绩效和风险指标：

### 绩效指标
- **总收益率**：策略总体收益表现
- **年化收益率**：年化后的收益率
- **夏普比率**：风险调整后收益
- **胜率**：盈利交易占比
- **盈亏比**：平均盈利/平均亏损

### 风险指标
- **最大回撤**：历史最大亏损幅度
- **VaR (95%)**：95%置信度下的风险价值
- **波动率**：收益率标准差
- **Beta系数**：相对市场的系统性风险
- **相关性**：与基准的相关程度

## 🔍 测试结果示例

### 特斯拉股票 (TSLA) - 优化策略
```
总收益率: 44.66%
夏普比率: 3.77
胜率: 45.83%
最大回撤: 16.68%
交易次数: 24笔
```

## 🚧 开发计划

### 近期计划
- [ ] 增加更多技术指标
- [ ] 支持加密货币数据
- [ ] 实现实盘交易接口
- [ ] 添加机器学习模型

### 长期计划
- [ ] 多资产组合优化
- [ ] 高频交易策略
- [ ] 情感分析集成
- [ ] Web界面开发

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## ⚠️ 免责声明

本系统仅用于教育和研究目的。使用本系统进行实际交易的风险由用户自行承担。过往表现不代表未来收益，投资有风险，入市需谨慎。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件：1781853359@qq.com

---