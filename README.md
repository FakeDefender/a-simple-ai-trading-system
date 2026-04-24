# AI 交易研究系统 v0.9

这是一个面向真实交易落地方向演进的研究与执行基线系统。

当前已经有九层能力：

1. `v0.1` 研究链路：历史数据 -> 技术指标 -> 交易信号 -> 回测
2. `v0.2` 单标的执行链路：信号 -> 订单 -> 成交 -> 持仓 -> 账户权益
3. `v0.3` 组合执行链路：多标的信号 -> 组合选标 -> 组合仓位分配 -> 组合级风控
4. `v0.4` 执行增强链路：调仓频率控制 -> 交易日历判断 -> 成本模型 -> broker adapter 抽象
5. `v0.5` 实盘骨架链路：订单生命周期 -> 交易时段控制 -> 撤单/重试 -> live 风控 -> dry-run 执行引擎
6. `v0.6` 实盘服务骨架：增量式 live engine -> 轮询 service -> real broker adapter scaffold
7. `v0.7` 多市场抽象：market profile -> 多时段 session -> symbol 级手数/做空规则 -> mixed-market paper execution
8. `v0.8` Web 控制台基础版：配置编辑 -> 本地 API -> 一键运行主流程 / paper / live dry-run -> 结果浏览
9. `v0.9` Web 控制台增强版：任务日志 -> 账户曲线 -> JSON / CSV / 图片产物预览 -> 结果检查闭环

当前定位已经不是单纯的回测脚本，而是一个可以继续接到真实券商接口上的交易执行骨架。默认仍然使用内置 `paper_live` 做 dry-run；真实券商接入仍然通过实现 `BrokerAPIClient` 完成。

## 当前能力

- 支持 `csv` 和在线行情加载
- 自动计算均线、RSI、MACD、布林带、ATR、成交量等指标
- 生成确定性多空信号
- 输出单标的回测权益曲线、交易记录、绩效指标、风险指标
- 支持单标的 `paper trading`
- 支持多标的组合 `paper trading`
- 支持成本模型、调仓频率、调仓日历、组合风控
- 支持 `live trading dry-run` 的订单状态流转、交易时段控制、撤单/重试、日内 hard halt
- `LiveTradingEngine` 已支持增量处理，只处理新 bar，不重复处理旧数据
- `LiveTradingService` 已支持 `run_once` 和 `run_forever` 两种模式
- `build_live_adapter` 已支持 `paper_live` 和 `real` 两种 live adapter 路径
- 已提供 `ConfigurableRESTBrokerClient` / `BrokerAPIClient` 作为真实券商 client 协议骨架
- 新增 `MarketProfileResolver`，可统一解析市场时区、交易日、多时段交易时段、手数、碎股和做空规则
- 组合 paper trading 已支持同一轮里按 symbol 套用不同市场规则，例如 A 股和美股混合
- 新增内置 Web 控制台，可直接在浏览器里改配置、启动任务、查看最近结果、任务日志、账户曲线和产物预览
- LLM 为可选增强，默认关闭，不影响主流程启动

## 项目结构

```text
src/
├── agents/
│   └── ml_strategy_agent.py
├── config/
│   ├── api_keys.yaml.template
│   └── config.yaml
├── execution/
│   ├── broker_adapter.py
│   ├── cost_model.py
│   ├── live_broker.py
│   ├── live_risk_manager.py
│   ├── live_trading_engine.py
│   ├── live_trading_service.py
│   ├── market_profile.py
│   ├── market_session.py
│   ├── models.py
│   ├── paper_broker.py
│   ├── paper_trading_engine.py
│   ├── portfolio_paper_trading_engine.py
│   └── trading_calendar.py
├── utils/
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── data_processor.py
│   ├── data_storage.py
│   ├── openai_client.py
│   └── risk_calculator.py
├── webui/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── app_service.py
├── live_trading.py
├── main.py
├── paper_trading.py
└── web_app.py
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置

主配置文件在 `src/config/config.yaml`。

### 研究回测配置

```yaml
data:
  source: api
  symbol: aapl.us
  interval: d
  start_date: '2024-01-01'
  end_date: '2025-01-01'

strategy:
  fast_ma: 10
  slow_ma: 20
  rsi_long_threshold: 55
  rsi_short_threshold: 45

risk:
  stop_loss_pct: 0.03
  take_profit_pct: 0.06
  commission: 0.001
  slippage: 0.0005
```

### 市场抽象配置

```yaml
market:
  profile: auto
  symbol_profiles:
    aapl.us: us_equity
    600000.sh: cn_equity
  symbol_overrides:
    0700.hk:
      lot_size: 100
```

说明：

- `market.profile=auto` 时，系统会按 symbol 后缀自动推断市场
- 当前内置 profile：`default`、`us_equity`、`cn_equity`、`hk_equity`、`crypto_spot`
- `symbol_profiles` 用于精确指定某个 symbol 的市场
- `symbol_overrides` 用于覆写某个 symbol 的特殊规则，例如个别标的的最小下单单位
- 同一套 adapter 可按 symbol 应用不同 market rule，适合组合里混合 A 股和美股

### 执行成本配置

```yaml
execution_costs:
  fixed_commission: 0.0
  min_commission: 0.0
  sell_tax_rate: 0.0
```

### 单标的 Paper Trading 配置

```yaml
paper_trading:
  enabled: true
  initial_cash: 100000.0
  allocation_pct: 0.95
  allow_fractional: false
  lot_size: 1.0
  close_positions_on_finish: true
  price_field: close
  quantity_precision: 6
  rebalance_frequency: daily
  rebalance_weekday: 0
  rebalance_day_of_month: 1
  turnover_buffer: 0.0
  max_account_drawdown: 0.20
  adapter: paper
```

说明：

- `paper_trading` 里的 `allow_fractional` / `lot_size` 仍然可保留
- 但当 `market` 或 `symbol_profiles` 指定了更严格的市场规则时，以市场规则为准

### 组合 Paper Trading 配置

```yaml
portfolio:
  enabled: true
  target_gross_allocation: 0.95
  max_positions: 3
  max_gross_exposure: 1.0
  max_symbol_allocation: 0.35
  max_portfolio_drawdown: 0.20
  close_positions_on_finish: true
  price_field: close
  selection_metric: market_strength
  rebalance_frequency: weekly
  rebalance_weekday: 0
  rebalance_day_of_month: 1
  turnover_buffer: 0.01
  adapter: paper
```

### Live Trading 配置

```yaml
live_trading:
  enabled: false
  initial_cash: 100000.0
  allocation_pct: 0.95
  allow_fractional: false
  lot_size: 1.0
  quantity_precision: 6
  price_field: close
  adapter: paper_live
  sessions:
    - start: '09:30'
      end: '11:30'
    - start: '13:00'
      end: '15:00'
  exit_only_start: '14:57'
  cancel_after_seconds: 300
  max_order_retries: 2
  flatten_outside_trading_hours: false
  min_signal_strength: 0.0
  fill_delay_seconds: 0
  reject_first_n_orders: 0
  close_positions_on_finish: false

live_risk:
  max_order_notional: 0.0
  max_position_notional: 0.0
  max_daily_drawdown: 0.0
  max_open_orders: 10
  max_orders_per_day: 0
  max_consecutive_failures: 0
```

说明：

- `MarketSession` 现在支持多段交易时段，例如 A 股午休场景
- `live_trading.sessions` 可以直接定义多段时段
- 如果不显式写 `sessions`，则会优先使用 `market` 里解析出的 profile 默认时段

### Broker 配置

```yaml
broker:
  provider: generic_rest
  paper: true
  base_url: ''
  account_id: ''
  timeout_seconds: 10.0
```

说明：

- 当 `live_trading.adapter=paper_live` 时，系统使用内置 dry-run broker
- 当 `live_trading.adapter=real` 时，系统走 `RealBrokerAdapter`
- `RealBrokerAdapter` 不直接内置任何券商 SDK，需要你实现 `BrokerAPIClient`
- `ConfigurableRESTBrokerClient` 只是一个通用 REST 协议占位，不包含具体下单细节
- 市场规则和券商接入现在是两层：市场负责约束，broker 负责下单回报

### Live Service 配置

```yaml
live_service:
  poll_interval_seconds: 60
  force_update_each_cycle: true
  save_results_each_cycle: true
  max_cycles: 1
  stop_on_error: true
  results_root: results/live_service
```

说明：

- `max_cycles=1` 表示只跑一轮，适合 dry-run 检查
- 改成 `0` 或 `null` 时可按你的调度方式长期运行
- `poll_interval_seconds` 控制轮询间隔
- `force_update_each_cycle` 决定每轮是否强制刷新行情
- `save_results_each_cycle` 决定是否每轮持久化结果文件

## 运行方式

### 1. 跑研究 + paper trading 主流程

```bash
python src/main.py
```

### 2. 只跑 paper trading

```bash
python src/paper_trading.py
```

### 3. 跑 live trading service

```bash
python src/live_trading.py
```

说明：

- 默认会走 `LiveTradingService.run_forever()`
- 由于 `live_service.max_cycles` 默认是 `1`，所以默认行为仍然是安全的单轮 dry-run
- 要改成常驻服务，只需要改 `live_service.max_cycles` 和 `poll_interval_seconds`

### 4. 启动 Web 控制台

```bash
python src/web_app.py
```

说明：

- 默认监听 `http://127.0.0.1:8800`
- 页面里可以直接修改 YAML、保存配置、启动 `主流程 / paper / live dry-run`
- 结果区会读取本地 `results/` 目录并显示最近运行记录
- 选中任务后可查看最近日志
- 选中结果后可查看权益曲线 / 账户曲线
- JSON、CSV、日志文本和图片产物可直接在页面内预览，也可以单独打开
- 产物文件可直接从页面打开

## 输出结果

运行后结果保存在 `results/<时间戳>/` 或 `results/live_service/<时间戳>/`。

### 单标的 Paper Trading 结果

- `paper_orders.csv`
- `paper_fills.csv`
- `paper_account_history.csv`
- `paper_positions.json`
- `paper_summary.json`

### 组合结果

- `portfolio_research_summary.json`
- `portfolio_orders.csv`
- `portfolio_fills.csv`
- `portfolio_account_history.csv`
- `portfolio_symbol_history.csv`
- `portfolio_positions.json`
- `portfolio_summary.json`
- `portfolio_symbol_summary.json`

### Live 结果

- `live_orders.csv`
- `live_fills.csv`
- `live_account_history.csv`
- `live_positions.json`
- `live_summary.json`

关键字段：

- `live_orders.csv` 包含订单最终状态，如 `submitted`、`filled`、`canceled`、`rejected`
- `live_account_history.csv` 包含 `market_profile`、`session_state`、`decision_reason`、`execution_event`、`daily_drawdown`、`daily_orders`、`consecutive_failures`、`open_orders`
- `portfolio_symbol_history.csv` 现在会记录每个 symbol 的 `market_profile`
- `live_summary.json` 汇总 `rejected_orders`、`canceled_orders`、`session_blocks`、`risk_blocks`

## 如何接真实券商

当前推荐的接入方式是：

1. 保持 `LiveTradingEngine` 和 `LiveTradingService` 不动
2. 通过 `market.profile` 或 `market.symbol_profiles` 固化目标市场规则
3. 为目标券商实现一个 `BrokerAPIClient`
4. 在 `submit_order` / `cancel_order` / `get_order` / `list_open_orders` / `list_fills` / `get_account` / `list_positions` 里填真实接口
5. 把配置改成：

```yaml
market:
  profile: cn_equity

live_trading:
  adapter: real

broker:
  provider: your_broker
  paper: true
  base_url: https://your-broker-api
  account_id: your_account_id
```

也就是说，后续真正需要替换的主要是“broker client”，而不是 live engine 主循环；市场规则层已经单独抽出来了。

## 测试

```bash
python -m compileall src tests
python -m unittest discover -s tests -v
python -m pytest tests -q
```

当前已覆盖的重点场景：

- 配置加载与离线模式回退
- CSV 数据加载与信号生成
- 单标的 paper broker 开平仓
- 最低佣金与卖出税费
- 单标的 weekly 调仓
- 组合选标与暴露限制
- 组合回撤暂停
- `turnover_buffer` 抑制小额调仓
- live trading 时段控制
- live trading exit-only 平仓而不反手
- live trading 拒单后的自动重试
- live trading 挂单超时撤单重提
- live trading 日内回撤 hard halt
- live engine 增量处理只消费新 bar
- live service 多轮运行不重复处理旧数据
- real adapter scaffold 可通过 fake broker client 协议驱动
- A 股多时段交易时段控制
- symbol 级市场规则解析
- 组合 paper trading 混合市场手数归整
- Web 控制台静态页面、配置 API、任务接口、任务日志接口、曲线预览接口、文件预览接口

## 当前明确不包含

以下能力还没有进入当前版本：

- 真实券商 SDK / WebSocket 适配实现
- 实时行情订阅与事件驱动撮合
- 多标的 live execution 与组合级 live 风控
- 真实成交回报、撤单回报和账户异步同步
- 更细粒度的盘口、冲击成本和排队模型
- 保证金、借券、融资融券约束
- 实时监控、告警和值守
- walk-forward 验证、参数搜索和策略运营层

## 下一阶段建议

建议按这个顺序继续演进：

1. 为目标市场补具体 broker client，例如 QMT / miniQMT 或券商 REST SDK
2. 把 `BrokerAPIClient` 接到真实订单回报、撤单回报和账户同步
3. 增加多标的 live execution 与组合级 live 风控
4. 最后再补监控、告警、报表和策略运营层
