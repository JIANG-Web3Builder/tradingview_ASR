# TradingView 策略目录说明

## 目录目的

这个目录结构的目标是把 **当前主版本**、**历史版本**、**说明文档** 分开，避免后续继续迭代时很难看出每一版改了什么。

## 当前目录结构

```text
tradingview/
├─ active/
│  └─ asr_btc_strategy_current_v7.pine
├─ archive/
│  ├─ asr_btc_strategy_v7.pine
│  ├─ asr_btc_strategy_v8.pine
│  ├─ asr_btc_strategy_v9.pine
│  └─ asr_btc_strategy_v10.pine
├─ docs/
│  ├─ README_v1.md
│  ├─ strategy_version_map_v1.md
│  └─ v7_strategy_logic_v1.md
├─ asr_btc.pine
├─ asr_btc_indicator.pine
├─ asr_btc_range_strategy.pine
├─ asr_btc_strategy.pine
└─ asr_btc_trend_strategy.pine
```

## 各目录用途

- **`active/`**
  - 只放你当前真正想用的版本。
  - 现在主版本是 `asr_btc_strategy_current_v7.pine`。

- **`archive/`**
  - 放所有保留的历史策略版本。
  - 用来回溯、对比、找回某次改动。

- **`docs/`**
  - 放策略说明文档。
  - `strategy_version_map_v1.md` 用来看每个版本改了什么。
  - `v7_strategy_logic_v1.md` 用来看 `v7` 的策略内容与开仓逻辑。

## 当前使用建议

- **回测主版本时**
  - 优先打开 `active/asr_btc_strategy_current_v7.pine`

- **查历史改动时**
  - 去 `archive/` 找对应版本

- **想确认某版改了什么时**
  - 先看 `docs/strategy_version_map_v1.md`

- **想看 `v7` 策略本身在做什么时**
  - 看 `docs/v7_strategy_logic_v1.md`

## 后续版本管理规则

- **新增策略版本时**
  - 新版本先进入 `archive/`
  - 文件名继续保留版本号，例如 `asr_btc_strategy_v11.pine`

- **如果某个新版本成为主版本**
  - 从 `archive/` 复制一份到 `active/`
  - 例如 `asr_btc_strategy_current_v11.pine`

- **每次版本变化后**
  - 同步更新 `docs/strategy_version_map_v1.md`
  - 如果主版本策略逻辑发生了实质变化，再补新的主版本策略说明文档

## 当前状态

- **主参考版本**
  - `v7`

- **已删除的旧版本**
  - `v5`
  - `v6`

- **保留的迭代版本**
  - `v7`
  - `v8`
  - `v9`
  - `v10`
