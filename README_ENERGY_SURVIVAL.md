# 多機器人能量求生模擬器
Multi-Robot Energy Survival Simulator

## 🎯 專案目標

這是一個**簡化**的多智能體模擬環境，專注於研究：

- **能量管理**：機器人如何在有限能量下生存
- **隨機探索 vs 理性決策**：epsilon-greedy 策略的群體動態
- **多智能體互動**：機器人之間的碰撞和資源競爭

**關鍵簡化**：
- ❌ 沒有家具障礙
- ❌ 沒有垃圾收集任務
- ✅ 純粹的能量求生機制
- ✅ 3×3 小地圖（預設）

---

## 🎮 核心機制

### 1. 環境設定
- **地圖**：n×n 網格（預設 3×3）
- **充電座**：4個角落 (0,0), (0,n-1), (n-1,0), (n-1,n-1)
- **機器人**：4台，初始在各自的充電座上

### 2. 動作空間
每台機器人每回合可執行 5 種動作：
- `0`: 向上
- `1`: 向下
- `2`: 向左
- `3`: 向右
- `4`: 停留（在充電座上可充電）

### 3. 能量系統
- **初始能量**：`initial_energy`（預設 100）
- **移動消耗**：`e_move`（預設 1）
- **碰撞消耗**：`e_collision`（預設 3）
- **充電恢復**：`e_charge`（預設 5，需在充電座停留）
- **能量耗盡**：`energy <= 0` 時機器人停機 (`is_active = False`)

### 4. 碰撞規則
機器人移動時會發生碰撞（消耗 `e_collision` 能量）：
1. **撞邊界**：移動超出地圖範圍
2. **撞其他機器人**：移動到已被佔據的格子
3. **搶佔衝突**：多個機器人嘗試移到同一格子

### 5. Epsilon-Greedy 策略

這是一個混合策略，平衡探索和利用：

```
if random() < ε:
    執行隨機動作  # 探索 (Explore)
else:
    執行理性求生策略  # 利用 (Exploit)
```

**理性求生策略**：
- 如果在家（充電座） → 停留充電
- 如果不在家 → 朝著家的方向移動

---

## 📦 安裝與執行

### 安裝依賴
```bash
pip install numpy pygame
```

### 基本執行
```bash
python robot_vacuum_env.py
```

這會啟動一個 3×3 的模擬，使用預設配置：
- 初始能量：100
- 探索率 (ε)：20%
- 總回合數：500

### 觀察視窗
模擬會開啟一個 Pygame 視窗，顯示：
- **地圖**：白色空地 + 藍色充電座
- **機器人**：4種顏色的圓圈
  - 紅色 = 機器人 0
  - 綠色 = 機器人 1
  - 黃色 = 機器人 2
  - 紫色 = 機器人 3
- **資訊面板**：每台機器人的能量條、充電次數和狀態

---

## ⚙️ 配置參數

使用 `energy_survival_config.py` 中的預設配置：

```python
from robot_vacuum_env import RobotVacuumEnv
from energy_survival_config import get_config

# 使用預設配置
config = get_config('base')
env = RobotVacuumEnv(config)
```

### 可用配置模式

| 模式 | 說明 | ε | 特點 |
|------|------|---|------|
| `base` | 基礎模式 | 20% | 標準平衡設定 |
| `high_explore` | 高探索 | 50% | 更多隨機行為 |
| `low_explore` | 低探索 | 10% | 更理性的決策 |
| `pure_rational` | 純理性 | 0% | 完全理性求生 |
| `pure_random` | 純隨機 | 100% | 完全隨機探索 |
| `energy_scarce` | 能量緊張 | 20% | 低能量，高消耗 |
| `energy_abundant` | 能量充裕 | 20% | 高能量，低消耗 |
| `large` | 大地圖 | 20% | 5×5 房間 |
| `tiny` | 超小地圖 | 20% | 2×2 房間（極限擁擠） |
| `quick` | 快速測試 | 20% | 100 回合 |
| `long` | 長期模擬 | 20% | 2000 回合 |

### 自訂配置

```python
custom_config = {
    'n': 3,                 # 房間大小
    'initial_energy': 100,  # 初始能量
    'e_move': 1,            # 移動消耗
    'e_charge': 5,          # 充電恢復
    'e_collision': 3,       # 碰撞消耗
    'n_steps': 500,         # 總回合數
    'epsilon': 0.2          # 探索率
}

env = RobotVacuumEnv(custom_config)
```

---

## 🔬 研究問題

這個簡化環境可以探索：

### 1. 探索率 (ε) 的影響
- **ε = 0** (純理性)：所有機器人都會立即回家充電，永遠不離開
- **ε = 1** (純隨機)：機器人隨機遊走，很容易能量耗盡
- **ε = 0.2** (平衡)：大部分時間理性，偶爾探索

**實驗建議**：
```bash
# 比較不同探索率
python -c "from robot_vacuum_env import *; from energy_survival_config import *; ..."
```

### 2. 群體動態
在 3×3 的小地圖中，4 台機器人會產生有趣的互動：
- **擁擠效應**：中央格 (1,1) 成為競爭熱點
- **碰撞模式**：機器人返家路徑會重疊
- **資源競爭**：充電座只能一次一個機器人使用

### 3. 能量管理策略
- **理性策略**：待在家不動 = 能量持續增長（但無探索）
- **隨機策略**：持續探索 = 能量持續下降（最終停機）
- **平衡策略**：適度探索 + 及時回家充電

---

## 📊 預期結果

### 純理性模式 (ε=0)
```
存活機器人: 4/4  ✅ 所有機器人存活
平均能量: 100    ⚡ 能量保持滿格
總充電: 0 次      ❓ 從不離開家，沒有充電
```

### 純隨機模式 (ε=1)
```
存活機器人: 0/4  ❌ 所有機器人停機
平均能量: 0      💀 能量耗盡
總充電: ~50 次    🔋 偶爾路過充電座
```

### 平衡模式 (ε=0.2)
```
存活機器人: 3-4/4  ✅ 大部分存活
平均能量: 40-70    ⚡ 能量維持中等
總充電: ~100 次    🔋 頻繁返家充電
```

---

## 🛠️ 程式碼結構

```
robot_vacuum_env.py          # 主環境類別
├── RobotVacuumEnv          # 環境類別
│   ├── __init__()          # 初始化
│   ├── reset()             # 重置環境
│   ├── step()              # 執行一步
│   ├── render()            # 視覺化
│   └── get_global_state()  # 獲取狀態
├── get_rational_action()   # 理性策略函數
└── main()                  # 主程式（epsilon-greedy）

energy_survival_config.py   # 配置文件
└── get_config()            # 獲取預設配置
```

---

## 🎓 與 RL 的關係

雖然這個模擬器**不包含 RL 訓練邏輯**，但它提供了一個理想的環境來理解：

### Epsilon-Greedy 的概念
```python
# 這是 RL 中最基礎的探索策略
if random() < epsilon:
    action = random_action()      # 探索
else:
    action = best_action(state)   # 利用
```

在我們的模擬中：
- **探索**：隨機動作
- **利用**：理性求生策略（手工設計的「最佳」策略）

### 未來擴展
這個環境可以作為基礎，添加：
- **Q-Learning**：學習每個 state-action 的價值
- **DQN**：使用神經網路逼近 Q 函數
- **Actor-Critic**：同時學習策略和價值函數
- **Multi-Agent RL**：學習機器人之間的協作

---

## 🚀 快速開始範例

### 範例 1：觀察不同探索率

```python
from robot_vacuum_env import RobotVacuumEnv, main
from energy_survival_config import get_config

# 測試純理性模式
config = get_config('pure_rational')
# 修改 main() 中的 config 或直接在這裡創建環境
```

### 範例 2：自訂實驗

```python
import numpy as np
from robot_vacuum_env import RobotVacuumEnv, get_rational_action
import random

# 實驗：比較不同 epsilon 值的存活率
epsilons = [0.0, 0.1, 0.2, 0.5, 1.0]
results = []

for eps in epsilons:
    config = {
        'n': 3,
        'initial_energy': 100,
        'e_move': 1,
        'e_charge': 5,
        'e_collision': 3,
        'n_steps': 500,
        'epsilon': eps
    }

    env = RobotVacuumEnv(config)
    env.reset()

    # ... 執行模擬 ...
    # results.append(survival_rate)

print(f"Epsilon vs 存活率: {results}")
```

---

## 📝 參數調整建議

### 讓模擬更困難
- ⬇️ 降低 `initial_energy`
- ⬆️ 增加 `e_move`
- ⬆️ 增加 `e_collision`
- ⬇️ 降低 `e_charge`
- ⬆️ 增加 `n`（更大的地圖）

### 讓模擬更簡單
- ⬆️ 增加 `initial_energy`
- ⬇️ 降低 `e_move`
- ⬇️ 降低 `e_collision`
- ⬆️ 增加 `e_charge`
- ⬇️ 降低 `n`（更小的地圖）

---

## 🤔 常見問題

### Q: 為什麼移除了家具和垃圾？
A: 為了專注於核心的「能量求生」動態。家具和垃圾增加了不必要的複雜性。

### Q: 為什麼預設是 3×3 而不是更大的地圖？
A: 3×3 提供了最小的有趣空間：
- 4 個角落給 4 台機器人
- 1 個中央格子成為競爭焦點
- 足夠小，容易觀察和理解
- 足夠大，產生非平凡的互動

### Q: 理性策略為什麼是「回家」？
A: 這是最簡單的生存策略：
- 待在充電座 = 能量持續恢復
- 不移動 = 不消耗能量
- 保證存活（但無探索）

### Q: Epsilon-greedy 是最佳策略嗎？
A: 不是！這只是最簡單的策略。更好的策略可能是：
- 根據當前能量決定何時探索
- 學習避開其他機器人
- 預測充電座的佔用情況

---

## 📚 延伸閱讀

- [Epsilon-Greedy Algorithm](https://en.wikipedia.org/wiki/Multi-armed_bandit#Epsilon-greedy_strategy)
- [Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1911.10635)
- [Exploration-Exploitation Trade-off](https://en.wikipedia.org/wiki/Exploration-exploitation_dilemma)

---

## 📄 授權

MIT License
