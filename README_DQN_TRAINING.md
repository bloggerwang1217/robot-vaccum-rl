# DQN 訓練指南

使用 DQN 訓練機器人學習能量求生策略

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

主要依賴：
- `numpy`: 數值計算
- `pygame`: 視覺化
- `gymnasium`: Gym API
- `stable-baselines3`: DQN 實現
- `tensorboard`: 訓練監控

---

## 🎮 Gym Wrapper 設計

### 核心概念

我們提供了一個**可配置的 Gym wrapper**，支援兩種觀察模式：

| 模式 | Observation | Network | 適用場景 |
|------|-------------|---------|---------|
| **Vector** | 扁平向量 (15維) | MLP | 3×3 小地圖 ✅ 推薦作為起點 |
| **Grid** | 多通道網格 (4×n×n) | CNN | 5×5+ 大地圖，未來擴展 |

**關鍵設計**：
- ✅ **現在**：從簡單的 `vector` 模式開始
- ✅ **未來**：只需改一個參數就能切換到 `grid` 模式
- ✅ **統一 API**：無論哪種模式，訓練代碼都一樣

---

## 📊 Observation 結構

### Vector 模式（15維）

```python
[
    # === 自己的信息 (4維) ===
    x / (n-1),                # 歸一化 x 座標 [0, 1]
    y / (n-1),                # 歸一化 y 座標 [0, 1]
    energy / initial_energy,  # 歸一化能量 [0, 1]
    is_on_charger,            # 是否在充電座 {0, 1}

    # === 到家的距離 (2維) ===
    (home_x - x) / (n-1),     # x 方向距離 [-1, 1]
    (home_y - y) / (n-1),     # y 方向距離 [-1, 1]

    # === 其他3個機器人 (3×3=9維) ===
    # 對每個機器人：
    (other.x - x) / (n-1),    # 相對 x
    (other.y - y) / (n-1),    # 相對 y
    other.is_active,          # 是否活躍
    # ... 重複3次
]
```

**為什麼選擇這 15 維？**
- ✅ **足夠的信息**：位置、能量、距離、其他機器人
- ✅ **小而簡單**：易於訓練，適合 MLP
- ✅ **歸一化**：所有值在 [-1, 1] 或 [0, 1]，利於學習

### Grid 模式（4×n×n）

4 個通道：
```python
Channel 0: 充電座層    # 所有充電座的位置
Channel 1: 自己        # 自己的位置
Channel 2: 其他機器人  # 其他機器人的位置
Channel 3: 能量信息    # 自己的歸一化能量
```

**何時使用？**
- 地圖 ≥ 5×5 時考慮
- 想保留空間結構信息
- 想使用 CNN

---

## 💰 Reward 設計

我們提供 3 種獎勵塑形方式（通過 `reward_shaping` 參數選擇）：

### 1. Dense Reward（推薦，預設）

```python
reward = 0.0

# 基礎存活獎勵
if is_active:
    reward += 0.01

# 能量變化獎勵
energy_delta = current_energy - previous_energy
reward += energy_delta * 0.01

# 充電獎勵
if just_charged:
    reward += 0.5

# 碰撞懲罰
if collision:
    reward -= 0.3

# 能量過低警告
if energy_ratio < 0.2:
    reward -= 0.1

# 死亡重罰
if died:
    reward -= 5.0
```

**優點**：
- ✅ 提供密集的學習信號
- ✅ 更快學習
- ✅ 推薦用於快速驗證

### 2. Simple Reward

```python
if is_active:
    reward = +0.1  # 每步存活
else:
    reward = -10.0  # 死亡
```

### 3. Sparse Reward

```python
# 只在 episode 結束時給獎勵
if done:
    reward = 1.0 if survived else 0.0
else:
    reward = 0.0
```

**選擇建議**：
- 🥇 **Dense**：適合快速驗證和學習
- 🥈 **Simple**：簡單但有效
- 🥉 **Sparse**：挑戰性高，需要更多探索

---

## 🎓 訓練流程

### 方法 1：使用訓練腳本（推薦）

```bash
# 訓練 vector 模式（3×3 地圖）
python train_dqn.py --mode vector --steps 100000

# 訓練 grid 模式（5×5 地圖）
python train_dqn.py --mode grid --config large --steps 200000

# 自訂參數
python train_dqn.py \
    --mode vector \
    --config base \
    --steps 50000 \
    --lr 0.0005 \
    --robot-id 0
```

**參數說明**：
- `--mode`: 觀察模式（`vector` 或 `grid`）
- `--config`: 環境配置（`base`, `large`, `energy_scarce` 等）
- `--steps`: 訓練步數
- `--lr`: 學習率（vector 預設 1e-3，grid 預設 1e-4）
- `--robot-id`: 訓練哪個機器人（0-3）

### 方法 2：自訂訓練腳本

```python
from stable_baselines3 import DQN
from gym_wrapper import make_env

# 創建環境
env = make_env(
    obs_type='vector',           # 'vector' 或 'grid'
    reward_shaping='dense'       # 'simple', 'dense', 'sparse'
)

# 創建 DQN 模型
model = DQN(
    policy="MlpPolicy",          # vector 用 MlpPolicy，grid 用 CnnPolicy
    env=env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1
)

# 訓練
model.learn(total_timesteps=100000)

# 保存
model.save("./models/my_dqn")
```

---

## 📈 監控訓練

訓練過程中會自動保存到 TensorBoard：

```bash
# 查看訓練曲線
tensorboard --logdir ./models/dqn_vector/tensorboard/
```

**關鍵指標**：
- `rollout/ep_rew_mean`: 平均回合獎勵
- `train/loss`: 訓練損失
- `rollout/ep_len_mean`: 平均回合長度
- `time/fps`: 訓練速度

---

## 🧪 測試訓練好的 Agent

### 可視化測試

```bash
# 測試最佳模型
python test_agent.py --model ./models/dqn_vector/best_model/best_model

# 測試最終模型
python test_agent.py --model ./models/dqn_vector/final_model --episodes 10

# 不渲染（快速測試）
python test_agent.py --model ./models/dqn_vector/final_model --no-render --episodes 20
```

### 與 Baseline 比較

```bash
# 比較 DQN vs 隨機策略
python test_agent.py --model ./models/dqn_vector/best_model/best_model --compare
```

這會輸出類似：
```
DQN Agent:
  平均獎勵: 15.32 ± 3.45
  存活率: 9/10 (90.0%)

Baseline (隨機):
  平均獎勵: -8.21 ± 5.67
  存活率: 2/10 (20.0%)

改進幅度:
  獎勵提升: +286.5%
  存活率提升: +70.0%
```

---

## 🎯 預期結果

### 3×3 地圖 (Vector 模式)

**訓練前（隨機策略）**：
- 平均獎勵: ~-10
- 存活率: ~20%
- 平均充電: ~2 次

**訓練後（100k 步）**：
- 平均獎勵: ~10-20
- 存活率: ~80-90%
- 平均充電: ~20-30 次

**學習到的策略**：
1. ✅ 在家時停留充電
2. ✅ 能量低時返回充電
3. ✅ 避免碰撞
4. ✅ 平衡探索和充電

---

## 🔬 實驗建議

### 實驗 1：觀察模式比較

```bash
# Vector 模式
python train_dqn.py --mode vector --config base --steps 100000

# Grid 模式（同樣 3×3 地圖）
python train_dqn.py --mode grid --config base --steps 100000
```

**研究問題**：
- 哪種模式學習更快？
- 最終表現有差異嗎？
- 對於 3×3 小地圖，vector 是否足夠？

### 實驗 2：Reward Shaping 比較

修改 `gym_wrapper.py` 中的 `reward_shaping` 參數：

```python
# Dense reward
env = make_env(reward_shaping='dense')

# Simple reward
env = make_env(reward_shaping='simple')

# Sparse reward
env = make_env(reward_shaping='sparse')
```

**研究問題**：
- Dense reward 是否學習更快？
- Sparse reward 能否學到同樣的策略？
- 需要多少步才能收斂？

### 實驗 3：環境難度

```bash
# 簡單：能量充裕
python train_dqn.py --config energy_abundant

# 中等：標準配置
python train_dqn.py --config base

# 困難：能量緊張
python train_dqn.py --config energy_scarce
```

### 實驗 4：多機器人協作

訓練所有 4 個機器人：

```bash
# 訓練機器人 0
python train_dqn.py --robot-id 0 --save-path ./models/robot_0

# 訓練機器人 1
python train_dqn.py --robot-id 1 --save-path ./models/robot_1

# 訓練機器人 2
python train_dqn.py --robot-id 2 --save-path ./models/robot_2

# 訓練機器人 3
python train_dqn.py --robot-id 3 --save-path ./models/robot_3
```

**研究問題**：
- 不同位置的機器人學到不同策略嗎？
- 是否出現專業化分工？

---

## 📂 項目結構

```
robot-vaccum-rl/
├── robot_vacuum_env.py          # 底層環境
├── gym_wrapper.py               # Gym API wrapper ⭐
├── energy_survival_config.py    # 環境配置
├── train_dqn.py                 # 訓練腳本 ⭐
├── test_agent.py                # 測試腳本 ⭐
├── requirements.txt             # 依賴
└── models/                      # 保存的模型
    ├── dqn_vector/
    │   ├── best_model/
    │   ├── checkpoints/
    │   ├── final_model.zip
    │   └── tensorboard/
    └── dqn_grid/
```

---

## 🐛 常見問題

### Q: Vector 還是 Grid 模式？

**現在（3×3）**：使用 `vector` 模式
- ✅ 簡單、快速
- ✅ 15 維向量足夠表達所有信息
- ✅ MLP 訓練更快

**未來（5×5+）**：考慮 `grid` 模式
- ✅ 保留空間結構
- ✅ CNN 可能學到更好的策略
- ⚠️ 但訓練更慢

### Q: 訓練多久才夠？

- **快速驗證**：10k-50k 步（~5-30 分鐘）
- **完整訓練**：100k-200k 步（~1-2 小時）
- **精細調優**：500k+ 步

**判斷標準**：
- TensorBoard 上 `ep_rew_mean` 穩定增長後趨於平穩
- 存活率達到 80%+

### Q: 如何提高訓練速度？

1. **減少渲染**：訓練時不要開 `render=True`
2. **並行環境**：使用 `VecEnv`（stable-baselines3 支援）
3. **減少 evaluation 頻率**：調整 `eval_freq`
4. **使用 GPU**：stable-baselines3 支援 CUDA

### Q: 如何調整超參數？

**學習率**：
- Vector: `1e-3` 到 `1e-4`
- Grid: `1e-4` 到 `1e-5`

**Exploration**：
- `exploration_fraction`: 0.3（前 30% 步數探索）
- `exploration_final_eps`: 0.05（最終 5% 探索率）

**Buffer size**：
- 小環境（3×3）: 50k
- 大環境（5×5+）: 100k+

---

## 🎓 擴展方向

### 1. 切換到 Grid 模式

```python
# 只需修改一個參數
env = make_env(obs_type='grid')  # 從 'vector' 改為 'grid'

model = DQN(
    policy="CnnPolicy",  # 從 MlpPolicy 改為 CnnPolicy
    env=env,
    # ... 其他參數
)
```

### 2. 多智能體 RL

當前是單個 agent 訓練，可以擴展到：
- **獨立學習**：每個機器人獨立訓練（已支援）
- **中心化訓練**：訓練一個控制所有機器人的策略
- **MARL 算法**：QMIX, MADDPG 等

### 3. 更複雜的環境

- 加回家具障礙
- 加入垃圾收集任務
- 動態充電座位置
- 更大的地圖（10×10, 20×20）

### 4. 其他 RL 算法

- **PPO**：通常比 DQN 更穩定
- **SAC**：適合連續動作空間
- **A2C/A3C**：多進程訓練

---

## 📚 延伸閱讀

- [DQN 論文](https://arxiv.org/abs/1312.5602)
- [Stable-Baselines3 文檔](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 文檔](https://gymnasium.farama.org/)
- [Reward Shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

---

## 🎉 總結

你現在有了一個**可擴展的 RL 訓練環境**：

✅ **現在**：從簡單開始（Vector + 3×3）
✅ **未來**：輕鬆擴展（Grid + 大地圖）
✅ **統一接口**：代碼無需大改

完美平衡了 **minimum working example** 和 **擴展性**！

開始訓練吧 🚀

```bash
python train_dqn.py --mode vector --steps 100000
```
