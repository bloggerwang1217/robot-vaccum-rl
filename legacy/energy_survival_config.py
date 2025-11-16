"""
能量求生模式配置檔案
專注於多機器人的能量管理和群體動態
"""

# 基礎模式：3×3 房間，標準參數
BASE_CONFIG = {
    'n': 3,                 # 3×3 房間
    'initial_energy': 100,  # 初始能量
    'e_move': 1,            # 移動消耗 1 能量
    'e_charge': 5,          # 充電增加 5 能量
    'e_collision': 3,       # 碰撞消耗 3 能量
    'n_steps': 500,         # 500 回合
    'epsilon': 0.2          # 20% 探索率
}

# 高探索模式：增加隨機性
HIGH_EXPLORATION_CONFIG = {
    'n': 3,
    'initial_energy': 100,
    'e_move': 1,
    'e_charge': 5,
    'e_collision': 3,
    'n_steps': 500,
    'epsilon': 0.5          # 50% 探索率（高隨機性）
}

# 低探索模式：更理性的行為
LOW_EXPLORATION_CONFIG = {
    'n': 3,
    'initial_energy': 100,
    'e_move': 1,
    'e_charge': 5,
    'e_collision': 3,
    'n_steps': 500,
    'epsilon': 0.1          # 10% 探索率（高理性）
}

# 純理性模式：完全理性求生
PURE_RATIONAL_CONFIG = {
    'n': 3,
    'initial_energy': 100,
    'e_move': 1,
    'e_charge': 5,
    'e_collision': 3,
    'n_steps': 500,
    'epsilon': 0.0          # 0% 探索率（純理性）
}

# 純隨機模式：完全隨機探索
PURE_RANDOM_CONFIG = {
    'n': 3,
    'initial_energy': 100,
    'e_move': 1,
    'e_charge': 5,
    'e_collision': 3,
    'n_steps': 500,
    'epsilon': 1.0          # 100% 探索率（純隨機）
}

# 能量緊張模式：低初始能量，考驗生存能力
ENERGY_SCARCE_CONFIG = {
    'n': 3,
    'initial_energy': 50,   # 低初始能量
    'e_move': 2,            # 高移動消耗
    'e_charge': 3,          # 低充電恢復
    'e_collision': 5,       # 高碰撞懲罰
    'n_steps': 500,
    'epsilon': 0.2
}

# 能量充裕模式：高初始能量，更容易生存
ENERGY_ABUNDANT_CONFIG = {
    'n': 3,
    'initial_energy': 200,  # 高初始能量
    'e_move': 1,            # 低移動消耗
    'e_charge': 10,         # 高充電恢復
    'e_collision': 2,       # 低碰撞懲罰
    'n_steps': 500,
    'epsilon': 0.2
}

# 大地圖模式：5×5 房間
LARGE_MAP_CONFIG = {
    'n': 5,                 # 5×5 房間（更大的空間）
    'initial_energy': 150,  # 需要更多能量
    'e_move': 1,
    'e_charge': 5,
    'e_collision': 3,
    'n_steps': 1000,        # 更長的回合數
    'epsilon': 0.2
}

# 超小地圖模式：2×2 房間（極限擁擠）
TINY_MAP_CONFIG = {
    'n': 2,                 # 2×2 房間（4個格子剛好4台機器人）
    'initial_energy': 50,
    'e_move': 1,
    'e_charge': 5,
    'e_collision': 5,       # 高碰撞懲罰（擁擠環境）
    'n_steps': 200,
    'epsilon': 0.2
}

# 快速測試模式：短回合，快速驗證
QUICK_TEST_CONFIG = {
    'n': 3,
    'initial_energy': 100,
    'e_move': 1,
    'e_charge': 5,
    'e_collision': 3,
    'n_steps': 100,         # 只有 100 回合
    'epsilon': 0.2
}

# 長期模擬模式：觀察長期動態
LONG_RUN_CONFIG = {
    'n': 3,
    'initial_energy': 100,
    'e_move': 1,
    'e_charge': 5,
    'e_collision': 3,
    'n_steps': 2000,        # 2000 回合
    'epsilon': 0.2
}


def get_config(mode='base'):
    """
    根據模式名稱獲取配置

    Args:
        mode: 配置模式名稱

    Returns:
        配置字典
    """
    configs = {
        'base': BASE_CONFIG,
        'high_explore': HIGH_EXPLORATION_CONFIG,
        'low_explore': LOW_EXPLORATION_CONFIG,
        'pure_rational': PURE_RATIONAL_CONFIG,
        'pure_random': PURE_RANDOM_CONFIG,
        'energy_scarce': ENERGY_SCARCE_CONFIG,
        'energy_abundant': ENERGY_ABUNDANT_CONFIG,
        'large': LARGE_MAP_CONFIG,
        'tiny': TINY_MAP_CONFIG,
        'quick': QUICK_TEST_CONFIG,
        'long': LONG_RUN_CONFIG
    }

    return configs.get(mode, BASE_CONFIG).copy()


if __name__ == '__main__':
    # 顯示所有配置
    print("可用的能量求生配置模式：\n")

    configs = {
        '基礎模式 (base)': BASE_CONFIG,
        '高探索 (high_explore)': HIGH_EXPLORATION_CONFIG,
        '低探索 (low_explore)': LOW_EXPLORATION_CONFIG,
        '純理性 (pure_rational)': PURE_RATIONAL_CONFIG,
        '純隨機 (pure_random)': PURE_RANDOM_CONFIG,
        '能量緊張 (energy_scarce)': ENERGY_SCARCE_CONFIG,
        '能量充裕 (energy_abundant)': ENERGY_ABUNDANT_CONFIG,
        '大地圖 (large)': LARGE_MAP_CONFIG,
        '超小地圖 (tiny)': TINY_MAP_CONFIG,
        '快速測試 (quick)': QUICK_TEST_CONFIG,
        '長期模擬 (long)': LONG_RUN_CONFIG
    }

    for name, config in configs.items():
        print(f"=== {name} ===")
        for key, value in config.items():
            if key == 'epsilon':
                print(f"  {key}: {value} ({value*100:.0f}%)")
            else:
                print(f"  {key}: {value}")
        print()
