import numpy as np
from typing import List

def inertial_navigation_filter(data: List[List[float]],
                               auto_threshold: bool = True
                               ) -> List[float]:
    """
    惯导数据滤波函数

    参数：
        data: 输入数据列表
        auto_threshold: 是否启用自适应阈值（默认启用）

    返回：
        滤波后的[经度, 纬度, 航向]
    """
    if not data:
        return []

    data_array = np.array(data)
    lons = data_array[:, 0]
    lats = data_array[:, 1]
    headings = data_array[:, 2]

    # 各维度异常检测
    lon_mask = mad_mask(lons, auto_threshold=auto_threshold)
    lat_mask = mad_mask(lats, auto_threshold=auto_threshold)
    head_mask = mad_mask(headings, is_angle=True, auto_threshold=auto_threshold)

    # 联合过滤
    valid_mask = lon_mask & lat_mask & head_mask
    filtered_data = data_array[valid_mask]

    # 有效性检查（至少保留50%数据）
    min_valid = max(len(data) // 2, 1)
    if len(filtered_data) < min_valid:
        print(f"[系统警告] 有效数据不足，回退原始数据（{len(filtered_data)}/{len(data)}）")
        filtered_data = data_array
    else:
        print(f"数据量合规（{len(filtered_data)}/{len(data)}）")

    # 加权平均值计算（离群值权重衰减）
    final_lons = med_mean(filtered_data[:, 0])
    final_lats = med_mean(filtered_data[:, 1])
    final_head = angle_mean(filtered_data[:, 2])

    return [final_lons, final_lats, final_head]

def mad_mask(values: np.ndarray,
             threshold: float = 3.0,
             is_angle: bool = False,
             auto_threshold: bool = False
             ) -> np.ndarray:
    """
    基于中位数和绝对中位差计算数据掩膜

    参数：
        values: 输入数据数组
        threshold: 固定阈值=3.0（当auto_threshold=False时使用）
        is_angle: 是否为角度数据（需要环形处理）
        auto_threshold: 是否启用自适应阈值计算

    返回：
        布尔掩膜数组（True表示正常值）
    """
    if len(values) == 0:
        return np.array([], dtype=bool)

    if is_angle:
        vectors = angle_vectors(angles=values)
        mean_vector = np.mean(vectors)
        med_rad = np.angle(mean_vector)
        med = np.rad2deg(med_rad) % 360
        diffs = np.abs(values - med)
        circular_diffs = np.minimum(diffs, 360 - diffs)
        mad = np.median(circular_diffs)
    else:
        med = np.median(values)
        diffs = np.abs(values - med)
        mad = np.median(diffs)

    # 零MAD掩膜
    if mad < 1e-9:
        return diffs <= 1e-9 if not is_angle else circular_diffs <= 1e-9

    # 动态阈值计算
    final_threshold = adaptive_threshold(values, mad, is_angle) if auto_threshold else threshold

    # 非零MAD掩膜
    return diffs <= final_threshold * mad if not is_angle else circular_diffs <= final_threshold * mad

def adaptive_threshold(values: np.ndarray,
                       mad: float,
                       is_angle: bool = False
                       ) -> float:
    """动态计算自适应阈值，考虑数据分布特性和样本量

    参数：
        values: 输入数据数组（一维）
        mad: 绝对中位差
        is_angle: 是否为角度数据

    返回：
        优化后的阈值（float）
    """
    n = len(values)

    # 基线阈值（样本量越大越严格）
    base_threshold = 3.0 - 0.5 * np.log10(n) if n > 1 else 3.0

    # 根据数据离散程度调整
    if is_angle:
        # 角度数据：环形离散度评估
        vectors = angle_vectors(angles=values)
        r = np.abs(np.mean(vectors))  # 平均合成向量长度（0~1）
        dispersion = 1 - r  # 离散度（0=完全集中，1=完全分散）
        adjustment = 1 + 2 * dispersion  # 离散度高时放宽阈值
    else:
        # 普通数据：基于变异系数（CV）调整
        med = np.median(values)
        cv = mad / med if med != 0 else 0
        adjustment = 1 + cv  # 数据波动大时放宽阈值

    return np.clip(base_threshold * adjustment, 1.0, 5.0)  # 限制在合理范围

def med_mean(values: np.ndarray) -> float:
    """基于中位数距离的加权平均值"""
    med = np.median(values)
    weights = 1 / (1 + np.abs(values - med))
    return float(np.average(values, weights=weights))

def angle_mean(angles: List[float]) -> float:
    """计算角度数据的环形均值"""
    rad = np.deg2rad(angles)
    vectors = np.exp(1j * rad)
    return np.rad2deg(np.angle(np.mean(vectors))) % 360

def angle_vectors(angles):
    """角度转单位向量"""
    rad = np.deg2rad(angles)
    vectors = np.exp(1j * rad)
    return vectors


if __name__ == "__main__":
    # 测试数据（增强异常场景覆盖）
    test_data = (
            [[116.404, 39.915, 45] for _ in range(18)] +  # 正常数据集群
            [[116.512, 40.123, 300],  # 经纬度异常
             [116.300, 39.900, 30],  # 航向异常
             [116.404, 39.915, 355],  # 边界正常值
             [116.404, 39.915, 360]]  # 角度循环边界测试
    )
    result = inertial_navigation_filter(test_data, auto_threshold=True)
    print(f"滤波结果：{result}")