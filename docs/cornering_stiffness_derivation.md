# 前/后轴侧偏刚度推算说明

`sim/configs/default.yaml` 的 `truck_trailer_vehicle` 段当前用 **Cf = 264 kN/rad、Cr = 335 kN/rad**。本文说明这两个数怎么算出来的。

## 数据来源

### 直接从 xlsx 拿到的

`sim/configs/L4电拖头_首台车_传感器位置参数及车辆参数_20241030-集成答复.xlsx`,Sheet1:

| 行 | 内容 | 值 |
|---|---|---|
| R76 | 空载轴荷 | 前桥 **4712 kg** / 中桥+后桥 **4588 kg** |
| R102 | 轮胎型号 | **295/80R22.5-18** |
| R95 | 驱动形式 | **6×4**(前轴 2 胎、2 轴 + 3 轴各双胎共 8 胎) |

xlsx 里 R54 / R55 "前/后轴侧偏刚度" 这两行**数值列是空的**,所以必须自己推算。

### 经验关系(轮胎工程文献)

商用车径向胎在线性区:

$$C_\alpha \approx k \cdot F_z$$

其中 $k$ 是轮胎特性系数,随载荷下降。**295/80R22.5** 这类重卡 22.5 寸径向胎的经验范围(来自 mchenry 软件公开资料、Gillespie《Fundamentals of Vehicle Dynamics》、SAE 912677):

| 单胎载荷区间 | k(per rad) |
|---|---|
| 轻载(~12 kN) | ~7.4 |
| 中载(~24 kN) | ~5.7 |
| 重载(~31 kN) | ~4.6-5.0 |

## 工况选择

按 yaml 当前配置:`m_t = 9300 kg`、`default_trailer_mass_kg = 0` —— 即**单机空载**。所以用 R76 的空载轴荷,不用 R77 的满载。

## 前轴 Cf

```
前轴载荷 = 4712 kg × 9.81 = 46.2 kN
单胎载荷 = 46.2 / 2 = 23.1 kN          (前轴只有 2 个转向胎)
单胎 C_α = 5.7 × 23.1 = 132 kN/rad     (k 取中载值)
前轴 Cf  = 2 × 132 = 264 kN/rad
```

## 后轴组 Cr

注意 6×4 构型下 2 轴和 3 轴**每根桥左右各双胎,共 8 个轮胎**承担后轴组载荷。

```
后轴组载荷 = 4588 kg × 9.81 = 45.0 kN
单胎载荷  = 45.0 / 8 = 5.6 kN          (8 胎平分,每胎严重欠载)
单胎 C_α  = 7.4 × 5.6 = 42 kN/rad      (k 取轻载值)
后轴组 Cr = 8 × 42 = 335 kN/rad
```

## 写进 yaml 的值

```yaml
truck_trailer_vehicle:
  Cf: 264000.0    # N/rad,前轴 1 桥 2 胎
  Cr: 335000.0    # N/rad,后轴组 2+3 轴共 8 胎合并
```

## 注意事项

- **单轨模型**:plant 用单轨自行车模型,Cf 是"前轴整体侧偏刚度"(2 胎合并),Cr 是"后轴组整体侧偏刚度"(8 胎合并),不是单胎值。
- **载荷敏感**:换载荷工况(满载、带挂车)下这两个值会显著变化。粗略估算:
  - 满载单机:Cf ~ 284,Cr ~ 1040(后轴载荷 181.7 kN,k 回到 5.5)
  - 单机空载后胎欠载严重(每胎 5.6 kN,远低于额定 ~33 kN/胎),线性外插 k=7.4 是工程估算,实测可能 ±30% 偏差
- **不同工况要重算**:如果以后改 `m_t` 或 `default_trailer_mass_kg`,建议同步把 Cf/Cr 按上面公式重新推算一遍。
- **更精确做法**:让供应商给 295/80R22.5 这款轮胎的 Pacejka 侧偏刚度曲线(把 R54/R55 填上),或用实车做 sine-sweep / step-steer 实验反辨识。

## 参考文献

- xlsx R76 / R102 — 整车厂供应商提供
- [mchenry Truck/Trailer Tire Properties](https://www.mchenrysoftware.com/medit32/readme/msmac/truck.trailertireproperties.htm)
- SAE 912677 — Measurement of Radial Truck Tire Dry Cornering Characteristics
- Gillespie, "Fundamentals of Vehicle Dynamics", SAE International
- Pacejka, "Tyre and Vehicle Dynamics", Butterworth-Heinemann
