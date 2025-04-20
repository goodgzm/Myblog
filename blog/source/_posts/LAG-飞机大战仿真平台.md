---
title: LAG 飞机大战仿真平台
date: 2025-04-20 14:08:19
tags:
categories:
    - RL
---
LAG 是一个轻量级的空战对抗环境，主要包含6种场景任务。 

##  一、安装
```bash
conda create -n LAG python==3.9 
```

```bash
annotated-types==0.7.0
asttokens==3.0.0
certifi==2024.12.14
charset-normalizer==3.4.1
click==8.1.8
cloudpickle==3.1.0
colorama==0.4.6
contourpy==1.1.1
cycler==0.12.1
docker-pycreds==0.4.0
docopt==0.6.2
eval_type_backport==0.2.2
executing==2.1.0
Farama-Notifications==0.0.4
filelock==3.13.1
fonttools==4.55.3
fsspec==2024.2.0
geographiclib==2.0
gitdb==4.0.11
GitPython==3.1.42
gym==0.22.0
gym-notices==0.0.8
gymnasium==1.0.0
icecream==2.1.4
idna==3.10
importlib_metadata==8.5.0
importlib_resources==6.4.5
Jinja2==3.1.3
JSBSim==1.1.6
jsonpickle==3.0.3
kiwisolver==1.4.7
MarkupSafe==2.1.5
matplotlib==3.7.5
mpmath==1.3.0
munch==4.0.0
networkx==3.0
numpy==1.24.4
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==9.1.0.70
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse-cu11==11.7.5.86
nvidia-nccl-cu11==2.21.5
nvidia-nvtx-cu11==11.8.86
packaging==23.2
pillow==10.4.0
platformdirs==4.3.6
protobuf==5.29.3
psutil==6.1.1
py-cpuinfo==9.0.0
pydantic==2.10.5
pydantic_core==2.27.2
Pygments==2.19.1
pymap3d==3.1.0
pyparsing==3.1.4
python-dateutil==2.9.0.post0
PyYAML==6.0.2
requests==2.32.3
sacred==0.8.5
sentry-sdk==2.19.2
setproctitle==1.3.4
Shapely==1.7.1
six==1.17.0
smmap==5.0.1
sympy==1.13.1
torch==2.6.0+cu118
torchaudio==2.6.0+cu118
torchvision==0.21.0+cu118
triton==3.2.0
typing_extensions==4.12.2
urllib3==2.2.3
wandb==0.19.2
wrapt==1.16.0
zipp==3.20.2

```

 

## 二、环境
### 奖励
#### Altitude Reward
**AltitudeReward 主要用于惩罚飞行器在低空飞行时的行为，分为两部分：  
    1.	速度惩罚 (Pv)：当高度低于 safe_altitude 时，根据下降速度 (ego_vz) 进行惩罚。  
   2.	高度惩罚 (PH)：当高度低于 danger_altitude 时，直接给予固定惩罚。**

1. 触发机制  
 •	Pv 触发条件：当 ego_z <= safe_altitude（默认 4.0 km）时触发。  
 •	PH 触发条件：当 ego_z <= danger_altitude（默认 3.5 km）时触发。
2. Reward 计算  
 •	速度惩罚 (Pv)
$$
Pv = -\text{clip} \left( \frac{\text{ego_vz}}{\text{Kv}} \times \frac{\text{safe_altitude} - \text{ego_z}}{\text{safe_altitude}}, 0, 1 \right)
$$
$Pv = -\text{clip} \left( \frac{\text{ego_vz}}{\text{Kv}} \times \frac{\text{safe_altitude} - \text{ego_z}}{\text{safe_altitude}}, 0, 1 \right)$

其中：  
   		•	$ ego_vz / Kv $代表速度对惩罚的影响 (Kv 默认 0.2 mh，即 68 m/s)。  
   	 	•	$ (\text{safe_altitude} - \text{ego_z}) / \text{safe_altitude}  $表示低空飞行的程度，越低惩罚越大。  
   	 	•	clip 限制 Pv 在 [-1, 0] 范围内。  
    	•	高度惩罚 (PH)

$ PH = \text{clip} \left( \frac{\text{ego_z}}{\text{danger_altitude}}, 0, 1 \right) - 1 - 1 $

其中：  
   		•	当 ego_z == danger_altitude 时：PH = -1。  
    		•	当 ego_z == 0 时：PH = -2。  
    		•	低于 danger_altitude 的 ego_z 越小，PH 罚分越大，最差 -2。

<font style="color:#0e0e0e;">•	最终 reward 计算  
</font>$ \text{reward} = Pv + PH $

<font style="color:#0e0e0e;">    •	    Pv 取值范围：[-1, 0]</font>

<font style="color:#0e0e0e;">    •	    PH 取值范围：[-2, -1]</font>

<font style="color:#0e0e0e;">    •	    最终 reward 取值范围：[-3, 0]</font>

---

#### EventDrivenReward
**EventDrivenReward 主要根据特定事件触发奖励或惩罚，核心逻辑如下：  
****    1.	被导弹击落 (-200)  
****    2.	意外坠毁 (-200)  
****    3.	成功击落敌机 (+200)**

1. 触发机制  
 •	$ reward -= 200 $ 触发条件：  
 •	env.agents[agent_id].is_shotdown == True（被导弹击落）。  
 •	env.agents[agent_id].is_crash == True（意外坠毁）。  
 •	$ reward += 200 $ 触发条件：  
 •	env.agents[agent_id].launch_missiles 中有 missile.is_success == True（成功击中敌机）。

---

#### <font style="color:#0e0e0e;">HeadingReward </font>
**<font style="color:#0e0e0e;">HeadingReward 主要用于衡量当前飞机的航向 (heading) 偏差，以及其他关键飞行参数（高度、高度、滚转角和速度）的误差，并计算奖励。</font>**

1. <font style="color:#0e0e0e;">触发机制</font>

<font style="color:#0e0e0e;">奖励的计算方式基于</font><font style="color:#0e0e0e;">高斯函数衰减</font><font style="color:#0e0e0e;">，即偏差越大，奖励越小。</font>

2. <font style="color:#0e0e0e;">reward计算</font>

<font style="color:#0e0e0e;">本函数根据四个飞行变量计算 reward，每个变量的奖励值 r 都由高斯函数计算：</font>

$ 
r = \exp\left(-\left(\frac{\text{误差值}}{\text{误差缩放因子}}\right)^2\right)
 $

![](https://cdn.nlark.com/yuque/0/2025/png/35217974/1742195745951-7758b016-da0a-4f39-a3bf-d65a0f86340d.png)

<font style="color:#0e0e0e;">最终 reward 计算：</font>

$ 
\text{reward} = (heading_r \times alt_r \times roll_r \times speed_r)^{\frac{1}{4}}
 $

<font style="color:#0e0e0e;">	•	这是几何均值，确保所有参数影响均衡，不会被某个单独参数主导。</font>

<font style="color:#0e0e0e;">	•	如果某个参数严重偏离目标值（其 r ≈ 0），则 reward 会大幅下降。</font>

---

#### PostureReward
P**ostureReward 旨在通过战机的朝向和距离来计算奖励，鼓励战机瞄准敌机并靠近敌机，同时惩罚被敌机瞄准或距离过远的情况。**

1. 触发机制涉及的关键变量

在 get_reward 方法中，奖励计算依赖于以下变量：  
    •	ego_feature（己方战机状态）：包含己方战机的位置 (north, east, down) 和速度 (vn, ve, vd)。  
    •	enm_feature（敌方战机状态）：包含敌机的位置和速度。  
    •	AO（Aspect Angle，目标偏向角）：描述敌机相对于己方战机的方向，影响朝向奖励（Orientation Reward）。  
    •	TA（Target Angle，目标指向角）：描述己方战机对敌机的瞄准程度，影响朝向奖励（Orientation Reward）。  
    •	R（Range，距离）：己方战机与敌机的距离，影响距离奖励（Range Reward）。

2. 奖励计算逻辑

总奖励 = 朝向奖励 × 距离奖励

new_reward += orientation_reward * range_reward  
    •	朝向奖励 (orientation_reward)：由 get_orientation_function 确定，不同版本 (v0, v1, v2) 计算方式不同，但核心思想是：  
    •	鼓励战机瞄准敌机 (AO 小、TA 大)。  
    •	惩罚被敌机瞄准 (AO 大)。  
    •	距离奖励 (range_reward)：由 get_range_funtion 确定，不同版本 (v0, v1, v2, v3) 计算方式不同，但核心思想是：  
    •	鼓励接近敌机 (R 小)。  
    •	惩罚距离过远 (R 过大)。

(1) 朝向奖励 (orientation_reward)

计算方式

orientation_reward = self.orientation_fn(AO, TA)

不同版本的计算：  
    •	v0:  
   	  •	通过 tanh 函数鼓励 AO 变小。  
  	  •	TA 越大，奖励越高。  
    •	v1:  
   	 •	AO 靠近 π/2（90°）时，奖励较低，意味着应当减少战机被敌机瞄准的可能性。  
   	 •	TA 越大，奖励越高。  
    •	v2:  
  	 •	AO 越小，奖励越高。  
   	 •	TA 越大，奖励越高。

(2) 距离奖励 (range_reward)

计算方式

range_reward = self.range_fn(R / 1000)

不同版本的计算：  
    •	v0: 使用 exp(-(R - target_dist) ** 2 * 0.004) 作为基准，惩罚远离目标距离的行为，并通过 sigmoid 限制惩罚程度。  
    •	v1: 在 target_dist 附近奖励较高，远离时奖励逐渐降低，最小不低于 0.3。  
    •	v2: 与 v1 类似，但额外增加 np.sign(7 - R)，保证当 R < 7 时有额外奖励，进一步鼓励接近敌机。  
    •	v3:  
  	  •	当 R < 5，奖励为 1。  
  	  •	当 R >= 5，使用二次函数 -0.032 * R^2 + 0.284 * R + 0.38 计算奖励，并通过 clip 限制范围 [0,1]。  
  	  •	额外引入 np.exp(-0.16 * R)，确保奖励不会骤降。

---

#### RelativeAltitudeReward 
_**RelativeAltitudeReward 用于约束战机的相对高度，防止战机在战斗中偏离合理的高度范围。**_

1. 关键变量

在 get_reward 方法中，涉及以下核心变量：  
   	  •  ego_z（己方战机高度）

    •   enm_z（敌机高度）

    •	  KH（高度阈值，默认为 1.0 km）

KH 控制战机允许的相对高度差，即 当己方战机与敌机的高度差超过 KH 时，会受到惩罚。

2. 奖励计算逻辑

计算奖励（惩罚）<font style="color:#0e0e0e;">：</font>

<font style="color:#0e0e0e;">	  •	当 </font>**<font style="color:#0e0e0e;">高度差 </font>**<font style="color:#0e0e0e;">≤ KH， </font>**<font style="color:#0e0e0e;">奖励为 </font>**<font style="color:#0e0e0e;">0，即无惩罚。</font>

<font style="color:#0e0e0e;">	  •	当 </font>**<font style="color:#0e0e0e;">高度差 </font>**<font style="color:#0e0e0e;">> KH，最终 new_reward = 负值，表示惩罚。</font>

---

#### <font style="color:#0e0e0e;">ShootPenaltyReward</font>
**<font style="color:#0e0e0e;">ShootPenaltyReward 旨在惩罚导弹的过度发射，防止一次性倾泻全部导弹，提高战术性。</font>**

1. <font style="color:#0e0e0e;">关键变量</font>

<font style="color:#0e0e0e;">	•    pre_remaining_missiles[agent_id] 表示 上一个时间步agent_id 战机的导弹数量。</font>

<font style="color:#0e0e0e;">	•    task.remaining_missiles 是 当前时间步各战机剩余导弹数量。</font>

<font style="color:#0e0e0e;">	•    task.remaining_missiles[agent_id] 获取 agent_id 战机的导弹数量。</font>

2. <font style="color:#0e0e0e;">reward计算</font>

<font style="color:#0e0e0e;">如果当前导弹数量比上一步少 1（说明刚刚发射了一枚导弹），则给予惩罚 -10</font>

---

### <font style="color:#0e0e0e;">终止条件</font>
#### <font style="color:#0e0e0e;">ExtremeState </font>
**<font style="color:#0e0e0e;">这个 ExtremeState会检查飞机是否处于极端状态（Extreme State），并在满足条件时终止仿真。</font>**

<font style="color:#0e0e0e;">•      env.agents[agent_id]：获取当前飞机（agent_id）的实例。</font>

<font style="color:#0e0e0e;">•      get_property_value(c.detect_extreme_state)：调用 get_property_value 方法，检查 		  detect_extreme_state 这个属性值。</font>

<font style="color:#0e0e0e;">•      bool(...)：将返回值转换为布尔值，如果 detect_extreme_state 为 True 或非零值，则 done = True，表示仿真需要终止。</font>

---

#### LowAltitude
**<font style="color:#0e0e0e;">该 LowAltitude 终止条件类用于检测飞机 高度过低 的情况，并在达到设定的高度阈值时终止仿真。</font>**

<font style="color:#0e0e0e;">•	altitude_limit 是高度阈值，默认 </font><font style="color:#0e0e0e;">2500 米</font><font style="color:#0e0e0e;">。	</font>

<font style="color:#0e0e0e;">•	c.position_h_sl_m：表示 </font><font style="color:#0e0e0e;">海平面高度（高度单位：米）</font><font style="color:#0e0e0e;">。</font>

<font style="color:#0e0e0e;">•	get_property_value(c.position_h_sl_m) 获取当前飞机的海拔高度。</font>

<font style="color:#0e0e0e;">•	如果 </font><font style="color:#0e0e0e;">飞机高度 ≤ 设定的 </font><font style="color:#0e0e0e;">altitude_limit</font><font style="color:#0e0e0e;">（默认为 2500 米）</font><font style="color:#0e0e0e;">，则：</font>

<font style="color:#0e0e0e;">•	done = True，表示仿真应该终止。</font>

#### Overload
<font style="color:#0e0e0e;">该 </font>**<font style="color:#0e0e0e;">Overload 终止条件类用于检测飞机 过载（加速度过高） 的情况，并在满足阈值时终止仿真。</font>**

<font style="color:#0e0e0e;">•	设定 </font><font style="color:#0e0e0e;">X、Y、Z 三个方向的加速度限制</font><font style="color:#0e0e0e;">，默认 </font><font style="color:#0e0e0e;">10g</font><font style="color:#0e0e0e;">（1g ≈ 9.81 m/s²）。</font>

<font style="color:#0e0e0e;">•	c.accelerations_n_pilot_x_norm：X 轴方向的飞行员加速度。</font>

<font style="color:#0e0e0e;">•	c.accelerations_n_pilot_y_norm：Y 轴方向的飞行员加速度。</font>

<font style="color:#0e0e0e;">•	c.accelerations_n_pilot_z_norm：Z 轴方向的飞行员加速度，</font><font style="color:#0e0e0e;">+1 是为了考虑重力加速度</font><font style="color:#0e0e0e;">（即实际测量的加速度应减去 1g）。</font>

<font style="color:#0e0e0e;">•	如果 X、Y、Z 三个方向任意一个加速度超过阈值，则触发终止（flag_overload = True）。</font>

---

#### SafeReturn
**<font style="color:#0e0e0e;">该 SafeReturn 终止条件类用于 检测战斗任务是否完成或飞机是否被摧毁，在满足条件时终止仿真。</font>**

<font style="color:#0e0e0e;">该类定义了 三种终止情况：</font>

<font style="color:#0e0e0e;">•      飞机被击落（Shot down）</font>

<font style="color:#0e0e0e;">•      飞机坠毁（Crash）</font>

<font style="color:#0e0e0e;">•       所有敌机被消灭，且当前飞机未受到攻击</font>

#### Timeout 
**<font style="color:#0e0e0e;">Timeout 终止条件用于 当仿真步数 (current_step) 超过最大限制 (max_steps) 时终止仿真。</font>**

#### UnreachHeading
**<font style="color:#0e0e0e;">UnreachHeading 终止条件用于 检测飞机在限定时间内是否达到了目标航向 (heading)、高度 (altitude)、速度 (velocity)。如果未能满足要求，终止仿真。</font>**

---

### 状态和动作
#### 状态:
1. **<font style="color:#0e0e0e;">delta_altitude: </font>**<font style="color:#0e0e0e;">目标高度差（delta altitude to target) ，用于表示飞行器与目标之间的高度差</font>
2. **<font style="color:#0e0e0e;">delta_heading:</font>**<font style="color:#0e0e0e;"> 表示当前飞行器朝向与目标点方向之间的偏差角（°），用于衡量飞机偏离目标方向的程度。</font>
3. **<font style="color:#0e0e0e;">delta_velocities_u</font>**<font style="color:#0e0e0e;">: 表示飞行器当前速度 u 分量与目标速度 u 分量之间的速度差，单位是 米/秒（m/s）。</font>
4. **<font style="color:#0e0e0e;">position_h_sl_m:</font>**<font style="color:#0e0e0e;"> 表示飞行器的海拔高度（单位：米），即飞行器相对于平均海平面的高度。</font>
5. **<font style="color:#0e0e0e;">attitude_roll_rad: </font>**<font style="color:#0e0e0e;">代表飞行器的滚转角（Roll Angle），单位为弧度（rad), 表示飞行器沿机身纵轴（X 轴)的旋转角度。</font>
6. **<font style="color:#0e0e0e;">attitude_pitch_rad:</font>**<font style="color:#0e0e0e;"> 代表飞行器的俯仰角（Pitch Angle），单位为弧度（rad), 表示飞行器沿机身横轴（Z 轴）的旋转角度</font>
7. **<font style="color:#0e0e0e;">velocities_u_mps:</font>**<font style="color:#0e0e0e;"> 代表飞行器在机体系（Body Frame）X 轴方向的速度，单位为米/秒（m/s）, 对应飞行器的前进方向（即机头指向的方向）。</font>
8. **<font style="color:#0e0e0e;">velocities_v_mps:</font>**<font style="color:#0e0e0e;"> 代表飞行器在机Í体系（Body Frame）Y 轴方向的速度，单位为米/秒（m/s）, 机体系 Y 轴（V 方向） 对应侧向速度（Side Velocity），即飞机相对于自身机体的侧向漂移速度。</font>
9. **<font style="color:#0e0e0e;">velocities_w_mps</font>**<font style="color:#0e0e0e;">: 代表飞行器在机体系（Body Frame）Z 轴方向的速度，单位为米/秒（m/s）。机体系 Z 轴（W 方向） 代表飞机沿机身垂直方向（正下方）的速度，也可称为升降速度（Vertical Speed）。</font>
10. **<font style="color:#0e0e0e;">velocities_vc_mps</font>**<font style="color:#0e0e0e;"> 代表真空速（Calibrated Airspeed, VC），单位是米/秒（m/s）。真空速（VC）用于测量飞机的飞行速度，通常是由皮托管测得的动态压力推算出来的。飞机的 VC 被用来优化航速和推力，而 u, v, w 分量用于计算姿态控制、侧滑角、航向控制等。</font>
11. **<font style="color:#0e0e0e;">velocities_v_north_mps:</font>**<font style="color:#0e0e0e;"> 代表飞机相对于真北方向的速度（单位：米/秒），该变量是只读的（access="R"）。</font>
12. **<font style="color:#0e0e0e;">velocities_v_east_mps</font>**<font style="color:#0e0e0e;"> 代表 飞机相对于正东方向的速度（单位：米/秒），该变量是只读的（access="R"）。</font>
13. **<font style="color:#0e0e0e;">velocities_v_down_mps</font>**<font style="color:#0e0e0e;"> 代表 飞机在地理坐标系下的垂直速度（向下的速度），单位是 米/秒（m/s），并且该变量是 只读的（access="R"）。</font>
14. **<font style="color:#0e0e0e;">position_long_gc_deg</font>**<font style="color:#0e0e0e;"> 代表地理坐标中的地理经度（Geodesic Longitude），单位为度（°），取值范围 [-180, 180]。</font>
15. **<font style="color:#0e0e0e;">position_lat_geod_deg</font>**<font style="color:#0e0e0e;"> 代表地理纬度（Geocentric Latitude），单位为度（°），取值范围 [-90, 90]。</font>
16. **<font style="color:#0e0e0e;">attitude_heading_true_rad</font>**<font style="color:#0e0e0e;"> 代表 真实航向角（True Heading），单位为弧度（radians），该变量是只读的（access="R"）。</font>
    1. <font style="color:#0e0e0e;">航向角（Heading）表示飞机机头相对于地理北极的方向，通常以真北（True North）为基准测量。</font>

<font style="color:#0e0e0e;">		•	0</font><font style="color:#0e0e0e;"> rad</font><font style="color:#0e0e0e;"> → 机头朝向 </font><font style="color:#0e0e0e;">正北</font>

<font style="color:#0e0e0e;">		•	π/2</font><font style="color:#0e0e0e;"> rad</font><font style="color:#0e0e0e;"> → 机头朝向 </font><font style="color:#0e0e0e;">正东</font>

<font style="color:#0e0e0e;">		•	π</font><font style="color:#0e0e0e;"> rad</font><font style="color:#0e0e0e;"> → 机头朝向 </font><font style="color:#0e0e0e;">正南</font>

<font style="color:#0e0e0e;">		•	3π/2</font><font style="color:#0e0e0e;"> rad</font><font style="color:#0e0e0e;"> → 机头朝向 </font><font style="color:#0e0e0e;">正西</font>

<font style="color:#0e0e0e;">		•	2π rad（或 0 rad） → 机头回到 正北</font>

17. **<font style="color:#0e0e0e;">accelerations_n_pilot_x_norm</font>**<font style="color:#0e0e0e;"> 代表飞行员所感受到的机体 X 轴方向的加速度（归一化）</font>
18. **<font style="color:#0e0e0e;">accelerations_n_pilot_y_norm</font>**<font style="color:#0e0e0e;"> 代表飞行员所感受到的机体 Y 轴方向的加速度（归一化）</font>
19. **<font style="color:#0e0e0e;">accelerations_n_pilot_z_norm</font>**<font style="color:#0e0e0e;"> 代表飞行员所感受到的机体 Z 轴方向的加速度（归一化）</font>

---

#### actions
1. **<font style="color:#0e0e0e;">fcs_aileron_cmd_norm</font>**<font style="color:#0e0e0e;"> 代表副翼（aileron）控制指令的归一化值，即飞行控制系统（FCS）对副翼的控制输入，范围在 [-1.0, 1.0] 之间。</font>
    1. <font style="color:#0e0e0e;">副翼是飞机机翼上的控制面，主要用于</font><font style="color:#0e0e0e;">滚转（Roll）控制</font><font style="color:#0e0e0e;">。当飞行员或自动驾驶系统输入</font><font style="color:#0e0e0e;">副翼指令</font><font style="color:#0e0e0e;">时：</font>

<font style="color:#0e0e0e;">		•	</font><font style="color:#0e0e0e;">正值 (</font><font style="color:#0e0e0e;">> 0</font><font style="color:#0e0e0e;">)</font><font style="color:#0e0e0e;">：右侧副翼</font><font style="color:#0e0e0e;">上升</font><font style="color:#0e0e0e;">，左侧副翼</font><font style="color:#0e0e0e;">下降</font><font style="color:#0e0e0e;">，飞机</font><font style="color:#0e0e0e;">向右滚转</font><font style="color:#0e0e0e;">（顺时针）。</font>

<font style="color:#0e0e0e;">		•	</font><font style="color:#0e0e0e;">负值 (</font><font style="color:#0e0e0e;">< 0</font><font style="color:#0e0e0e;">)</font><font style="color:#0e0e0e;">：左侧副翼</font><font style="color:#0e0e0e;">上升</font><font style="color:#0e0e0e;">，右侧副翼</font><font style="color:#0e0e0e;">下降</font><font style="color:#0e0e0e;">，飞机</font><font style="color:#0e0e0e;">向左滚转</font><font style="color:#0e0e0e;">（逆时针）。</font>

<font style="color:#0e0e0e;">		•	零 (0)：副翼处于中立位置，飞机不发生滚转（如果没有外部干扰）。</font>

2. **<font style="color:#0e0e0e;">fcs_elevator_cmd_norm</font>**<font style="color:#0e0e0e;"> 代表升降舵（elevator）控制指令的归一化值，用于控制飞机的俯仰（Pitch）运动，范围在 [-1.0, 1.0] 之间</font>
    1. <font style="color:#0e0e0e;">升降舵(elevator)的作用升降舵通常位于飞机水平尾翼上，主要用于控制飞机的俯仰角（Pitch），进而影响高度（altitude）和爬升率（climb rate）。当飞行员或自动驾驶系统输入升降舵指令时：</font>

<font style="color:#0e0e0e;">			•	</font><font style="color:#0e0e0e;">正值 (</font><font style="color:#0e0e0e;">> 0</font><font style="color:#0e0e0e;">)</font><font style="color:#0e0e0e;">：</font><font style="color:#0e0e0e;">升降舵向上偏转</font><font style="color:#0e0e0e;">，增大尾部下压力，使飞机</font><font style="color:#0e0e0e;">机头上仰</font><font style="color:#0e0e0e;">，即</font><font style="color:#0e0e0e;">上升</font><font style="color:#0e0e0e;">。</font>

<font style="color:#0e0e0e;">			•	</font><font style="color:#0e0e0e;">负值 (</font><font style="color:#0e0e0e;">< 0</font><font style="color:#0e0e0e;">)</font><font style="color:#0e0e0e;">：</font><font style="color:#0e0e0e;">升降舵向下偏转</font><font style="color:#0e0e0e;">，减少尾部下压力，使飞机</font><font style="color:#0e0e0e;">机头下俯</font><font style="color:#0e0e0e;">，即</font><font style="color:#0e0e0e;">下降</font><font style="color:#0e0e0e;">。</font>

<font style="color:#0e0e0e;">			•	零 (0)：升降舵处于中立位置，飞机维持当前俯仰角（若无外部干扰）。</font>

3. **<font style="color:#0e0e0e;">fcs_rudder_cmd_norm</font>**<font style="color:#0e0e0e;"> 代表方向舵（rudder）控制指令的归一化值，用于控制飞机的偏航（Yaw）运动，范围在 [-1.0, 1.0] 之间</font>
    1. <font style="color:#0e0e0e;">方向舵位于飞机垂直尾翼上，主要用于控制飞机的偏航角（Yaw），即机头左右转向的角度。当飞行员或自动驾驶系统输入方向舵指令时：</font>

<font style="color:#0e0e0e;">	</font><font style="color:#0e0e0e;">•</font><font style="color:#0e0e0e;">	</font><font style="color:#0e0e0e;">正值 (</font><font style="color:#0e0e0e;">> 0</font><font style="color:#0e0e0e;">)</font><font style="color:#0e0e0e;">：</font><font style="color:#0e0e0e;">方向舵向左偏转</font><font style="color:#0e0e0e;">，增加左侧空气阻力，使飞机</font><font style="color:#0e0e0e;">机头向右偏航</font><font style="color:#0e0e0e;">。</font>

<font style="color:#0e0e0e;">	</font><font style="color:#0e0e0e;">•</font><font style="color:#0e0e0e;">	</font><font style="color:#0e0e0e;">负值 (</font><font style="color:#0e0e0e;">< 0</font><font style="color:#0e0e0e;">)</font><font style="color:#0e0e0e;">：</font><font style="color:#0e0e0e;">方向舵向右偏转</font><font style="color:#0e0e0e;">，增加右侧空气阻力，使飞机</font><font style="color:#0e0e0e;">机头向左偏航</font><font style="color:#0e0e0e;">。</font>

<font style="color:#0e0e0e;">	•	零 (0)：方向舵处于中立位置，飞机维持当前航向（若无外部干扰）。</font>

4. **<font style="color:#0e0e0e;">fcs_throttle_cmd_norm </font>**<font style="color:#0e0e0e;">代表油门（Throttle）控制指令的归一化值，用于控制飞机的发动机推力，范围在 [0.0, 0.9] 之间。</font>

---

### 任务
**<font style="background-color:#E7E9E8;">tips: 各任务对应不同状态空间，动作空间, 终止条件， 奖励</font>**

#### HeadingTask
```python
observation(dim 12):
    - [0] ego delta altitude      (unit: km)
    - [1] ego delta heading       (unit: rad)
    - [2] ego delta velocities_u  (unit: mh)
    - [3] ego_altitude            (unit: 5km)
    - [4] ego_roll_sin
    - [5] ego_roll_cos
    - [6] ego_pitch_sin
    - [7] ego_pitch_cos
    - [8] ego v_body_x            (unit: mh)
    - [9] ego v_body_y            (unit: mh)
    - [10] ego v_body_z           (unit: mh)
    - [11] ego_vc                 (unit: mh)
```

```python
actions(dim 4):
    - [0] fcs_aileron_cmd_norm
    - [1] fcs_elevator_cmd_norm
    - [2] fcs_rudder_cmd_norm
    - [3] fcs_throttle_cmd_norm
```

```python
self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
        ]
```

```python
self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
```

---



#### SingleCombatTask
```python
observation(dim 15):
    - ego info
        - [0] ego_altitude         
        - [1] ego_roll_sin
        - [2] ego_roll_cos
        - [3] ego_pitch_sin
        - [4] ego_pitch_cos
        - [5] ego_v_body_x         
        - [6] ego_v_body_y          
        - [7] ego_v_body_z          
        - [8] ego_vc              
    - relative enm info
        - [9] delta_v_body_x         (unit: mh)# 敌方 v_body_x 与自身的速度差
        - [10] delta_altitude        (unit: km)# 敌方与自身的高度差
        - [11] ego_AO                (unit: rad) [0, pi]#自身相对于敌方的朝向角度差
        - [12] ego_TA                (unit: rad) [0, pi]#描述敌方相对于自身的方向
        - [13] relative_distance     (unit: 10km)#敌方距离
        - [14] side_flag             1 or 0 or -1 #左右侧标记    
```

```python
actions(dim 4):
    - [0] fcs_aileron_cmd_norm
    - [1] fcs_elevator_cmd_norm
    - [2] fcs_rudder_cmd_norm
    - [3] fcs_throttle_cmd_norm
```

```python
self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config)
        ]
```

```python
self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            SafeReturn(self.config),
            Timeout(self.config),
        ]
```

---



#### SingleCombatDodgeMissileTask
```python
observation(dim 21):
    - ego info
            - [0] ego_altitude          
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego_v_body_x           
            - [6] ego_v_body_y           
            - [7] ego_v_body_z           
            - [8] ego_vc                
        - relative enm info
            - [9] delta_v_body_x         
            - [10] delta_altitude        
            - [11] ego_AO               
            - [12] ego_TA               
            - [13] relative_distance     
            - [14] side_flag             
        - relative missile info
            - [15] delta_v_body_x #计算导弹的速度与本机的速度差
            - [16] delta_altitude #计算导弹的高度（missile_feature[2]）与飞机高度的差值
            - [17] ego_AO #自身相对于导弹的朝向角度差
            - [18] ego_TA #描述导弹相对于自身的方向
            - [19] relative_distance #本体与导弹距离
            - [20] side flag #记录导弹在本机左侧 (-1) 或右侧 (1)
```

```python
actions(dim 4):
    - [0] fcs_aileron_cmd_norm
    - [1] fcs_elevator_cmd_norm
    - [2] fcs_rudder_cmd_norm
    - [3] fcs_throttle_cmd_norm
```

```python
self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]
```

```python
self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            SafeReturn(self.config),
            Timeout(self.config),
        ]
```

---



#### SingleCombatShootMissileTask
```python
observation(dim 21):
    - ego info
            - [0] ego altitude        
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x     
            - [6] ego v_body_y     
            - [7] ego v_body_z        
            - [8] ego_vc               
        - relative enm info
            - [9] delta_v_body_x        
            - [10] delta_altitude       
            - [11] ego_AO             
            - [12] ego_TA             
            - [13] relative distance   
            - [14] side_flag           
        - relative missile info
            - [15] delta_v_body_x 
            - [16] delta altitude
            - [17] ego_AO 
            - [18] ego_TA 
            - [19] relative distance 
            - [20] side flag 
```

```python
actions(dim 5):
    - [0] fcs_aileron_cmd_norm
    - [1] fcs_elevator_cmd_norm
    - [2] fcs_rudder_cmd_norm
    - [3] fcs_throttle_cmd_norm
    - [4] shoot_action
```

```python
self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]
```

```python
self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            SafeReturn(self.config),
            Timeout(self.config),
        ]
```

---



#### MultipleCombatTask
```python
observation(dim 9 + 6 * n): #n为总共partner和enemy数
    - ego info
            - [0] ego altitude        
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x         
            - [6] ego v_body_y          
            - [7] ego v_body_z           
            - [8] ego_vc                
        - relative partner and enm info #i为机器编号(i < n)
            - [9 + 6 * i] delta_v_body_x         
            - [10 + 6 * i] delta_altitude        
            - [11 + 6 * i] ego_AO                
            - [12 + 6 * i] ego_TA               
            - [13 + 6 * i] relative distance    
            - [14 + 6 * i] side_flag         
```

```python
actions(dim 4):
    - [0] fcs_aileron_cmd_norm
    - [1] fcs_elevator_cmd_norm
    - [2] fcs_rudder_cmd_norm
    - [3] fcs_throttle_cmd_norm
```

```python
self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            EventDrivenReward(self.config)
        ]  
```

```python
self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
```

---



#### MultipleCombatShootTask
```python
observation(dim 15 + 6 * n): #n为总共partner和enemy数
    - ego info
            - [0] ego altitude          
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x         
            - [6] ego v_body_y         
            - [7] ego v_body_z          
            - [8] ego_vc                
        - relative partner and enm info #i为机器编号(i < n)
            - [9 + 6 * i] delta_v_body_x        
            - [10 + 6 * i] delta_altitude       
            - [11 + 6 * i] ego_AO               
            - [12 + 6 * i] ego_TA               
            - [13 + 6 * i] relative distance    
            - [14 + 6 * i] side_flag          
         - relative missile info
            - [15 + 6 * (n - 1)] delta_v_body_x
            - [16 + 6 * (n - 1)] delta altitude
            - [17 + 6 * (n - 1)] ego_AO 
            - [18 + 6 * (n - 1)] ego_TA 
            - [19 + 6 * (n - 1)] relative distance 
            - [20 + 6 * (n - 1)] side flag 
```

```python
actions(dim 5):
    - [0] fcs_aileron_cmd_norm
    - [1] fcs_elevator_cmd_norm
    - [2] fcs_rudder_cmd_norm
    - [3] fcs_throttle_cmd_norm
    - [4] shoot_action
```

```python
self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]
```

```python
self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
```

---



## 三、训练与评估
### 训练
#### 特点
支持多种训练模式:  
vs-baseline: 对抗预设的基线策略  
selfplay: 自我对抗训练  
hierarchical: 分层控制架构

提供先验知识:  
导弹发射的距离和角度约束  
使用Beta分布作为先验

---

#### 脚本运行
```bash
cd scripts
bash train_*.sh
```

**Heading Task**

+ <u>train_heading.sh</u>

[**SingleCombatTask**](#F8kL2)

+ <u>train_vsbaseline.sh</u>
+ <u>train_selfplay.sh</u>

[**SingleCombatDodgeMissile**](#ux3rl)**Task**

+ <u>train_vsbaseline_dodge.sh</u>

[**SingleCombatShootMissile**](#VJFWO)**Task**

+ <u>train_vsbaseline_shoot.sh</u>
+ <u>train_selfplay_shoot.sh</u>

[**MultipleCombatTask**](#ToVZZ)

+ <u>train_vsbaseline_share.sh</u>
+ <u>train_selfplay_share.sh</u>

[**MultipleCombatShootTask**](#xkc8y)

+ <u>train_vsbaseline_shoot_share.sh</u>
+ <u>train_selfplay_shoot_share.sh</u>

---

### 评估
#### 特点
<font style="color:rgb(31, 35, 40);">这将生成一个 </font>`<font style="color:rgb(31, 35, 40);">*.acmi</font>`<font style="color:rgb(31, 35, 40);"> 文件。我们可以使用 </font>[TacView](https://www.tacview.net/)<font style="color:rgb(31, 35, 40);">（一种通用的飞行分析工具）来打开文件并观看渲染视频。</font>

---

#### <font style="color:rgb(31, 35, 40);">脚本运行</font>
```bash
cd renders
python render*.py
```

---

# 四、拓展
### player控制
+ free fly: <u>human_free_fly.sh</u>
+ player vs player: <u>human_1v1.sh</u>
+ player vs agent: <u>human_vsbaseline.sh</u>
+ ~~players vs players ~~（未实现）
+ ~~players vs agents ~~（未实现）



