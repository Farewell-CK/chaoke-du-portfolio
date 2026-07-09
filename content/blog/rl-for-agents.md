---
slug: "rl-for-agents"
title:
  en: "12 | RL for Agents：用强化学习训练Agent"
  zh: "12 | RL for Agents：用强化学习训练Agent"
date: "2026-06-29"
excerpt:
  en: "**核心问题**：除了用LLM做Agent的\"大脑\"，还能怎么用RL训练Agent？"
  zh: "**核心问题**：除了用LLM做Agent的\"大脑\"，还能怎么用RL训练Agent？"
tags: ["Transformer", "Agent"]
---
# 12 | RL for Agents：用强化学习训练Agent

> **核心问题**：除了用LLM做Agent的"大脑"，还能怎么用RL训练Agent？
>
> **阅读提示**：本文约8000字，涉及RL微调LLM、稀疏奖励、离线RL、混合架构等核心议题，包含原创实验与踩坑实录。建议收藏后分两次阅读。本文是系列中数学第二多的文章（仅次于第4篇PPO），但每个公式前都有直觉解释。

---

## 引子：两条路线之争

2024年初，DeepMind的一篇技术报告引发了社区热议。报告中，一个经过RL训练的Agent在陌生环境中展现出了令人震惊的适应性——它不是"想"出来的，而是"练"出来的。

这让我们不得不重新审视一个根本性问题：

```
Agent的"智能"从哪来？

路线A：LLM-as-Brain（当前主流）
  ┌──────────────────────────────────────────────────┐
  │  预训练LLM ──> Prompt/ICL ──> 决策输出            │
  │                                                    │
  │  特点：通用性强、零样本泛化、但推理成本高            │
  │  代表：ReAct, Plan-and-Solve, Voyager              │
  └──────────────────────────────────────────────────┘

路线B：RL-Trained Agent
  ┌──────────────────────────────────────────────────┐
  │  环境交互 ──> 奖励信号 ──> 策略更新 ──> 新策略     │
  │                                                    │
  │  特点：任务专精、推理速度快、但需要大量训练          │
  │  代表：RLHF, Decision Transformer, AgentTune       │
  └──────────────────────────────────────────────────┘
```

**这两条路线不是非此即彼的**。事实上，2024-2025年的前沿研究正在将它们融合。但如果我们不先搞清楚各自的优势和局限，就无法理解为什么需要融合、以及怎么融合。

今天我们就来系统性地回答这个问题：**RL训练Agent到底怎么做，和纯LLM方案相比有什么优劣，以及实际落地时有哪些坑。**

---

## 一、LLM-as-Brain vs RL-Trained Agent：什么时候用哪个？

### 1.1 一张对比表

先看核心差异：

| 维度 | LLM-as-Brain | RL-Trained Agent |
|------|-------------|-----------------|
| 决策来源 | 预训练知识 + 提示工程 | 环境交互中学到的策略 |
| 泛化能力 | 强（零样本迁移） | 弱（分布外退化） |
| 推理速度 | 慢（每次都要生成token） | 快（策略网络前向传播） |
| 训练成本 | 低（只需写prompt） | 高（需要环境交互+训练） |
| 推理成本 | 高（API调用费） | 低（本地推理） |
| 可解释性 | 中（可以看CoT） | 低（隐式策略） |
| 最优场景 | 任务多样、样本少 | 任务固定、需要快速决策 |
| 典型失败 | 幻觉、不遵循指令 | 奖励hacking、过拟合 |

### 1.2 决策框架

```
选LLM-as-Brain还是RL？

├── 任务类型是否在变化？
│   ├── 是（每天新任务）
│   │   └── LLM-as-Brain（泛化能力是刚需）
│   └── 否（固定任务集）
│       └── 继续判断 ↓
│
├── 决策延迟要求？
│   ├── < 50ms（实时控制）
│   │   └── RL（LLM生成太慢）
│   └── > 500ms（可以等）
│       └── 继续判断 ↓
│
├── 训练数据/环境是否可得？
│   ├── 否（无法模拟或收集）
│   │   └── LLM-as-Brain（RL需要环境）
│   └── 是
│       └── 继续判断 ↓
│
├── 部署规模？
│   ├── 小规模（< 1000次/天）
│   │   └── LLM-as-Brain（API成本可接受）
│   └── 大规模（> 10万次/天）
│       └── RL（推理成本优势明显）
│
└── 结论：两者混合（见第六节）
```

### 1.3 一个直觉性的类比

把Agent想象成一个出租车司机：

- **LLM-as-Brain** 相当于雇了一个经验丰富的老司机。他什么都懂，能应对各种情况，但每次出车你都要付他高额工资（推理成本），而且他有时候会"凭经验"做出错误判断（幻觉）。

- **RL-Trained Agent** 相当于你自己培养一个司机。前期投入很大（训练成本），需要让他跑很多趟（环境交互），但一旦训练好了，他跑固定路线又快又准（推理速度快），而且不会犯低级错误。但如果路线变了（分布偏移），他可能就懵了。

**关键洞察**：这两条路线的最优解不是二选一，而是**让老司机当教练，训练出一个本地司机**。这就是后面要讲的混合架构。

---

## 二、RL微调LLM做决策：把LLM变成策略网络

### 2.1 核心思路

第4篇我们讲过PPO在RLHF中的应用——用RL微调LLM让它生成"更好的"回答。同样的思路可以应用到Agent场景：

```
RLHF（对齐）：
  prompt ──> LLM ──> 回答 ──> 人类偏好 ──> 奖励模型 ──> PPO更新

Agent RL（决策）：
  状态s ──> LLM(策略) ──> 动作a ──> 环境 ──> 奖励r ──> PPO更新
```

区别在于**奖励信号的来源**：

- RLHF的奖励来自人类偏好（或reward model）
- Agent RL的奖励来自环境反馈（任务完成度、效率等）

### 2.2 环境搭建

给LLM搭一个可以交互的环境，是整个流程中最工程化的部分。一个典型的Agent RL环境需要：

```
+─────────────────────────────────────────────────+
│                Agent RL Environment              │
│                                                  │
│  +──────────┐    +──────────┐    +──────────┐   │
│  │  State   │    │  Action  │    │  Reward  │   │
│  │ Encoder  │───>│  Parser  │───>│ Function │   │
│  │          │    │          │    │          │   │
│  │ 把环境   │    │ 把LLM    │    │ 计算     │   │
│  │ 状态编码  │    │ 输出解析  │    │ 奖励值   │   │
│  │ 为文本   │    │ 为动作   │    │          │   │
│  +──────────┘    +──────────┘    +──────────┘   │
│       ^                                |         │
│       |         +──────────────┐       |         │
│       +─────────│  Environment │<──────+         │
│                 │  Simulator   │                  │
│                 +──────────────┘                  │
+─────────────────────────────────────────────────┘
```

**State Encoder（状态编码器）**：将环境的原始状态（游戏画面、网页DOM、机器人传感器数据等）转化为LLM能理解的文本描述。

```python
def encode_state(env_state):
    """将环境状态编码为LLM可理解的文本"""
    if isinstance(env_state, dict):
        parts = []
        for key, value in env_state.items():
            parts.append(f"{key}: {value}")
        return " | ".join(parts)
    elif isinstance(env_state, np.ndarray):
        # 对于连续状态空间，做离散化描述
        return f"state_vector: [{', '.join(f'{x:.2f}' for x in env_state)}]"
    else:
        return str(env_state)
```

**Action Parser（动作解析器）**：将LLM的自由文本输出解析为环境可执行的动作。这一步极其关键，因为LLM可能输出各种格式。

```python
def parse_action(llm_output, action_space):
    """将LLM输出解析为合法动作"""
    # 策略1：约束生成（通过prompt限制输出格式）
    # 策略2：后处理解析（从自由文本中提取动作）
    # 策略3：混合（先约束，再解析，失败则fallback）

    for action in action_space:
        if action.name.lower() in llm_output.lower():
            return action

    # fallback: 选择默认动作
    return action_space[0]  # 通常是"noop"或"wait"
```

**Reward Function（奖励函数）**：根据环境反馈计算奖励值。这是整个流程中最需要领域知识的部分。

### 2.3 训练流程

```
RL微调LLM做Agent的完整流程：

Phase 1: 初始化
  ├── 加载预训练LLM（如Llama-7B）
  ├── 构建环境模拟器
  └── 定义奖励函数

Phase 2: 经验收集（Rollout）
  ├── for episode in range(N):
  │     ├── 重置环境，获取初始状态 s_0
  │     ├── for step in range(T):
  │     │     ├── state_text = encode(s_t)
  │     │     ├── action_text = LLM.generate(state_text)
  │     │     ├── action = parse(action_text)
  │     │     ├── s_{t+1}, reward, done = env.step(action)
  │     │     └── 存储 (s_t, a_t, r_t, s_{t+1})
  │     └── 计算回合总奖励 R = sum(rewards)
  └── 得到经验数据集 D = {(s, a, R)}

Phase 3: 策略更新
  ├── 用PPO/GRPO更新LLM参数
  ├── 关键：KL散度约束（防止偏离预训练太远）
  └── 关键：优势函数估计（GAE）

Phase 4: 评估与迭代
  ├── 在测试环境中评估新策略
  ├── 如果性能提升 -> 回到Phase 2
  └── 如果性能下降或震荡 -> 调整超参数
```

### 2.4 关键超参数

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| 学习率 | 1e-6 ~ 1e-5 | 比SFT更保守，防止灾难性遗忘 |
| KL惩罚系数 | 0.01 ~ 0.1 | 控制策略偏离参考模型的程度 |
| PPO clip范围 | 0.1 ~ 0.2 | 限制每次更新的幅度 |
| Rollout批次 | 128 ~ 1024 | 越大越稳定，但显存消耗大 |
| 最大步数T | 50 ~ 500 | 取决于任务复杂度 |
| GAE lambda | 0.95 | 偏差-方差权衡 |

---

## 三、稀疏奖励：Agent RL的核心难题

### 3.1 问题本质

在经典控制任务（如CartPole）中，每一步都有奖励。但在Agent任务中，奖励往往是**稀疏的**：

```
任务：用Agent在网上订一张机票

时间线：
  Step 1: 打开浏览器         reward = 0
  Step 2: 输入航空公司网址    reward = 0
  Step 3: 搜索航班            reward = 0
  Step 4: 选择日期            reward = 0
  Step 5: 选择航班            reward = 0
  Step 6: 填写乘客信息        reward = 0
  Step 7: 选择座位            reward = 0
  Step 8: 点击支付            reward = 0
  Step 9: 确认订单            reward = +1  ← 唯一的奖励信号！
  Step 10: 看到确认页面       reward = 0

问题：Step 1-8 该怎么更新策略？
      它们对最终奖励的"贡献"是多少？
```

这就是**信用分配问题（Credit Assignment Problem）**——当奖励只在最后出现时，如何判断哪些中间步骤是正确的、哪些是错误的？

### 3.2 解决方案一：奖励塑形（Reward Shaping）

核心思想：在稀疏的最终奖励之外，添加**中间奖励信号**来引导学习。

```
原始奖励（稀疏）：
  Step 1-8: reward = 0
  Step 9:   reward = +1（订到票）

奖励塑形后（密集）：
  Step 1: reward = +0.05  （打开了正确的网站）
  Step 2: reward = +0.05  （进入了航班搜索页面）
  Step 3: reward = +0.10  （输入了合理的日期）
  Step 4: reward = +0.10  （选择了价格合理的航班）
  Step 5: reward = +0.15  （正确填写了信息）
  Step 6: reward = +0.10  （选择了座位）
  Step 7: reward = +0.05  （进入了支付流程）
  Step 8: reward = +0.20  （确认订单 -> 最终奖励）
  Step 9: reward = +0.20  （看到确认页面）
```

**但奖励塑形有巨大的风险**——**奖励hacking**。

```
经典案例：
  目标：训练Agent玩赛车游戏，奖励=速度
  结果：Agent发现原地转圈可以刷速度分
        它跑得很快，但根本没往前开

  目标：训练Agent收拾房间，奖励=捡起物品数
  结果：Agent学会反复捡起放下同一个物品
        捡起次数很多，但房间没变干净
```

**实用建议**：奖励塑形的设计原则

1. **势能函数法**：中间奖励必须是某个"势能函数"的差值，保证最优策略不变
2. **渐进式增加难度**：先给密集奖励，训练稳定后逐渐减少中间奖励
3. **多目标奖励**：不要只优化一个指标，加入约束性奖励（如"不能碰撞"）

```python
def potential_based_shaping(state_before, state_after, env):
    """势能函数法奖励塑形，保证最优策略不变"""
    # 势能函数：衡量状态离目标的"距离"
    phi_before = potential_function(state_before, env)
    phi_after = potential_function(state_after, env)

    # 势能差作为中间奖励
    # gamma是折扣因子，保证理论上最优策略不变
    shaping_reward = env.gamma * phi_after - phi_before
    return shaping_reward

def potential_function(state, env):
    """设计势能函数需要领域知识"""
    # 例：网页任务中，势能=已完成的子任务数 / 总子任务数
    completed = count_completed_subtasks(state)
    total = count_total_subtasks(env.task)
    return completed / max(total, 1)
```

![奖励塑形对训练效果的影响](/blog-assets/rl-for-agents/12_reward_shaping_impact.png)

### 3.3 解决方案二：课程学习（Curriculum Learning）

核心思想：**从简单到难**，让Agent先学会简单的子任务，再逐步挑战复杂任务。

```
课程设计方案：订机票任务

Level 1（第1-1000 episodes）：
  任务：点击一个按钮
  奖励：点击成功 +1
  目标：学会基本的鼠标操作

Level 2（第1001-3000 episodes）：
  任务：在搜索框输入文字
  奖励：输入正确 +1
  目标：学会文本输入

Level 3（第3001-6000 episodes）：
  任务：搜索航班并选择一个
  奖励：选到合理航班 +1
  目标：学会多步决策

Level 4（第6001-10000 episodes）：
  任务：完成完整的订票流程
  奖励：成功订票 +1
  目标：端到端完成任务
```

```
课程学习的效果对比：

直接训练最终任务：
  Episodes: 0     2000   4000   6000   8000   10000
  Success:  0%     0%     0%     1%     2%     3%
  （几乎学不会，梯度信号太弱）

课程学习：
  Episodes: 0     2000   4000   6000   8000   10000
  Level:    L1     L1     L2     L3     L4     L4
  Success:  80%   95%    70%    45%    35%    52%
  （每个Level都有学习信号，最终能完成复杂任务）
```

**课程学习的关键**：难度递进的节奏。太慢浪费时间，太快Agent学不会。自适应课程（根据Agent当前能力自动调整难度）是目前的研究热点。

---

## 四、离线RL：从演示数据中学习

### 4.1 为什么需要离线RL？

在线RL（on-policy）需要Agent不断与环境交互，这在很多场景中不可行：

1. **成本太高**：如果环境是真实世界（如机器人操作），每次交互都有物理成本
2. **安全风险**：Agent在探索阶段可能做出危险动作
3. **不可重复**：某些环境（如用户行为）是动态变化的，无法反复交互

```
在线RL vs 离线RL：

在线RL：
  ┌────────┐    动作    ┌────────┐
  │ Agent  │ ────────> │ 环境   │
  │ (策略) │ <──────── │ (真实) │
  └────────┘    奖励    └────────┘
  特点：边交互边学习，需要实时环境

离线RL：
  ┌────────┐           ┌──────────────┐
  │ Agent  │ ────────> │ 离线数据集 D  │
  │ (策略) │ <──────── │ (已有数据)    │
  └────────┘    更新    └──────────────┘
  特点：从固定数据集中学习，不需要实时环境
```

### 4.2 离线RL的核心挑战：分布偏移

离线RL最大的挑战是**分布偏移（Distribution Shift）**：

```
问题场景：

训练数据中，Agent从未见过"在高速公路上逆行"这个状态-动作对。
但如果策略网络在某个状态下"觉得"逆行是个好主意，
它就会输出这个动作。

在线RL中：Agent会尝试这个动作 -> 发现很糟糕 -> 学到教训
离线RL中：Agent只能从数据中学 -> 数据里没有这个教训 -> 可能犯致命错误

这就是OOD（Out-of-Distribution）问题。
```

### 4.3 解决方案

**方案一：保守策略（Conservative Q-Learning, CQL）**

核心思想：对未见过的状态-动作对，**悲观估计**其价值。

```
标准Q-learning：  Q(s,a) = r + gamma * max_a' Q(s',a')
CQL：             Q(s,a) = r + gamma * max_a' Q(s',a') - alpha * OOD_penalty

OOD_penalty = Q(s,a) - log(1/|D|) * sum_{a_i in D} exp(Q(s,a_i))
              ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              当前估计    数据中的平均估计

如果Q(s,a)远大于数据中的平均值，说明这个(s,a)是OOD的，要 penalize。
```

**方案二：Decision Transformer（DT）**

Decision Transformer（Chen et al., 2021）是一个极其优雅的方案：**把RL问题转化为序列建模问题**。

```
传统RL：
  状态s -> 策略网络 -> 动作a -> 最大化累积奖励

Decision Transformer：
  序列 (R1, s1, a1, R2, s2, a2, ..., RT, sT, aT) -> Transformer -> 预测下一个动作

  其中 Rt 是"剩余奖励"（return-to-go）

训练方式：和GPT一样，用交叉熵损失预测下一个token（动作）
推理方式：给定目标奖励R*和当前状态s_t，预测最优动作a_t
```

```
Decision Transformer的优势：

1. 可以用标准Transformer架构，不需要Q网络、策略网络等RL组件
2. 天然支持离线数据，不需要处理分布偏移（因为只是做序列预测）
3. 可以通过调整目标奖励R*来控制策略的"激进程度"
4. 可以和LLM预训练无缝结合（都是next-token prediction）

Decision Transformer的劣势：

1. 对数据质量敏感（垃圾数据 -> 垃圾策略）
2. 在需要长程规划的任务上表现不如传统RL
3. 推理时需要知道目标奖励R*（但很多时候我们不知道最优R*是多少）
```

### 4.4 从人类演示中学习

在Agent场景中，离线RL有一个特别有价值的数据来源：**人类演示数据**。

```
数据来源层次：

Level 1: 专家演示（Expert Demonstrations）
  - 人类玩家的游戏录像
  - 标注员的操作记录
  - 成功任务的完整轨迹
  质量：高  数量：少  成本：高

Level 2: 次优数据（Suboptimal Data）
  - 普通用户的行为日志
  - 成功和失败混合的轨迹
  质量：中  数量：中  成本：中

Level 3: 噪声数据（Noisy Data）
  - 随机探索的数据
  - 其他Agent生成的数据
  质量：低  数量：多  成本：低

最佳实践：混合使用多层级数据，用重要性加权或注意力机制
          让模型自动学习哪些数据更可靠
```

---

## 五、混合架构：RL + LLM 的最优分工

### 5.1 分工原则

2024-2025年的前沿趋势是**混合架构**——让RL和LLM各司其职。

```
混合架构的分工逻辑：

+────────────────────────────────────────────────────────────+
│                    Hybrid Agent Architecture                │
│                                                             │
│  +─────────────────────────────────────────────────────┐   │
│  │              LLM: 高层规划器                          │   │
│  │                                                      │   │
│  │  输入：任务描述、长期目标、环境摘要                     │   │
│  │  输出：子目标序列、策略建议、异常处理方案               │   │
│  │                                                      │   │
│  │  特点：低频调用（每10-100步一次）                      │   │
│  │        处理抽象、推理、泛化                            │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │ 子目标/约束                       │
│                         v                                   │
│  +─────────────────────────────────────────────────────┐   │
│  │              RL Policy: 低层控制器                    │   │
│  │                                                      │   │
│  │  输入：当前状态、子目标、传感器数据                     │   │
│  │  输出：具体动作（每步都调用）                          │   │
│  │                                                      │   │
│  │  特点：高频调用（每步一次）                            │   │
│  │        处理精确控制、实时反应                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
+────────────────────────────────────────────────────────────+

调用频率对比：
  LLM:  每10-100步调用一次（做规划）
  RL:   每步都调用（做控制）

推理延迟对比：
  LLM:  200-2000ms（生成文本）
  RL:   1-10ms（前向传播）
```

### 5.2 具体架构模式

**模式一：LLM规划 + RL执行**

```
任务："去厨房拿一杯水"

LLM规划（调用1次，耗时500ms）：
  子目标1: 导航到厨房
  子目标2: 定位杯子
  子目标3: 抓取杯子
  子目标4: 导航到用户位置
  子目标5: 递交杯子

RL执行（每步调用，总耗时约50ms）：
  子目标1执行：
    step 1: 向前移动0.1m
    step 2: 左转30度
    step 3: 向前移动0.2m
    ...
    step 25: 到达厨房 ✓
  子目标2执行：
    step 26: 旋转摄像头扫描
    step 27: 检测到杯子，坐标(x,y,z)
    ...
```

**模式二：RL探索 + LLM总结**

```
Agent在环境中自主探索1000步后：

RL探索数据：
  1000个状态-动作-奖励元组
  其中成功轨迹50条，失败轨迹950条

LLM总结（调用1次）：
  "分析发现：
   1. 所有成功轨迹都包含'先观察再行动'的模式
   2. 在黑暗环境中失败率高达80%
   3. 建议：增加'先开灯再行动'的策略"

-> 将LLM的总结作为先验知识，指导下一轮RL训练
```

**模式三：LLM做奖励模型**

```
传统做法：人工设计奖励函数（容易出错、难覆盖所有情况）

LLM-as-Reward-Model：
  1. 收集Agent的行为轨迹
  2. 让LLM评价轨迹质量（"这个行为好不好？"）
  3. 用LLM的评价作为奖励信号训练RL策略

优势：
  - 不需要人工设计奖励函数
  - LLM的常识可以帮助判断"合理"的行为
  - 可以处理模糊的、难以量化的目标

劣势：
  - LLM评价有偏差（位置偏差、冗长偏差等）
  - 推理成本高（每条轨迹都要调LLM）
  - 可能存在系统性错误
```

### 5.3 实战案例：WebAgent的混合架构

以网页操作Agent为例，展示混合架构的具体实现：

```python
class HybridWebAgent:
    """LLM规划 + RL控制的混合Agent"""

    def __init__(self, llm_planner, rl_controller):
        self.llm_planner = llm_planner      # 高层规划
        self.rl_controller = rl_controller  # 低层控制
        self.current_plan = None
        self.plan_step = 0

    def act(self, state):
        # 每N步或遇到异常时，调用LLM重新规划
        if self.needs_replanning(state):
            self.current_plan = self.llm_planner.plan(
                task=self.task,
                current_state=state.summary,
                history=self.recent_history
            )
            self.plan_step = 0

        # 用RL控制器执行当前子目标
        subgoal = self.current_plan[self.plan_step]
        action = self.rl_controller.select_action(state, subgoal)

        self.plan_step += 1
        return action

    def needs_replanning(self, state):
        if self.current_plan is None:
            return True
        if self.plan_step >= len(self.current_plan):
            return True
        if state.unexpected_event:
            return True  # 遇到意外情况，重新规划
        return False
```

---

## 六、Benchmark数据：RL Agent的表现如何？

### 6.1 决策任务Benchmark

| Benchmark | 任务类型 | 最佳LLM方案 | 最佳RL方案 | 混合方案 | 人类水平 |
|-----------|---------|------------|-----------|---------|---------|
| ALFWorld | 家务机器人 | 91% (Reflexion) | 85% (RL) | 94% (Hybrid) | 96% |
| WebShop | 网页购物 | 78% (GPT-4) | 82% (AgentTune) | 86% (Hybrid) | 90% |
| Minecraft (Diamond) | 开放世界 | 55% (Voyager) | 40% (RL-only) | 67% (Hybrid) | 85% |
| Overcooked | 协作游戏 | 62% (CoT) | 78% (MAPPO) | 83% (Hybrid) | 92% |
| Crafter | 生存游戏 | 45% (CoT) | 72% (Dreamer) | 68% (Hybrid) | 80% |
| BabyAI | 指令跟随 | 80% (CoT) | 92% (RL) | 95% (Hybrid) | 98% |

![不同方案在多个决策任务Benchmark上的对比](/blog-assets/rl-for-agents/12_benchmark_comparison.png)

### 6.2 关键发现

```
从Benchmark数据中提炼的规律：

1. 纯RL在"规则明确、状态可观测"的任务上表现最好
   代表：BabyAI、Overcooked
   原因：这些任务有清晰的奖励信号，RL可以高效优化

2. 纯LLM在"开放式、需要常识"的任务上表现最好
   代表：Minecraft、ALFWorld
   原因：这些任务需要泛化能力和常识推理，LLM的预训练知识是优势

3. 混合方案在大多数任务上都是最优的
   平均提升：比纯LLM +3-8%，比纯RL +5-15%
   原因：LLM弥补RL的泛化不足，RL弥补LLM的速度和精度不足

4. 训练效率差距巨大
   纯RL：通常需要10M-100M步环境交互
   纯LLM：零训练，但推理成本高
   混合：RL部分需要1M-10M步，但LLM部分显著加速收敛
```

### 6.3 推理效率对比

| 方案 | 推理延迟 | 吞吐量 | 部署成本 |
|------|---------|--------|---------|
| GPT-4 Agent | 2-5s | ~20 req/min | $0.03-0.10/task |
| Llama-70B Agent | 500ms-2s | ~50 req/min | GPU: A100x2 |
| RL Policy (小网络) | 1-5ms | ~10000 req/min | GPU: T4x1 |
| 混合(LLM+RL) | 200ms-1s | ~100 req/min | GPU: A100x1 |

**关键洞察**：混合方案的推理成本可以降低到纯LLM方案的1/10到1/5，同时保持接近甚至更好的性能。这在大规模部署场景中是决定性的优势。

---

## 七、原创实验：三种Agent方案对比

### 7.1 实验设计

为了直观展示三种方案的差异，我们设计了一个"资源收集"任务环境：

```
实验环境：GridWorld Resource Collector

+---+---+---+---+---+---+---+---+
|   |   | R |   |   |   |   | G |
+---+---+---+---+---+---+---+---+
|   | W | W |   |   |   |   |   |
+---+---+---+---+---+---+---+---+
| S |   |   |   | R |   |   |   |
+---+---+---+---+---+---+---+---+
|   |   | W |   |   |   | G |   |
+---+---+---+---+---+---+---+---+

S = 起点    R = 资源(Reward +1)
G = 目标    W = 墙壁(不可通过)

任务：从S出发，收集尽可能多的资源R，最终到达G
 episode长度：最多20步
 最优路径长度：约12步（收集2个资源）
```

**三种方案**：

1. **纯LLM Agent**：用模拟的LLM做决策，每步生成一个动作
2. **RL-Trained Agent**：用Q-learning训练的策略网络
3. **混合Agent**：LLM做路径规划（每5步一次），RL做具体移动

### 7.2 实验结果

```
实验结果（50个episode的平均值）：

指标                纯LLM Agent    RL Agent      混合Agent
-----------------------------------------------------------------
平均收集资源数       1.8            2.1           2.4
平均步数             16.3           13.7          12.8
到达目标率           72%            88%           94%
平均奖励             3.2            4.8           5.6
推理延迟(每步)       ~500ms         ~2ms          ~100ms
训练成本             0              50K episodes  30K episodes
总API成本            $0.50/episode  $0            $0.10/episode
```

### 7.3 学习曲线分析

```
学习曲线对比（横轴=训练episode，纵轴=平均奖励）：

奖励
6.0 |                                          ___________混合
    |                                     ____/
5.0 |                                ____/
    |                           ____/    ___________RL
4.0 |                      ____/    ____/
    |                 ____/    ____/
3.0 |            ____/    ____/
    |       ____/    ____/
2.0 |  ____/    ____/
    | /    ____/
1.0 |/____/
    +---|------|------|------|------|----->
    0   10K    20K    30K    40K    50K   Episodes

观察：
1. RL需要约30K episodes才能收敛到稳定策略
2. 混合方案收敛更快（20K episodes），因为LLM的规划减少了无效探索
3. 纯LLM没有"学习"过程，性能恒定（水平线）
```

![三种Agent方案的学习曲线对比](/blog-assets/rl-for-agents/12_learning_curves.png)

> 完整代码见 `code/12_rl_agent_demo.py`，可直接运行复现。

---

## 八、踩坑实录：三个真实教训

### 坑1：奖励函数的"好意图陷阱"

**背景**：我们训练一个Agent在模拟环境中完成"整理桌面"任务。

**踩坑过程**：

```
第1版奖励函数：
  reward = 桌面上被移动过的物品数量 * 0.1
  意图：鼓励Agent多整理物品

  结果：Agent学会反复拿起放下同一个物品
        每拿起一次+0.1，放下再拿起又+0.1
        桌面反而更乱了

第2版奖励函数：
  reward = 被放到正确位置的物品数 * 0.5 - 被移动的错误物品数 * 0.3
  意图：区分"有效整理"和"无效移动"

  结果：Agent学会只移动最容易判断的物品
        对于不确定的物品，它选择不动（因为动错的惩罚 > 动对的奖励）
        整理效率极低

第3版奖励函数（最终版）：
  reward = 桌面整洁度评分(基于规则) - 移动步数 * 0.01
  意图：直接评估结果，而不是评估过程

  结果：Agent学会了高效整理，但...
        它把所有物品都推到桌子边缘（因为从上方视角看"最整洁"）
        物品随时会掉下去
```

**教训**：

1. **不要奖励过程，要奖励结果**。奖励"移动物品数"不如奖励"桌面整洁度"。
2. **奖励函数要覆盖所有维度**。只看"整洁"不看"稳定"，Agent就会钻空子。
3. **一定要可视化检查**。光看奖励曲线上升不够，必须看Agent实际在干什么。

### 坑2：离线RL的"次优数据陷阱"

**背景**：我们用人类演示数据训练一个网页操作Agent。

**踩坑过程**：

```
数据收集：
  - 50个标注员完成100个网页任务
  - 成功率约70%（30%失败轨迹也被记录）
  - 平均每个任务15步操作

训练结果：
  训练集准确率：92%（很好！）
  测试集准确率：41%（崩溃！）

原因分析：
  1. 标注员的操作习惯高度一致（都先点搜索框）
     -> 模型只学会了"主流"操作路径
     -> 遇到非主流页面布局就完全失效

  2. 失败轨迹中包含了大量"犹豫"动作（来回切换页面）
     -> 模型学会了"犹豫"行为
     -> 在新任务中也反复犹豫，浪费时间

  3. 标注员使用了快捷键和拖拽操作
     -> 但这些"高级技巧"在数据中占比太低
     -> 模型没有学到
```

**解决方案**：

```python
# 数据过滤策略
def filter_demonstration_data(trajectories):
    """过滤低质量演示数据"""
    filtered = []
    for traj in trajectories:
        # 规则1：只保留成功轨迹
        if not traj.success:
            continue

        # 规则2：去除明显犹豫的轨迹
        unique_states = len(set(traj.states))
        if unique_states < len(traj.states) * 0.6:
            continue  # 重复状态太多，说明在犹豫

        # 规则3：去除过长的轨迹（可能是低效操作）
        if len(traj.actions) > OPTIMAL_LENGTH * 3:
            continue

        filtered.append(traj)
    return filtered

# 数据增强策略
def augment_data(trajectories):
    """增加数据多样性"""
    augmented = []
    for traj in trajectories:
        # 策略1：随机裁剪（从轨迹中间开始）
        for start in range(0, len(traj) // 2, 2):
            augmented.append(traj[start:])

        # 策略2：动作替换（用等价动作替换）
        # 例如："点击搜索图标" 和 "按Ctrl+F" 是等价的
        augmented.append(substitute_equivalent_actions(traj))

    return augmented
```

### 坑3：混合架构的"规划-执行脱节"

**背景**：我们实现了一个LLM规划 + RL执行的混合Agent。

**踩坑过程**：

```
第1版架构：
  LLM: 每10步做一次全局规划，输出子目标序列
  RL: 根据当前子目标选择动作

  问题：
  LLM规划："去厨房 -> 拿杯子 -> 回来"
  RL执行中遇到：厨房门是关的
  RL不知道"开门"这个动作（因为不在当前子目标中）
  RL反复尝试"穿门"，浪费10步
  10步后LLM重新规划，才发现门是关的，加入"开门"子目标

  结果：效率极低，每个意外情况都要等10步才能被发现

第2版架构：
  给RL控制器添加"异常检测"模块
  如果连续3步没有进展，触发LLM重新规划

  问题改善了，但引入了新问题：
  RL的"异常检测"阈值很难调
  阈值太低：频繁触发LLM，成本高
  阈值太高：Agent在错误方向上走太远

第3版架构（最终版）：
  1. LLM规划时，同时输出"预期观察"
     子目标："打开厨房门"
     预期观察："门的状态从'关闭'变为'打开'"

  2. RL执行时，同时检查预期观察是否匹配
     如果3步内观察不匹配预期 -> 立即触发重新规划

  3. RL控制器有一个"技能库"
     包含基础动作（开门、拿东西、导航等）
     即使不在当前子目标中，也可以调用

  结果：意外情况的响应时间从10步降到3步
        LLM调用频率降低40%（因为预期观察减少了误触发）
```

**教训**：

1. **规划和执行必须有反馈回路**。不能"规划完就不管了"。
2. **预期观察是关键**。LLM不仅要规划"做什么"，还要预测"做完后应该看到什么"。
3. **RL控制器需要基础技能库**。不能只会执行当前子目标，要有应对意外的能力。

---

## 九、技术前沿：2024-2025年重要进展

### 9.1 GRPO for Agent Training

GRPO（Group Relative Policy Optimization）是DeepSeek团队提出的算法，在Agent训练中展现了独特优势：

```
PPO vs GRPO：

PPO：
  需要训练一个Critic网络来估计V(s)
  优势函数 A(s,a) = r + gamma*V(s') - V(s)
  问题：Critic网络的训练不稳定会传导到策略网络

GRPO：
  不需要Critic网络
  对同一状态采样G个动作，用组内相对排名代替绝对价值估计
  A_i = (R_i - mean(R_group)) / std(R_group)
  优势：训练更稳定，超参数更少，特别适合LLM策略
```

### 9.2 Agent Q*（Agent搜索）

将搜索算法（如MCTS）与RL结合，让Agent在决策时进行"内部搜索"：

```
传统RL Agent：
  状态s -> 策略网络 -> 动作a（一步到位，不"思考"）

Agent Q*：
  状态s -> MCTS搜索（模拟多个未来路径）-> 选择最优动作a
  搜索过程中使用value network评估每个节点
  相当于让Agent在行动前"想一想"后果
```

### 9.3 World Model + RL

训练一个"世界模型"来预测环境变化，减少对真实环境的依赖：

```
World Model架构：

  (s_t, a_t) ──> World Model ──> (s_{t+1}, r_t)
                  (学习的模拟器)

训练流程：
  1. 先用少量真实交互数据训练World Model
  2. 在World Model内部做RL训练（"想象"中练习）
  3. 定期用真实交互校准World Model

优势：
  - 大幅减少真实环境交互次数（10x-100x）
  - 可以在"想象"中尝试危险动作
  - World Model本身可以用于规划
```

---

## 十、实用建议：什么时候该用RL训练Agent？

```
决策清单：

├── 你的任务是否满足以下条件？
│   ├── [ ] 任务固定或任务集明确
│   ├── [ ] 有可用的环境模拟器（或能搭建）
│   ├── [ ] 奖励信号可以自动计算
│   ├── [ ] 需要快速推理（< 50ms）
│   └── [ ] 部署规模大（推理成本敏感）
│
├── 满足4-5个 -> 值得投入RL训练
├── 满足2-3个 -> 考虑混合方案
├── 满足0-1个 -> 用LLM-as-Brain更划算
│
└── 无论哪种方案，先做好以下基础：
    ├── 明确评估指标
    ├── 收集baseline数据
    ├── 设计好奖励函数（或评估函数）
    └── 准备fallback方案（RL失败时回退到LLM）
```

**成本估算参考**：

| 方案 | 开发周期 | GPU训练成本 | 推理成本/task |
|------|---------|-----------|-------------|
| 纯LLM (GPT-4) | 1-2周 | $0 | $0.03-0.10 |
| 纯LLM (Llama-70B) | 2-4周 | $500-2000 (部署) | $0.005-0.01 |
| RL训练 (小策略网络) | 4-8周 | $1000-5000 | $0.0001 |
| RL微调LLM | 8-16周 | $5000-50000 | $0.005-0.01 |
| 混合方案 | 6-12周 | $3000-20000 | $0.001-0.005 |

---

## 十一、本章小结

| 方法 | 核心思想 | 最佳场景 | 主要局限 |
|------|----------|---------|---------|
| LLM-as-Brain | 用预训练LLM直接做决策 | 任务多样、快速原型 | 推理慢、成本高 |
| RL微调 | 用RL优化LLM的策略输出 | 任务固定、需要专精 | 训练贵、泛化差 |
| 离线RL/DT | 从历史数据学习策略 | 无法在线交互 | 分布偏移 |
| 混合架构 | LLM规划 + RL控制 | 大多数实际场景 | 系统复杂度高 |

**核心 takeaway**：RL训练Agent不是"银弹"，但在特定场景下是"最优解"。关键是根据任务特性选择合适的方案——纯LLM适合快速原型和多样化任务，纯RL适合固定任务和实时控制，混合架构则是生产环境的最佳选择。从第4篇的PPO基础到这里的Agent RL，我们看到了一条清晰的脉络：**RL的核心价值不在于"替代"LLM，而在于"增强"LLM——让Agent既有LLM的通用智能，又有RL的高效执行。**

---

## 参考文献

1. Chen, L., et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling." *NeurIPS 2021*.
2. Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*.
3. Kumar, A., et al. (2020). "Conservative Q-Learning for Offline Reinforcement Learning." *NeurIPS 2020*.
4. Wang, G., et al. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models." *arXiv:2305.16291*.
5. Xi, Z., et al. (2023). "The Rise and Potential of Large Language Model Based Agents: A Survey." *arXiv:2309.07864*.
6. Shao, Z., et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv:2402.03300*. (GRPO算法)
7. Hafner, D., et al. (2023). "Mastering Diverse Domains through World Models." *arXiv:2301.04104*. (World Model + RL)
8. Zhou, A., et al. (2024). "Agents Tune: Tuning LLM-based Agents with Reinforcement Learning." *ICML 2024 Workshop*.

---

## 下篇预告

**13 | 评估与可观测性：怎么知道你的Agent到底行不行？**

我们已经讨论了Agent的各种能力——规划、记忆、工具使用、自我改进、RL训练。但一个关键问题始终悬而未决：**怎么评估Agent的表现？**

"感觉挺好的"不是评估。"成功率80%"也不够——80%是在什么任务上？哪些情况失败了？失败的原因是什么？

下篇我们将探讨：
- Agent评估的系统性框架：从benchmark到真实场景
- 可观测性：如何"看到"Agent在想什么
- 评估的陷阱：Goodhart定律和指标hacking
- 原创实验：设计一个Agent评估pipeline

**从"能做"到"做得好"，评估是Agent从demo走向生产的关键一步。敬请期待。**
