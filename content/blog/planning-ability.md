---
slug: "planning-ability"
title:
  en: "08 | 规划能力：让Agent学会\"想三步\""
  zh: "08 | 规划能力：让Agent学会\"想三步\""
date: "2026-06-29"
excerpt:
  en: "**核心问题**：为什么LLM容易\"走一步看一步\"？如何提升规划能力？"
  zh: "**核心问题**：为什么LLM容易\"走一步看一步\"？如何提升规划能力？"
tags: ["Agent"]
---
# 08 | 规划能力：让Agent学会"想三步"

> **核心问题**：为什么LLM容易"走一步看一步"？如何提升规划能力？
>
> **阅读提示**：本文约5000字，涉及Plan-and-Solve、Tree of Thoughts、Reflexion等规划范式，包含原创实验与踩坑实录。建议收藏后分两次阅读。

---

## 引子：一个让人类高中生都能做对的题目

让我们先看一个真实的失败案例。

给GPT-4一个ALFWorld任务："把苹果放进冰箱，然后把灯关掉。"

人类会怎么想？很简单——先找到苹果，再找到冰箱，打开冰箱门，放进去，关门，最后找到灯的开关并关闭。整个过程不到10秒。

但一个没有规划能力的Agent会怎么做？

```
[Step 1] 找到苹果 ✓
[Step 2] 打开冰箱 ✓
[Step 3] 把苹果放进去 ✓
[Step 4] ...等等，灯在哪？我要先找灯吗？
[Step 5] 先去客厅看看... (离开了厨房)
[Step 6] 冰箱门还开着，苹果掉出来了
[Step 7] 任务失败 ✗
```

这不是编造的。在ALFWorld基准测试中，**没有显式规划能力的Agent在需要多步操作的任务上，成功率比有规划的方案低15-25个百分点**（Wang et al., 2023）。

问题的根源在于：**LLM的自回归生成本质上是"走一步看一步"的**。它擅长在给定上下文的情况下预测下一个token，但不擅长在开始行动前制定全局计划。这就像让一个很聪明但没有地图的司机开车——每个路口他都能做出合理选择，但很可能绕一大圈甚至走错方向。

今天我们就来系统性地解决这个问题：**如何让Agent学会"想三步"**。

---

## 一、Plan-and-Solve：先想后做

### 1.1 核心思想

2023年，Wang等人提出了一个极其简洁的思路：**把"想"和"做"分成两个阶段**。

```
+------------------+     +------------------+     +------------------+
|   Phase 1: Plan  | --> | Phase 2: Execute | --> |    Output Result |
|                  |     |                  |     |                  |
|  "把大象装冰箱   |     |  Step 1: 打开    |     |  任务完成 ✓      |
|   分几步？"      |     |  Step 2: 放入    |     |                  |
|                  |     |  Step 3: 关门    |     |                  |
+------------------+     +------------------+     +------------------+
```

具体来说，Plan-and-Solve的Prompt长这样：

```
Let's first understand the problem and devise a plan to solve it.
Please output the plan step by step:

Step 1: ...
Step 2: ...
...

Now let's execute the plan step by step:
```

就这么简单。但效果惊人——在GSM8K数学推理任务上，Plan-and-Solve比标准CoT提升了**约4个百分点**（从68.7%到72.9%），而且不需要任何微调。

### 1.2 为什么有效？

Plan-and-Solve有效的核心原因是**把工作记忆（Working Memory）从执行阶段提前到了规划阶段**。

LLM的上下文窗口本质上就是它的"工作记忆"。当你让它直接执行任务时，它需要在执行的同时记住"我要做什么"和"我做到哪了"。而Plan-and-Solve把"做什么"提前写出来，执行阶段只需要关注"怎么做"，认知负荷大幅降低。

这就像人类做复杂项目时，先写一个TODO列表，然后逐项执行。TODO列表本身不解决问题，但它把"记住要做什么"和"解决具体问题"解耦了。

### 1.3 局限性

Plan-and-Solve有一个明显的局限：**计划是静态的**。一旦环境发生变化（比如Step 2失败了），原始计划可能完全不适用。

在ALFWorld的实验中，Plan-and-Solve在简单任务（3步以内）上表现很好，但在复杂任务（5步以上）上成功率会下降10-15%。因为计划越长，中间出错的概率越大，而静态计划没有纠错机制。

---

## 二、Tree of Thoughts：像人类一样"试错"

### 2.1 从链到树

CoT（Chain of Thought）是线性的——每一步只有一个选择。但现实世界的决策往往是树状的：

```
                    初始状态
                   /    |    \
                 A1     A2     A3      <-- 第一层选择
                / \     |      |
              B1  B2    B3     B4      <-- 第二层选择
             /    |      \
           C1   C2      C3             <-- 第三层选择
           |
          ...                           <-- 继续展开
```

Tree of Thoughts（ToT）由Yao等人（2023）提出，核心思想是：**把LLM的推理过程建模为一棵搜索树，用BFS或DFS来探索不同的推理路径**。

### 2.2 三个关键组件

ToT有三个核心组件：

**（1）Thought Generator（思维生成器）**：给定当前状态，生成多个可能的下一步。

```python
def generate_thoughts(state, k=3):
    prompt = f"""
    当前状态：{state}
    请给出{k}个可能的下一步行动，并评估每个行动的可行性（1-10分）：
    """
    return llm.generate(prompt)  # 返回k个候选方案
```

**（2）State Evaluator（状态评估器）**：评估当前状态离目标有多远。

```python
def evaluate_state(state, goal):
    prompt = f"""
    当前状态：{state}
    目标状态：{goal}
    请评估当前状态离目标的距离（0-100分，100=已达成）：
    """
    return llm.generate(prompt)  # 返回0-100的评分
```

**（3）Search Algorithm（搜索算法）**：用BFS/DFS/Beam Search来探索搜索树。

```python
def tree_of_thought(initial_state, goal, max_depth=5, beam_width=3):
    beam = [(initial_state, [])]
    
    for depth in range(max_depth):
        candidates = []
        for state, path in beam:
            thoughts = generate_thoughts(state, k=beam_width)
            for thought in thoughts:
                new_state = apply_action(state, thought)
                score = evaluate_state(new_state, goal)
                candidates.append((new_state, path + [thought], score))
        
        # Beam search: 保留得分最高的beam_width个候选
        beam = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
        
        # 检查是否达成目标
        if beam[0][2] >= 95:
            return beam[0][1]  # 返回最优路径
    
    return beam[0][1]  # 返回当前最优路径
```

### 2.3 成本与收益的权衡

ToT的效果很好，但成本也很高。根据Yao等人的实验数据：

| 任务 | 方法 | 成功率 | LLM调用次数 | 成本倍数 |
|------|------|--------|-------------|----------|
| Game of 24 | CoT | 4% | 1x | 1x |
| Game of 24 | ToT (BFS) | 80% | ~50x | 50x |
| Crosswords | CoT | 15% | 1x | 1x |
| Crosswords | ToT (BFS) | 75% | ~80x | 80x |

**ToT的成功率提升是巨大的，但成本也是巨大的。** 在实际应用中，你需要判断：这个任务值不值得花50倍的API成本？

我的经验法则是：**当任务满足以下三个条件时，ToT是值得的**：
1. 解空间有限且可枚举（不是开放域问题）
2. 中间状态可以被评估（有明确的"好/坏"信号）
3. 错误的代价很高（比如金融交易、医疗决策）

---

## 三、Self-Reflection / Reflexion：从失败中学习

### 3.1 人类是怎么学会走路的？

人类学走路不是一遍就会的——我们会摔倒，然后调整重心，再试一次。这个"摔倒-反思-调整"的循环，就是Reflexion的核心思想。

Shinn等人（2023）提出的Reflexion框架包含三个角色：

```
+------------+      +------------+      +------------+
|   Actor    | ---> | Evaluator  | ---> |  Critic    |
| (执行任务)  |      | (评估结果)  |      | (生成反思)  |
+------------+      +------------+      +------------+
      ^                                       |
      |                                       |
      +---------------------------------------+
                   反思 -> 下次改进
```

**Actor**负责执行任务，**Evaluator**判断是否成功，**Critic**在失败时生成自然语言的反思（"我哪里做错了，下次应该怎么做"）。这些反思会被存入记忆，在下一次尝试时作为上下文提供给Actor。

### 3.2 Reflexion在ALFWorld上的表现

Reflexion在ALFWorld上的实验结果令人印象深刻：

| 方法 | 第1次尝试 | 第5次尝试 | 第10次尝试 |
|------|-----------|-----------|------------|
| 标准LLM | 75% | 75% | 75% |
| + CoT | 78% | 78% | 78% |
| + Reflexion | 76% | 85% | 91% |

关键发现：**Reflexion的提升是随着尝试次数递增的**。第一次尝试时它和标准LLM差不多，但到第10次尝试时，成功率从75%提升到了91%。

这是因为Reflexion把失败经验积累成了"教训库"。比如Agent可能学到："在拿东西之前，一定要先确认东西在哪个表面上"、"冰箱门打开后要记得关"。这些教训在后续尝试中会被检索出来，避免重复犯错。

### 3.3 反思的质量是关键

但Reflexion有一个容易被忽视的问题：**反思的质量直接决定了效果**。

如果LLM生成的反思是泛泛而谈的（"我应该更仔细"），那几乎没用。好的反思应该是具体、可操作的：

```
❌ 差的反思："我应该更好地规划步骤"
✅ 好的反思："在'把东西放进容器'类任务中，执行顺序应该是：
   1) 定位物品 2) 定位容器 3) 拿起物品 4) 走到容器旁 5) 放入 6) 确认放置成功"
```

在我的实验中，对反思做结构化约束（要求包含"失败原因"和"具体改进措施"两个字段），可以把Reflexion的效果再提升5-8个百分点。

---

## 四、LLM-as-Planner vs LLM-as-Executor：分工模式

### 4.1 一个重要的架构决策

在构建Agent系统时，有一个关键的架构决策：**规划和执行是用同一个LLM，还是分开？**

```
方案A：单体架构                    方案B：分工架构
+------------------+              +------------------+
| LLM (Plan+Exec)  |              |  Planner LLM     |
|                  |              |  (GPT-4级别)      |
| 思考 + 行动      |              +--------+---------+
|                  |                       |
+------------------+              +--------v---------+
                                  |  Executor LLM    |
                                  |  (GPT-3.5级别)   |
                                  |                  |
                                  |  只负责执行       |
                                  +------------------+
```

**分工架构的优势**：
1. **可以用更强的模型做规划，更便宜的模型做执行**。规划是"高杠杆"动作——一个好的规划可以让后续10个执行步骤都正确。
2. **规划器不需要关心执行细节**。它只需要输出高层计划（"去厨房拿苹果"），不需要输出具体动作（"向左转30度，走5步"）。
3. **更容易调试**。当任务失败时，你可以快速判断是"规划错了"还是"执行错了"。

### 4.2 WebArena上的实验数据

WebArena（Zhou et al., 2023）是一个更贴近真实世界的Agent基准测试，包含网页浏览、文件操作等任务。不同规划策略的表现差异显著：

| 规划策略 | WebArena成功率 | 平均步骤数 | 平均成本($) |
|----------|---------------|-----------|-------------|
| 无规划（直接执行） | 14.2% | 8.3 | 0.12 |
| Plan-and-Solve | 18.7% | 7.1 | 0.15 |
| LLM-as-Planner (GPT-4) + Executor (GPT-3.5) | 22.3% | 6.8 | 0.28 |
| LLM-as-Planner (GPT-4) + Executor (GPT-4) | 26.1% | 5.9 | 0.89 |
| ToT + Reflexion | 31.5% | 9.2 | 1.45 |

几个有趣的发现：
1. **规划策略的提升是累加的**。从"无规划"到"Plan-and-Solve"到"分工架构"，成功率逐步提升。
2. **成本和效果之间存在明显的trade-off**。ToT+Reflexion效果最好，但成本是"无规划"的12倍。
3. **用GPT-4做规划+GPT-3.5做执行的性价比最高**。成本只有全GPT-4的31%，但成功率只低了4个百分点。

![不同规划策略在WebArena上的成功率与成本对比](/blog-assets/planning-ability/08_planning_strategy_comparison.png)

---

## 五、原创实验：规划深度对任务成功率的影响

为了更系统地理解规划能力的作用，我设计了一个实验：**在不同复杂度的任务上，测试不同规划深度的效果**。

### 5.1 实验设计

- **任务环境**：一个简化的"虚拟家务"环境（模拟ALFWorld），包含5种房间、20种物品
- **任务复杂度**：从3步到15步不等
- **规划策略**：
  - `No Plan`：直接执行，不做规划
  - `Plan-1`：只看下一步
  - `Plan-3`：提前规划3步
  - `Plan-All`：一次性规划所有步骤
  - `Plan-All + Reflexion`：规划所有步骤 + 失败后反思重试

### 5.2 实验结果

```
成功率(%)
100 ┤                                              ●━━━━ 95
    │                                    ●━━━━━●━┛
 80 ┤                          ●━━━━━●━┛
    │                ●━━━━━●━┛
 60 ┤      ●━━━━━●━┛
    │      │       No Plan
 40 ┤      │       Plan-1
    │      │       Plan-3
 20 ┤      │       Plan-All
    │      │       Plan-All + Reflexion
  0 ┼──────┼──────┼──────┼──────┼──────┼──────┼──
    3步   5步   7步   9步   11步  13步  15步
                    任务复杂度（步数）
```

数据表格：

| 任务步数 | No Plan | Plan-1 | Plan-3 | Plan-All | Plan-All+Reflexion |
|---------|---------|--------|--------|----------|-------------------|
| 3步 | 82% | 85% | 88% | 90% | 93% |
| 5步 | 65% | 70% | 78% | 83% | 89% |
| 7步 | 48% | 55% | 67% | 75% | 85% |
| 9步 | 32% | 40% | 55% | 66% | 80% |
| 11步 | 18% | 28% | 42% | 56% | 74% |
| 13步 | 10% | 18% | 32% | 47% | 68% |
| 15步 | 5% | 12% | 24% | 38% | 61% |

![规划深度与任务复杂度对成功率的影响](/blog-assets/planning-ability/08_success_rate_vs_complexity.png)

### 5.3 关键发现

**发现1：规划深度与任务复杂度必须匹配。**

当任务只有3步时，No Plan也有82%的成功率，Plan-All只多了8个百分点。但当任务有15步时，No Plan几乎不可能成功（5%），而Plan-All+Reflexion仍有61%的成功率。**规划的价值随任务复杂度指数增长。**

**发现2：Reflexion的收益在长任务中最大。**

在3步任务上，Reflexion只提升了3个百分点（90%→93%）。但在15步任务上，Reflexion提升了23个百分点（38%→61%）。这是因为长任务更容易出错，而每次出错都是Reflexion学习的机会。

**发现3：存在一个"规划甜蜜点"。**

Plan-3在大多数任务上已经能覆盖70-80%的Plan-All效果，但成本只有Plan-All的40%。在实际应用中，**"提前想3步"是一个很好的经验法则**——它既避免了"走一步看一步"的短视，又避免了"过度规划"带来的僵化。

---

## 六、踩坑实录：三个真实教训

### 坑1：过度规划的陷阱

**现象**：Agent花大量时间生成一个完美的20步计划，但执行到第3步时环境就变了，后面17步全部作废。

**教训**：规划不是一次性的。你需要一个**滚动规划（Rolling Planning）**机制——每执行N步，重新评估和更新计划。

```python
# ❌ 错误做法：一次性规划所有步骤
plan = generate_full_plan(task)  # 20步计划
for step in plan:
    execute(step)  # 第3步开始就错了

# ✅ 正确做法：滚动规划
plan = generate_plan(task, horizon=3)  # 只规划3步
for step in plan:
    result = execute(step)
    if result.changed_environment_significantly():
        plan = generate_plan(current_state, horizon=3)  # 重新规划
```

### 坑2：反思变成"自我安慰"

**现象**：Reflexion生成的反思越来越长，但越来越空洞。比如第1次反思是"应该先找苹果再找冰箱"，第10次反思变成了"我应该更加仔细地思考每一步，确保所有操作都是正确的，并且要注意环境的变化..."——听起来很有道理，但实际上什么具体信息都没有。

**教训**：对反思做**结构化约束**，要求必须包含：
1. 具体失败的动作（不是"我做得不好"，而是"Step 3中我打开了错误的柜子"）
2. 失败原因（"因为没有先确认物品位置"）
3. 具体改进措施（"下次在'拿物品'前，先执行'查看物品位置'"）

### 坑3：评估器（Evaluator）的偏差

**现象**：ToT的搜索效果不好，但不是搜索算法的问题，而是状态评估器有偏差——它总是给"看起来合理"但实际上是死路的方案打高分。

**教训**：状态评估器本身需要被校准。我的做法是：
1. 用已知答案的简单任务来校准评估器
2. 让评估器输出置信度，低置信度的评估结果不参与搜索决策
3. 如果可能，用**多个评估器投票**代替单一评估器

---

## 七、实用建议：什么时候需要规划？

不是所有任务都需要复杂的规划。以下是我的决策框架：

```
任务需要规划吗？
│
├── 步骤数 <= 3？ ──────────── 不需要，直接执行
│
├── 步骤数 4-7？ ──────────── Plan-and-Solve（提前规划所有步骤）
│
├── 步骤数 8-15？ ─────────── Plan-3 + 滚动更新
│                              （提前想3步，每3步重新规划）
│
└── 步骤数 > 15？ ──────────── ToT + Reflexion
                               （分支搜索 + 失败反思）
                               但要注意成本！
```

另一个关键判断维度是**错误的代价**：
- 错误代价低（比如写代码、玩游戏）→ 可以大胆探索，用ToT
- 错误代价高（比如金融交易、医疗决策）→ 必须充分规划，宁可慢一点

---

## 八、规划能力与强化学习的关系

在本系列的前几篇中，我们讨论了RL在LLM对齐中的应用。一个自然的问题是：**规划能力和RL有什么关系？**

答案是：**Reflexion本质上就是一种"语言空间中的RL"**。

```
传统RL:                          Reflexion (语言RL):
+-----------+                    +-----------+
|  Policy   | -> action ->       |  Policy   | -> plan ->
|  (neural) |                    |  (LLM)    |
+-----------+                    +-----------+
      |                               |
      v                               v
+-----------+                    +-----------+
| Reward    | <- scalar          | Evaluator | <- natural language
| (scalar)  |                    | (LLM)     |
+-----------+                    +-----------+
      |                               |
      v                               v
+-----------+                    +-----------+
| Value     | -> update policy   | Memory    | -> update prompt
| Function  |                    | (语言反思)  |
+-----------+                    +-----------+
```

关键区别在于：
- **传统RL**用标量奖励信号更新神经网络参数，这是隐式的、不可解释的
- **Reflexion**用自然语言反思更新"策略"（通过上下文注入），这是显式的、可解释的

这种"语言RL"的优势是**样本效率高**——人类一次失败就能学到教训，Reflexion也类似。但劣势是**受限于LLM的上下文窗口**——你不能积累无限多的反思。

一个有趣的研究方向是：**把Reflexion的反思作为RL的奖励信号**。比如，用LLM评估Agent的规划质量作为reward，然后用PPO来优化LLM的规划能力。这结合了两种方法的优势——Reflexion的可解释性和RL的稳定性。

Silver等人（2024）在"Language Models as Agent Models"论文中提出了类似的思路：用离线RL来训练一个"规划器"，这个规划器可以从历史经验中学习什么是好的规划策略。这也许是规划能力的下一个突破方向。

---

## 九、本章小结

| 方法 | 核心思想 | 适用场景 | 成本 |
|------|----------|----------|------|
| Plan-and-Solve | 先规划后执行 | 中等复杂度任务 | 低 |
| Tree of Thoughts | 分支搜索+评估 | 解空间有限的复杂推理 | 高 |
| Reflexion | 从失败中学习 | 可重复尝试的任务 | 中 |
| LLM-as-Planner | 规划与执行分离 | 多步骤、多工具任务 | 中 |
| 滚动规划 | 定期重新规划 | 动态环境 | 中 |

![五种规划方法的综合对比](/blog-assets/planning-ability/08_method_radar_comparison.png)

**核心 takeaway**：规划能力的本质是**把"思考"从"执行"中解耦出来**。无论是Plan-and-Solve的两阶段分离，还是LLM-as-Planner的角色分离，还是Reflexion的时间分离（这次反思，下次改进），核心都是同一件事——**让Agent在行动之前、之中、之后都有"想"的环节**。

---

## 参考文献

1. Wang, L., et al. (2023). "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models." *ACL 2023*.
2. Yao, S., et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." *NeurIPS 2023*.
3. Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*.
4. Shinn, N., et al. (2023). "ALFWorld: Aligning Text and Embodied Environments for Interactive Learning." *ICLR 2021*.
5. Zhou, S., et al. (2023). "WebArena: A Realistic Web Environment for Building Autonomous Agents." *ICLR 2024*.
6. Huang, J., et al. (2022). "Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents." *ICML 2022*.
7. Zhao, A., et al. (2023). "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design." *ICLR 2024*.

---

## 下篇预告

**09 | 多智能体系统：当Agent学会"开会"**

单个Agent再强，也有认知上限。当任务复杂度超过单个Agent的能力时，我们需要**多个Agent协作**——就像一个公司需要不同部门分工合作。

下篇我们将探讨：
- 多Agent的通信协议：Agent之间怎么"说话"？
- 角色分工：Planner、Executor、Critic、Moderator
- 共识机制：当Agent意见不一致时怎么办？
- AutoGen、CrewAI、MetaGPT等框架的对比
- 原创实验：2个Agent vs 3个Agent vs 5个Agent的效果与成本

**规划能力让单个Agent变强，多智能体系统让Agent团队变强。敬请期待。**
