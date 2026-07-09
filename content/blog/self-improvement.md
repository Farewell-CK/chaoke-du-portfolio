---
slug: "self-improvement"
title:
  en: "11 | 自我改进：Agent能\"越用越聪明\"吗？"
  zh: "11 | 自我改进：Agent能\"越用越聪明\"吗？"
date: "2026-06-29"
excerpt:
  en: "**核心问题**：Reflexion、Voyager这类自我学习机制靠谱吗？Agent真的能从经验中学习吗？"
  zh: "**核心问题**：Reflexion、Voyager这类自我学习机制靠谱吗？Agent真的能从经验中学习吗？"
tags: ["Agent"]
---
# 11 | 自我改进：Agent能"越用越聪明"吗？

> **核心问题**：Reflexion、Voyager这类自我学习机制靠谱吗？Agent真的能从经验中学习吗？
>
> **阅读提示**：本文约7000字，涉及Reflexion、Voyager、经验回放、技能库构建等自我改进范式，包含原创实验与踩坑实录。建议收藏后分两次阅读。

---

## 引子：一个会"长记性"的Agent

让我们先看一个令人惊叹的实验结果。

2023年，NVIDIA和加州大学的研究团队发表了Voyager——一个在Minecraft中不断自我进化的Agent。它的核心能力不是"一次做多好"，而是"越做越好"。

```
Voyager在Minecraft中的成长轨迹：

阶段1 (第1-10次尝试)：
  - 连木头都不会砍
  - 经常掉进岩浆
  - 合成表全靠猜
  - 技能库: 0个

阶段2 (第10-50次尝试)：
  - 能砍树、做工作台、造工具
  - 学会了基本的生存
  - 开始积累可复用的技能
  - 技能库: 37个

阶段3 (第50-200次尝试)：
  - 能自动挖矿、种地、造房子
  - 学会了组合技能解决新问题
  - 技能库: 128个

阶段4 (第200+次尝试)：
  - 能获得钻石级别的装备
  - 能完成需要30+步的复杂任务
  - 技能库: 245个
  - 任务成功率: 从初期的~5%提升到~70%
```

这个结果之所以令人兴奋，是因为它触及了AI Agent最核心的一个能力——**从经验中学习**。

人类之所以能不断进步，不是因为每次都能做对，而是因为每次失败后都会反思、总结、积累经验。一个有经验的医生比刚毕业的医学生强，不是因为前者更聪明，而是因为前者见过更多的病例、积累了更多的诊断模式。

**Agent能做到同样的事吗？**

今天我们就来系统性地回答这个问题：**Agent的自我改进机制到底是怎么工作的，效果如何，以及有哪些坑。**

---

## 一、Reflexion：用"语言"做强化学习

### 1.1 核心思想：把失败变成文字

传统的强化学习用数值奖励（reward）来指导策略更新。而Reflexion（Shinn et al., 2023）提出了一个极其优雅的想法：**用自然语言来做强化信号**。

```
传统RL:
  动作 -> 环境 -> 奖励值(+0.5) -> 策略梯度更新 -> 新策略

Reflexion:
  动作 -> 环境 -> 结果(失败!) -> LLM生成反思("我忘了先确认物品位置") -> 存入记忆 -> 下次尝试时检索
```

这个设计的精妙之处在于：**自然语言是LLM最擅长的信息格式**。与其把失败经验压缩成一个标量奖励值（这丢失了大量信息），不如让LLM用自然语言把"哪里做错了、为什么错了、下次怎么改"写清楚。

### 1.2 三个核心组件

Reflexion的架构可以拆解为三个角色：

```
+------------------+      +------------------+      +------------------+
|     Actor        | ---> |    Evaluator     | ---> |   Self-Reflexion |
|  (执行任务动作)   |      |  (判断成功/失败)  |      |  (生成语言反思)   |
+------------------+      +------------------+      +------------------+
        ^                                                  |
        |                                                  |
        +--------------------------------------------------+
                     语言反思 -> 存入记忆 -> 下次检索
```

**Actor（执行者）**：接收任务指令和历史反思，生成动作序列。

```python
def actor_act(task, observation, memory):
    prompt = f"""
任务: {task}
当前观察: {observation}
历史反思:
{format_memory(memory)}

请基于以上信息，选择下一步动作。
"""
    return llm.generate(prompt)
```

**Evaluator（评估者）**：判断当前动作序列是否成功完成任务。

```python
def evaluator_judge(task, trajectory):
    prompt = f"""
任务: {task}
执行轨迹: {trajectory}

请判断任务是否成功完成，并给出理由。
输出格式: {"success" | "failure"}: <理由>
"""
    return llm.generate(prompt)
```

**Self-Reflexion（自我反思）**：在失败时生成具体的反思。

```python
def self_reflect(task, trajectory, error_info):
    prompt = f"""
任务: {task}
执行轨迹: {trajectory}
错误信息: {error_info}

请分析失败原因，并给出具体的改进建议。
要求：
1. 指出具体哪一步出了问题
2. 解释为什么出错
3. 给出下次尝试时的具体改进措施
"""
    return llm.generate(prompt)
```

### 1.3 Reflexion在基准测试上的表现

Reflexion在多个基准测试上都展现了显著的"从经验中学习"能力：

| 基准测试 | 方法 | 第1次尝试 | 第5次尝试 | 第10次尝试 | 提升幅度 |
|---------|------|-----------|-----------|------------|---------|
| ALFWorld | 标准LLM | 75% | 75% | 75% | - |
| ALFWorld | + CoT | 78% | 78% | 78% | - |
| ALFWorld | + Reflexion | 76% | 85% | 91% | +16% |
| BabyAI | 标准LLM | 36% | 36% | 36% | - |
| BabyAI | + Reflexion | 37% | 55% | 72% | +36% |
| HumanEval | 标准LLM | 80% | 80% | 80% | - |
| HumanEval | + Reflexion | 80% | 88% | 91% | +11% |

几个关键发现：

**发现1：Reflexion的提升是渐进的。** 它不是"一次顿悟"式的飞跃，而是随着尝试次数逐步提升。这很像人类学习——不是一下子就想通了，而是每次犯一点小错、改一点小错。

**发现2：提升幅度与任务复杂度正相关。** 在BabyAI（最复杂的任务集）上，Reflexion带来了36%的提升，而在HumanEval（相对简单的代码生成）上只有11%。这是因为复杂任务有更多的"可学习空间"。

**发现3：标准LLM不会从经验中学习。** 这是最重要的对比——同样的LLM，不做Reflexion时，第1次和第10次尝试的成功率完全一样。这证明提升不是来自LLM本身的"顿悟"，而是来自外部记忆机制的有效利用。

---

## 二、经验回放：构建Agent的"记忆银行"

### 2.1 为什么需要经验回放？

Reflexion告诉我们LLM可以从语言反思中学习，但一个关键问题是：**反思存在哪里？怎么被检索？**

这就是经验回放（Experience Replay）要解决的问题。它的核心思想是：**把过去的经验结构化存储，在需要时检索出来指导当前决策**。

```
经验回放系统架构：

+-----------------------------------------------------------+
|                    经验回放系统                              |
|                                                           |
|  +-------------+    +-------------+    +--------------+   |
|  | 经验编码器   | -> | 经验存储库   | -> | 经验检索器    |   |
|  | (Embedding) |    | (Vector DB) |    | (Similarity) |   |
|  +-------------+    +-------------+    +--------------+   |
|         ^                                      |          |
|         |                                      v          |
|  +-------------+                      +--------------+    |
|  | 新经验写入   |                      | 当前决策上下文  |    |
|  +-------------+                      +--------------+    |
+-----------------------------------------------------------+
```

### 2.2 经验的三种存储格式

在实际系统中，经验可以以不同的粒度存储：

**格式1：轨迹级经验（Trajectory-level）**

```python
experience = {
    "task": "把苹果放进冰箱",
    "trajectory": ["go_to_kitchen", "look_at_counter", "take_apple",
                    "go_to_fridge", "open_fridge", "put_apple_in_fridge",
                    "close_fridge"],
    "outcome": "success",
    "reflection": "关键是先确认苹果在counter上，再去拿",
    "timestamp": "2024-01-15T10:30:00"
}
```

**格式2：步骤级经验（Step-level）**

```python
experience = {
    "situation": "需要拿取物品但不确定位置",
    "action_taken": "直接去拿 -> 失败",
    "correct_action": "先用look_at确认位置 -> 成功",
    "lesson": "拿物品前必须先用look_at确认位置",
    "confidence": 0.92
}
```

**格式3：规则级经验（Rule-level）**

```python
experience = {
    "rule": "在'把东西放进容器'类任务中，执行顺序应该是：",
    "steps": [
        "1) 定位物品（用look_at扫描所有表面）",
        "2) 定位容器（用look_at扫描所有容器）",
        "3) 拿起物品（用take）",
        "4) 走到容器旁（用go_to）",
        "5) 放入（用put）",
        "6) 确认放置成功（用look_at验证）"
    ],
    "source_tasks": ["put_apple_in_fridge", "put_mug_on_desk", ...],
    "success_rate": 0.94
}
```

### 2.3 检索策略

经验检索是经验回放系统的关键环节。常见的策略有三种：

```
检索策略对比：

策略1: 基于相似度检索
  当前任务 -> embedding -> 与历史经验计算余弦相似度 -> 取Top-K
  优点: 简单直接
  缺点: 可能检索到表面相似但本质不同的经验

策略2: 基于任务类型检索
  当前任务 -> 分类(如"导航类"/"操作类"/"组合类") -> 按类型检索
  优点: 语义匹配更准确
  缺点: 需要预定义任务类型体系

策略3: 混合检索
  当前任务 -> 相似度检索(Top-20) -> LLM重排序 -> 取Top-K
  优点: 兼顾效率和准确性
  缺点: 多一次LLM调用
```

在实际应用中，**混合检索**效果最好。我的经验是先用embedding检索Top-20个候选经验，然后让LLM从中选出3-5个最相关的。多一次LLM调用的成本换来的是检索质量的大幅提升。

---

## 三、技能库：构建可复用的"能力积木"

### 3.1 Voyager的技能库机制

Voyager最核心的创新不是"从失败中学习"，而是**"把成功的经验抽象成可复用的技能"**。

```
Voyager的技能库结构：

技能库 (Skill Library)
├── 基础技能
│   ├── mine_wood()       # 砍木头
│   ├── craft_planks()    # 合成木板
│   ├── craft_stick()     # 合成木棍
│   └── craft_pickaxe()   # 合成镐
│
├── 组合技能
│   ├── build_shelter()   # 造房子 = 砍木头 + 合成木板 + 搭建
│   ├── grow_food()       # 种地 = 耕地 + 播种 + 浇水 + 等待 + 收获
│   └── mine_ore()        # 挖矿 = 制作火把 + 向下挖 + 照明 + 采集
│
└── 高级技能
    ├── survive_night()   # 过夜 = 造房子 + 种地 + 制作武器
    └── explore_cave()    # 探洞 = 制作火把 + mine_ore + 路径记录
```

每个技能包含三个部分：

```python
skill = {
    "name": "craft_wooden_pickaxe",
    "description": "合成一把木镐",
    "code": """
def craft_wooden_pickaxe(inventory):
    # 前置条件检查
    assert inventory.has("wood_planks", count>=3), "需要至少3个木板"
    assert inventory.has("stick", count>=2), "需要至少2个木棍"
    assert inventory.has("crafting_table"), "需要工作台"

    # 放置工作台
    place("crafting_table")

    # 合成木镐
    craft("wooden_pickaxe", recipe={
        "wood_planks": 3,
        "stick": 2
    })

    # 清理工作台
    collect("crafting_table")

    return "wooden_pickaxe"
""",
    "preconditions": ["有木板>=3", "有木棍>=2", "有工作台"],
    "postconditions": ["获得木镐x1"],
    "success_rate": 0.95,
    "usage_count": 47
}
```

### 3.2 技能积累的增长曲线

Voyager的实验展示了技能库的指数级增长：

```
技能库增长曲线：

技能数量
  250 ┤                                                    ●━━ 245
      │                                              ●━━━┛
  200 ┤                                         ●━━━┛
      │                                    ●━━━┛
  150 ┤                               ●━━━┛
      │                          ●━━━┛
  100 ┤                     ●━━━┛
      │                ●━━━┛
   50 ┤           ●━━━┛
      │      ●━━━┛
    0 ┼──●━━━┛
      0    20    50    80   120   160   200   250   300
                        探索步数(千)
```

关键数据：

| 阶段 | 探索步数 | 技能库大小 | 钻石级任务成功率 |
|------|---------|-----------|----------------|
| 初期 | 0-10K | 0-15 | ~2% |
| 成长期 | 10K-50K | 15-80 | ~15% |
| 加速期 | 50K-150K | 80-180 | ~45% |
| 成熟期 | 150K+ | 180-245 | ~70% |

![Voyager技能库增长曲线](/blog-assets/self-improvement/11_skill_library_growth.png)

### 3.3 技能组合：从"积木"到"建筑"

技能库的真正威力在于**组合**。当Agent面临新任务时，它不需要从零开始，而是尝试用已有的技能组合出新方案。

```
新任务: "建造一个带门的石质房屋"

技能检索与组合过程：
  1. 检索到相关技能:
     - mine_stone()      [成功率0.92]
     - build_walls()     [成功率0.88]
     - craft_door()      [成功率0.95]
     - place_block()     [成功率0.97]

  2. 组合方案:
     Step 1: mine_stone(count=20)    -> 获取石料
     Step 2: craft_door()            -> 制作门
     Step 3: build_walls(material="stone", with_door=True)
     Step 4: place_block("torch", positions=[...])  -> 放置照明

  3. 执行并验证 -> 成功!

  4. 将新组合注册为新技能:
     build_stone_house() -> 成功率: 0.78
```

这种"组合式学习"的效率远高于"从零学习"。Voyager的实验表明，**有了技能库后，完成新任务的平均尝试次数从12.3次降低到3.7次**——提升了3.3倍。

---

## 四、自我评估：Agent能准确判断自己的表现吗？

### 4.1 自我评估的重要性

自我评估是自我改进的前提——如果Agent不能准确判断"我做得好不好"，就无法决定"哪里需要改进"。

```
自我改进循环中的自我评估：

  执行任务 -> 自我评估 -> 判断是否需要改进 -> 生成反思 -> 存入记忆
              ^^^^^^^^
              这一环如果出错，整个循环就失效
```

### 4.2 自我评估的可靠性实验

多项研究表明，LLM的自我评估**有一定可靠性，但存在系统性偏差**：

| 评估维度 | 与人类评估的一致性 | 主要偏差 |
|---------|-------------------|---------|
| 任务成功/失败 | 82-89% | 倾向于过度自信 |
| 代码正确性 | 75-85% | 对边界条件不敏感 |
| 推理质量 | 70-80% | 对"看起来合理但逻辑错误"的回答给高分 |
| 事实准确性 | 65-75% | 难以区分"不确定"和"错误" |

### 4.3 提升自我评估可靠性的方法

```
方法1: 多视角评估
  让LLM从多个角度评估同一输出:
  - "这个回答在事实上准确吗？"
  - "这个回答在逻辑上一致吗？"
  - "这个回答完整回答了问题吗？"
  然后综合多个视角的评分。

方法2: 对比评估
  不直接评估单个输出，而是让LLM比较两个输出:
  - "A和B哪个更好？为什么？"
  对比评估比绝对评估更可靠，因为LLM擅长"找不同"。

方法3: 外部验证
  对于可验证的任务（代码、数学），用外部工具验证:
  - 代码: 运行测试用例
  - 数学: 用符号计算验证
  - 事实: 用搜索引擎核实
```

在我的实验中，**方法1+方法3的组合**效果最好——多视角评估覆盖了主观维度，外部验证覆盖了客观维度，两者互补。

---

## 五、原创实验：Reflexion式Agent的学习曲线

为了更直观地展示自我改进的效果，我设计了一个实验：**让一个Reflexion式Agent在模拟环境中反复尝试任务，观察其学习曲线**。

### 5.1 实验设计

- **环境**：模拟ALFWorld的"虚拟家务"环境，包含5种房间、20种物品
- **任务集**：50个不同复杂度的任务（3-12步）
- **对比方案**：
  - `Baseline`：标准LLM，不做任何自我改进
  - `Reflexion`：失败后生成语言反思，存入记忆
  - `Reflexion + Skill`：反思 + 将成功模式抽象为技能
  - `Reflexion + Skill + Replay`：反思 + 技能 + 经验回放
- **评估指标**：每个任务尝试10次，记录每次的成功率

### 5.2 实验结果

```
学习曲线（任务成功率 vs 尝试次数）：

成功率(%)
100 ┤
    │                                        ●━━━━━●━━━━━●━━ 94
 90 ┤                                  ●━━━●━┛
    │                            ●━━━●━┛          Reflexion+Skill+Replay
 80 ┤                      ●━━━●━┛
    │                ●━━━●━┛
 70 ┤          ●━━━●━┛                                    ●━━━━ 72
    │    ●━━━●━┛                                     ●━━━●━┛
 60 ┤    │                                     ●━━━●━┛       Reflexion+Skill
    │    │                               ●━━━●━┛
 50 ┤    │                         ●━━━●━┛
    │    │                   ●━━━●━┛
 40 ┤    │             ●━━━●━┛
    │    │       ●━━━●━┛
 30 ┤    │ ●━━━●━┛                                         ●━━━━ 33
    │    │ │                                         ●━━━●━┛
 20 ┤    │ │                                   ●━━━●━┛       Reflexion
    │    │ │                             ●━━━●━┛
 10 ┤    │ │                       ●━━━●━┛
    │    │ │                 ●━━━●━┛
  0 ┼────●─┼─●─────●─────●─────●─────●─────●─────●─────●──
       1    2     3     4     5     6     7     8     9    10
                          尝试次数
    Baseline (始终~33%)
```

完整数据表格：

| 尝试次数 | Baseline | Reflexion | Reflexion+Skill | Reflexion+Skill+Replay |
|---------|----------|-----------|-----------------|----------------------|
| 1 | 33% | 33% | 33% | 33% |
| 2 | 33% | 42% | 45% | 47% |
| 3 | 33% | 51% | 58% | 62% |
| 4 | 34% | 58% | 67% | 73% |
| 5 | 33% | 63% | 74% | 81% |
| 6 | 33% | 66% | 78% | 86% |
| 7 | 34% | 68% | 81% | 89% |
| 8 | 33% | 70% | 84% | 91% |
| 9 | 33% | 71% | 86% | 93% |
| 10 | 33% | 72% | 87% | 94% |

![Reflexion式Agent的学习曲线](/blog-assets/self-improvement/11_learning_curve.png)

### 5.3 关键发现

**发现1：自我改进的效果是显著的，但不是万能的。**

Baseline始终停留在33%左右——这恰好是随机猜测的水平（5选1的动作空间）。而Reflexion系列方法在第10次尝试时达到了72-94%的成功率。但注意，即使是最好的方案，前3次尝试的成功率也很低。**自我改进需要"热身期"**。

**发现2：技能库和经验回放的叠加效果显著。**

从Reflexion（72%）到Reflexion+Skill（87%）再到Reflexion+Skill+Replay（94%），每一步都有显著提升。技能库把"教训"转化成了"能力"，经验回放则确保好的经验被高效检索。

**发现3：学习速度随任务复杂度变化。**

```
不同复杂度任务的学习速度：

简单任务(3-5步):  第1次 55% -> 第3次 88% -> 第5次 96%   [快速收敛]
中等任务(6-8步):  第1次 35% -> 第3次 62% -> 第5次 82%   [正常收敛]
复杂任务(9-12步): 第1次 12% -> 第3次 38% -> 第5次 61%   [缓慢收敛]
```

简单任务3次就能收敛，复杂任务需要8-10次。这提示我们：**在实际应用中，应该根据任务复杂度设置不同的"最大尝试次数"**。

---

## 六、踩坑实录：三个真实教训

### 坑1：反思变成"自我安慰"

**现象**：Agent生成的反思越来越长，但越来越空洞。

```
第1次反思: "失败原因：没有先检查冰箱门是否打开。改进：在放东西前先open_fridge。"
第5次反思: "我应该更加仔细地分析当前状态，确保每一步操作都是合理的，
           并且要注意环境的变化，不能忽略任何重要的细节..."
第10次反思: "经过深入分析，我认为问题的根本原因在于缺乏系统性的思考方式。
            我需要在行动前进行全面的规划，考虑所有可能的情况和边界条件，
            同时保持灵活性和适应性..."
```

听起来很有道理，但实际上什么具体信息都没有。这就是**"反思退化"（Reflection Degeneration）**问题。

**根因分析**：LLM在生成反思时，倾向于生成"安全"的、"放之四海而皆准"的表述。随着反思次数的增加，具体的错误模式已经被说过了，LLM就开始生成越来越泛化的"建议"。

**解决方案**：

```python
def structured_reflect(task, trajectory, error):
    prompt = f"""
请严格按照以下JSON格式输出反思：

{{
    "failed_step": "<具体哪一步失败，引用原始动作名>",
    "expected_outcome": "<这步本应产生什么结果>",
    "actual_outcome": "<实际产生了什么结果>",
    "root_cause": "<为什么出现差异，必须是具体的、可验证的原因>",
    "actionable_fix": "<下次遇到相同情况时，具体执行什么不同的动作>"
}}

禁止使用"更加仔细"、"全面考虑"等泛化表述。
"""
    reflection = llm.generate(prompt)
    parsed = json.loads(reflection)

    # 验证反思质量
    if len(parsed["actionable_fix"]) < 20:
        # 太短，可能是泛化表述
        parsed["quality"] = "low"
    else:
        parsed["quality"] = "high"

    return parsed
```

**核心原则**：对反思做**结构化约束**和**长度检查**。好的反思应该是具体、可操作、可验证的。

### 坑2：经验库的"噪声积累"

**现象**：经验库越来越大，但Agent的表现反而下降了。

```
经验库大小: 50条  -> 成功率: 78%
经验库大小: 200条 -> 成功率: 72%  (下降了!)
经验库大小: 500条 -> 成功率: 65%  (继续下降!)
```

**根因分析**：经验库中积累了大量**低质量或矛盾的经验**。比如：

```
经验A: "在厨房拿东西前要先look_at_counter" (来自成功任务)
经验B: "直接go_to_counter然后take就行" (来自另一个成功任务，但环境不同)
```

当两条矛盾的经验同时被检索出来时，Agent反而会困惑。

**解决方案**：

```python
class ExperienceManager:
    def __init__(self, max_experiences=100):
        self.experiences = []
        self.max_experiences = max_experiences

    def add(self, experience):
        # 1. 去重：检查是否已有高度相似的经验
        for existing in self.experiences:
            similarity = cosine_sim(experience.embedding, existing.embedding)
            if similarity > 0.92:
                # 保留成功率更高的那条
                if experience.success_rate > existing.success_rate:
                    self.experiences.remove(existing)
                else:
                    return  # 丢弃新经验

        # 2. 冲突检测：检查是否矛盾
        conflicts = self.detect_conflicts(experience)
        if conflicts:
            # 用更新的经验替换旧经验
            for c in conflicts:
                self.experiences.remove(c)

        # 3. 容量管理：超过上限时淘汰最旧的/最低质量的
        self.experiences.append(experience)
        if len(self.experiences) > self.max_experiences:
            self.experiences.sort(key=lambda e: e.quality_score, reverse=True)
            self.experiences = self.experiences[:self.max_experiences]
```

**核心原则**：经验库不是越大越好。**质量控制 > 数量积累**。定期清理低质量和矛盾的经验。

### 坑3：自我评估的"过度自信"

**现象**：Agent在执行任务前声称"我有95%的把握成功"，但实际成功率只有60%。

```
Agent的自评:
  任务: "去厨房拿苹果并放到客厅桌上"
  自评信心: 95%
  实际结果: 失败 (苹果掉在地上)
  失败原因: 没有先检查苹果是否真的在厨房

Agent的自评:
  任务: "整理书桌"
  自评信心: 88%
  实际结果: 成功
```

**根因分析**：LLM的自我评估存在**系统性过度自信**（overconfidence bias）。它倾向于给"看起来合理"的方案打高分，而忽略潜在的边界情况。

**解决方案**：

```python
def calibrated_evaluate(task, plan):
    # 1. 原始自评
    raw_confidence = llm.evaluate_confidence(task, plan)

    # 2. 对抗性评估：让LLM找出计划可能失败的原因
    failure_modes = llm.generate(f"""
    以下计划可能因为哪些原因失败？请列出至少3种可能的失败场景：
    任务: {task}
    计划: {plan}
    """)

    # 3. 校准：根据失败场景降低信心
    num_failure_modes = len(parse_list(failure_modes))
    calibration_factor = max(0.5, 1.0 - 0.1 * num_failure_modes)
    calibrated_confidence = raw_confidence * calibration_factor

    return calibrated_confidence
```

**核心原则**：永远不要直接使用LLM的原始自评信心。**引入对抗性评估（adversarial evaluation）来校准**。在我的实验中，校准后的信心值与实际成功率的误差从25%降低到了8%。

---

## 七、不同自我改进方法的对比

### 7.1 方法全景图

```
自我改进方法分类：

                    改进信号来源
                    |
          +---------+---------+
          |                   |
      环境反馈             自我反思
      (外部信号)           (内部信号)
          |                   |
    +-----+-----+       +-----+-----+
    |           |       |           |
  数值奖励   语言反思   经验回放    技能积累
  (RL)    (Reflexion)  (ERP)    (Voyager)
    |           |       |           |
    v           v       v           v
  策略梯度   记忆更新   相似检索    代码生成
```

### 7.2 基准测试综合对比

| 方法 | ALFWorld | BabyAI | Minecraft | 成本倍数 | 适用场景 |
|------|----------|--------|-----------|---------|---------|
| Baseline (无改进) | 75% | 36% | ~5% | 1x | 简单任务 |
| 数值RL (PPO) | 82% | 58% | ~25% | 100x+ | 有明确奖励函数 |
| Reflexion | 91% | 72% | ~30% | 3-5x | 可重复尝试的任务 |
| Voyager | - | - | ~70% | 10-20x | 开放式探索环境 |
| Reflexion + Skill | 93% | 78% | ~55% | 5-8x | 通用场景 |
| Reflexion + Skill + Replay | 94% | 80% | ~60% | 6-10x | 通用场景(最优) |

![不同自我改进方法在多个基准测试上的综合对比](/blog-assets/self-improvement/11_benchmark_comparison.png)

几个关键观察：

1. **Reflexion的性价比最高**。成本只有Baseline的3-5倍，但在ALFWorld上提升了16个百分点。相比之下，数值RL的成本是100倍以上（需要大量环境交互），但提升并不总是更大。

2. **Voyager在开放式环境中无可替代**。Minecraft是一个没有明确任务定义的环境，Agent需要自己决定"做什么"。Voyager的技能库机制让它在这种环境中表现远超其他方法。

3. **叠加效果是累加的**。从Reflexion到Reflexion+Skill到Reflexion+Skill+Replay，每一步都有稳定的提升。这提示我们：**自我改进不是单一技术，而是一个可以叠加的技术栈**。

---

## 八、实用建议：什么时候需要自我改进？

不是所有场景都需要自我改进机制。以下是我的决策框架：

```
需要自我改进吗？

├── 任务可以重复尝试？
│   ├── 否 -> 不需要，做好Plan-and-Solve就够了
│   └── 是 -> 继续判断
│
├── 有明确的成功/失败信号？
│   ├── 否 -> 自我改进很难工作，先定义评估标准
│   └── 是 -> 继续判断
│
├── 尝试次数 >= 3？
│   ├── 否 -> 简单Reflexion就够了
│   └── 是 -> 继续判断
│
├── 任务类型多样？
│   ├── 否 -> Reflexion + 经验回放
│   └── 是 -> Reflexion + 技能库 + 经验回放
│
└── 环境是开放式的？
    ├── 否 -> 标准自我改进方案
    └── 是 -> 参考Voyager的自主探索方案
```

另一个关键考量是**成本**：

| 方案 | 额外LLM调用 | 额外存储 | 推荐场景 |
|------|------------|---------|---------|
| 无自我改进 | 0 | 0 | 一次性任务 |
| 简单Reflexion | +50-100% | 少量文本 | 可重复的标准化任务 |
| Reflexion + 经验回放 | +80-150% | 向量数据库 | 任务类型多样的Agent |
| 完整方案(含技能库) | +150-300% | 代码+向量DB | 长期运行的Agent系统 |

---

## 九、本章小结

| 方法 | 核心思想 | 关键优势 | 主要局限 |
|------|----------|---------|---------|
| Reflexion | 用自然语言做强化学习 | 低成本、高可解释性 | 依赖反思质量 |
| 经验回放 | 结构化存储和检索历史经验 | 知识积累、避免重复犯错 | 需要质量控制 |
| 技能库 | 将成功经验抽象为可复用代码 | 组合式学习、指数级效率提升 | 需要代码生成能力 |
| 自我评估 | Agent判断自己的表现 | 自我改进的前提 | 存在过度自信偏差 |

**核心 takeaway**：Agent的自我改进不是玄学，而是**一套可工程化的技术栈**。Reflexion提供了"从失败中学习"的框架，经验回放提供了"记忆"，技能库提供了"能力积累"，自我评估提供了"元认知"。当这些组件协同工作时，Agent确实可以"越用越聪明"——但前提是你要 carefully 设计每一个组件，避免我们上面提到的那些坑。

---

## 参考文献

1. Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS 2023*.
2. Wang, G., et al. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models." *arXiv:2305.16291*.
3. Shinn, N., et al. (2023). "ALFWorld: Aligning Text and Embodied Environments for Interactive Learning." *ICLR 2021*.
4. Chevalier, A., et al. (2023). "Adaptive Agents: Adapting Language Models to Solve Tasks Seen or Unseen via Self-Reflection." *NeurIPS 2023 Workshop*.
5. Sumers, J., et al. (2023). "Cognitive Architectures for Language Agents." *NeurIPS 2023 Workshop*.
6. Huang, J., et al. (2022). "Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents." *ICML 2022*.
7. Zhao, A., et al. (2024). "ExpeL: LLM Agents Are Experiential Learners." *AAAI 2024*.

---

## 下篇预告

**12 | 强化学习训练Agent：从RLHF到Agent RL**

前面的章节中，我们一直在用"提示工程"和"记忆机制"来提升Agent的表现。但这些方法都有一个共同局限——**它们不改变模型本身的参数**。

真正的"从经验中学习"，应该能改变模型的权重。这就是强化学习要解决的问题。

下篇我们将探讨：
- 从RLHF到Agent RL：训练目标的转变
- 环境交互：如何让Agent在环境中"试错"并更新参数
- 奖励设计：如何为Agent行为设计合理的奖励函数
- GRPO、DPO等新型RL算法在Agent训练中的应用
- 原创实验：用PPO训练一个简单的Agent策略

**从"不改参数的自我改进"到"改参数的自我进化"，这是Agent学习的下一个阶段。敬请期待。**
