# vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention

> **论文信息**
> - 标题: Efficient Memory Management for Large Language Model Serving with PagedAttention
> - 作者: Woosuk Kwon, Zhuohan Li, et al. (UC Berkeley, Stanford)
> - 会议: SOSP 2023
> - 链接: https://github.com/vllm-project/vllm

---

## 目录

- [一、核心问题与动机](#一核心问题与动机)
- [二、PagedAttention 核心创新](#二pagedattention-核心创新)
- [三、系统架构与关键技术](#三系统架构与关键技术)
- [四、实验结果与性能分析](#四实验结果与性能分析)
- [五、关键实现细节](#五关键实现细节)
- [六、核心贡献与影响](#六核心贡献与影响)
- [七、局限与未来方向](#七局限与未来方向)

---

## 一、核心问题与动机

### 1.1 LLM 推理面临的挑战

这篇论文解决的核心问题是：**大模型推理时 KV Cache 的内存管理效率极低**。

#### 内存分布现状

以 OPT-13B 模型在 NVIDIA A100 (40GB) 上运行为例：

| 内存组成 | 占比 | 特点 |
|---------|------|------|
| 模型参数 | ~65% (26GB) | 静态固定 |
| **KV Cache** | **~30%** | **动态变化** |
| 激活值 | ~5% | 临时使用 |

#### KV Cache 的特点

1. **空间占用巨大**
   - 单个 token 的 KV Cache: 800 KB
   - 计算公式: `2 (key+value) × 5120 (hidden) × 40 (layers) × 2 (FP16)`
   - 单个请求最大可达 1.6 GB (2048 tokens)

2. **动态特性**
   - 长度随生成过程动态增长
   - 生命周期不可预知
   - 与传统深度学习 tensor 有本质区别

3. **性能瓶颈**
   - Autoregressive 生成是 memory-bound 操作
   - GPU 计算能力未被充分利用
   - 内存容量限制了 batch size

### 1.2 现有系统的内存浪费

论文通过精确测量发现，现有系统（FasterTransformer, Orca）的内存利用率极低。

#### 内存浪费的三种形式

```
┌─────────────────────────────────────────────────────────┐
│  Reserved Memory (预留内存)                              │
│  - 为未来的 token 预留空间                               │
│  - 在整个请求生命周期中被占用                             │
│  - 无法被其他请求使用                                    │
├─────────────────────────────────────────────────────────┤
│  Internal Fragmentation (内部碎片)                       │
│  - 预分配了最大长度，但实际使用更短                       │
│  - 只有在请求完成后才知道浪费了多少                       │
│  - 例：预分配 2048，实际只用 100                         │
├─────────────────────────────────────────────────────────┤
│  External Fragmentation (外部碎片)                       │
│  - 不同请求的最大长度不同                                │
│  - Buddy allocator 导致的碎片                           │
│  - 无法被任何请求使用                                    │
├─────────────────────────────────────────────────────────┤
│  Token States (实际有效数据) ✓                          │
│  - 真正存储 KV Cache 的空间                             │
│  - 在现有系统中仅占 20.4% - 38.2%                       │
└─────────────────────────────────────────────────────────┘
```

#### 实测数据（图 2）

| 系统 | 有效内存 | 预留 | 内部碎片 | 外部碎片 |
|-----|---------|------|---------|---------|
| **Orca (Max)** | 20.4% | 13.3% | 26.8% | 41.6% |
| **Orca (Pow2)** | 38.2% | 17.9% | 13.6% | 25.2% |
| **Orca (Oracle)** | 57.3% | 8.9% | - | 36.6% |
| **vLLM** | **96.3%** | - | 3.7% | - |

> 💡 **关键发现**: 现有系统浪费了 60%-80% 的 KV Cache 内存！

### 1.3 复杂解码算法的挑战

现代 LLM 服务需要支持多种解码算法：

1. **Parallel Sampling** (并行采样)
   - 用例：代码补全（GitHub Copilot）
   - 特点：共享 prompt 的 KV Cache
   - 内存共享比例：12% (实验数据)

2. **Beam Search** (束搜索)
   - 用例：机器翻译
   - 特点：动态共享，不同候选之间部分共享
   - 内存共享比例：37%-66% (实验数据)

3. **Shared Prefix** (共享前缀)
   - 用例：Few-shot learning
   - 特点：多个请求共享 system prompt
   - 内存共享比例：取决于 prefix 长度

**现有系统的问题**：
- 无法实现跨序列的内存共享
- 需要频繁复制 KV Cache
- 限制了批处理的灵活性

---

## 二、PagedAttention 核心创新

### 2.1 设计灵感：操作系统的虚拟内存

vLLM 的核心创新是将 **操作系统的虚拟内存和分页技术** 应用到 KV Cache 管理。

#### 概念映射

| 操作系统概念 | PagedAttention 对应 | 说明 |
|------------|-------------------|------|
| **进程 (Process)** | **请求 (Request)** | 独立的执行单元 |
| **虚拟页 (Virtual Page)** | **逻辑 KV Block** | 连续的逻辑地址空间 |
| **物理页 (Physical Page)** | **物理 KV Block** | 实际的 GPU 内存块 |
| **字节 (Byte)** | **Token** | 最小的数据单元 |
| **页表 (Page Table)** | **Block Table** | 逻辑到物理的映射 |
| **虚拟内存 (Virtual Memory)** | **KV Cache Manager** | 统一的内存管理接口 |

#### 关键优势

```
传统方法（连续内存）:
┌────────────────────────────────────────────────┐
│ Token 1 ... Token 2048 (预分配)                │
└────────────────────────────────────────────────┘
问题: 浪费、无法共享

PagedAttention（分块内存）:
┌──────┐    ┌──────┐    ┌──────┐
│Block0│ -> │Block1│ -> │Block2│  (按需分配)
└──────┘    └──────┘    └──────┘
  ↓           ↓           ↓
 物理7        物理1        物理3   (非连续)
优势: 零浪费、灵活共享
```

### 2.2 PagedAttention 算法

#### 传统 Attention 公式

对于输入序列中的第 `i` 个 token：

$a_{i,j} = exp(q_i^T · k_j / √d) / Σ_{t=1}^i exp(q_i^T · k_t / √d)$
$o_i = Σ_{j=1}^i a_ij · v_j$

其中：
- `q_i`: 查询向量
- `k_j, v_j`: 键和值向量
- `d`: 维度
- `a_ij`: attention score
- `o_i`: 输出

**约束**: 所有 `k_j` 和 `v_j` 必须存储在连续内存中

#### PagedAttention 改进

将 KV Cache 分成固定大小的块，每块包含 `B` 个 token：

```
K_j = (k_{(j-1)B+1}, ..., k_{jB})  # 第 j 个 key block
V_j = (v_{(j-1)B+1}, ..., v_{jB})  # 第 j 个 value block
```

块级 Attention 计算：

```
A_ij = exp(q_i^T · K_j / √d) / Σ_{t=1}^⌈i/B⌉ exp(q_i^T · K_t · 1 / √d)

o_i = Σ_{j=1}^⌈i/B⌉ V_j · A_ij^T
```

其中 `A_ij` 是一个行向量，包含对第 `j` 块的所有 attention scores。

#### 算法流程示例

```
Prompt: "Four score and seven years ago our fathers brought forth"

逻辑块分布:
Block 0: [Four, score, and, seven]      -> 物理块 7
Block 1: [years, ago, our, fathers]     -> 物理块 1
Block 2: [brought, forth]               -> 物理块 3

查询 token: "forth"
计算步骤:
1. 读取物理块 7 -> 计算 A_{i,0}
2. 读取物理块 1 -> 计算 A_{i,1}
3. 读取物理块 3 -> 计算 A_{i,2}
4. 合并: o_i = V_0·A_{i,0}^T + V_1·A_{i,1}^T + V_2·A_{i,2}^T
```

### 2.3 Block Table 机制

#### 数据结构

每个请求维护一个 Block Table：

```python
class BlockTable:
    entries: List[BlockEntry]

class BlockEntry:
    logical_block_id: int      # 逻辑块号
    physical_block_id: int     # 物理块号
    num_filled: int            # 已填充的 token 数
    ref_count: int             # 引用计数（用于共享）
```

#### 示例

```
Request A: "Four score and seven years ago our fathers brought"

Block Table:
┌─────────┬──────────┬───────────┬───────────┐
│ Logical │ Physical │ # Filled  │ Ref Count │
├─────────┼──────────┼───────────┼───────────┤
│    0    │    7     │     4     │     1     │
│    1    │    1     │     3     │     1     │
│    2    │    3     │     1     │     1     │
└─────────┴──────────┴───────────┴───────────┘

生成过程:
1. Prefill: 处理 prompt -> 填充 Block 0, 1
2. Decode step 1: 生成 "fathers" -> 填充到 Block 1
3. Decode step 2: 生成 "brought" -> 分配 Block 2
```

---

## 三、系统架构与关键技术

### 3.1 vLLM 系统架构

```
┌────────────────────────────────────────────────────────┐
│              Centralized Scheduler                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │            KV Cache Manager                       │  │
│  │  ┌────────────────┬────────────────────────────┐ │  │
│  │  │ Block Tables   │  Mapping: Logical -> Phys  │ │  │
│  │  ├────────────────┼────────────────────────────┤ │  │
│  │  │ CPU Allocator  │  Swap space in CPU RAM     │ │  │
│  │  ├────────────────┼────────────────────────────┤ │  │
│  │  │ GPU Allocator  │  Physical blocks on GPU    │ │  │
│  │  └────────────────┴────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────┬──────────────┬─────────────┬─────────┘
                  │              │             │
         ┌────────▼──────┐  ┌───▼─────────┐  ┌▼──────────┐
         │   Worker 0    │  │  Worker 1   │  │ Worker N-1│
         │ ┌───────────┐ │  │ ┌─────────┐ │  │ ┌────────┐│
         │ │Model      │ │  │ │Model    │ │  │ │Model   ││
         │ │Shard 0    │ │  │ │Shard 1  │ │  │ │Shard N││
         │ └───────────┘ │  │ └─────────┘ │  │ └────────┘│
         │ ┌───────────┐ │  │ ┌─────────┐ │  │ ┌────────┐│
         │ │Cache      │ │  │ │Cache    │ │  │ │Cache   ││
         │ │Engine     │ │  │ │Engine   │ │  │ │Engine  ││
         │ └───────────┘ │  │ └─────────┘ │  │ └────────┘│
         └───────────────┘  └─────────────┘  └───────────┘
```

#### 组件说明

1. **Centralized Scheduler**
   - 管理所有请求的调度
   - 协调分布式 GPU workers
   - 实现 FCFS 调度策略

2. **KV Cache Manager**
   - 维护所有请求的 Block Tables
   - 管理 CPU 和 GPU 的物理块分配
   - 处理块的分配、释放、共享

3. **GPU Workers**
   - 执行模型推理
   - 通过 Block Table 访问 KV Cache
   - 支持张量并行（Megatron-LM 风格）

### 3.2 核心技术详解

#### 技术 1: 动态内存分配

**按需分配策略**：

```python
# 伪代码
def process_request(prompt, max_tokens):
    # 1. Prefill 阶段：只分配 prompt 需要的块
    prompt_length = len(prompt)
    num_blocks_needed = ceil(prompt_length / BLOCK_SIZE)

    logical_blocks = [0, 1, ..., num_blocks_needed-1]
    physical_blocks = allocator.allocate(num_blocks_needed)

    # 建立映射
    block_table[request_id] = {
        logical_blocks[i]: physical_blocks[i]
        for i in range(num_blocks_needed)
    }

    # 2. Decode 阶段：按需分配新块
    for step in range(max_tokens):
        token = generate_token(request_id)

        last_block = block_table[request_id][-1]
        if last_block.is_full():
            # 分配新的物理块
            new_physical = allocator.allocate(1)
            block_table[request_id].append(new_physical)
```

**优势**：
- 零预留内存浪费
- 内部碎片仅限于最后一个块
- 平均浪费率: `BLOCK_SIZE / 2` per request

#### 技术 2: Copy-on-Write (写时复制)

用于实现高效的内存共享。

**Parallel Sampling 示例**：

```
初始状态: 2 个 sample 共享 prompt
┌─────────────────────────────────────────┐
│         Shared KV Cache (Prompt)        │
│  Block 0 (Ref=2)  │  Block 1 (Ref=2)   │
└──────────┬──────────────────┬───────────┘
           │                  │
      Sample A1           Sample A2

生成阶段: Sample A1 需要写入 Block 1
┌─────────────────────────────────────────┐
│  Block 0 (Ref=2)  │  Block 1 (Ref=1)   │ <- A2 独占
│                   │  Block 3 (Ref=1)   │ <- A1 的副本
└───────────────────┴────────────────────┘
          │                    │
     触发 CoW           A1: Block 0 -> Block 3
     Ref--             A2: Block 0 -> Block 1
```

**实现细节**：

```python
def write_to_block(request_id, block_id, kv_data):
    block_entry = block_table[request_id][block_id]
    physical_block = block_entry.physical_block_id

    if block_entry.ref_count > 1:
        # Copy-on-Write
        new_physical = allocator.allocate(1)
        copy_block_data(physical_block, new_physical)

        # 更新引用
        block_entry.ref_count -= 1
        block_entry.physical_block_id = new_physical
        block_entry.ref_count = 1

    # 写入数据
    write_data(block_entry.physical_block_id, kv_data)
```

#### 技术 3: Beam Search 的动态共享

Beam Search 的共享模式比 Parallel Sampling 更复杂。

**演化过程**：

```
初始 (4 个 beam candidates):
         ┌─ Block 0 (Prompt)
         ├─ Candidate 0: Block 0 -> Block 1 -> Block 2 -> Block 4
         ├─ Candidate 1: Block 0 -> Block 1 -> Block 3 -> Block 6
         ├─ Candidate 2: Block 0 -> Block 1 -> Block 3 -> Block 7
         └─ Candidate 3: Block 0 -> Block 5 -> Block 8 -> Block 9

筛选后 (保留 top-2，都来自 Candidate 1 和 2):
         ┌─ Block 0 (Prompt)
         ├─ New Candidate 0: Block 0 -> Block 1 -> Block 3 -> Block 6 -> Block 10
         └─ New Candidate 1: Block 0 -> Block 1 -> Block 3 -> Block 7 -> Block 11

释放的块: Block 2, 4, 5, 8, 9 (Ref=0)
共享的块: Block 0 (Ref=2), Block 1 (Ref=2), Block 3 (Ref=2)
```

**内存节省**：
- ShareGPT 数据集: 44.3% - 66.3%
- Alpaca 数据集: 37.6% - 55.2%

#### 技术 4: Shared Prefix

预先缓存常用的 system prompt。

**应用场景**：

```
机器翻译服务:

System Prompt (341 tokens):
"Translate English to French:
'sea otter' => 'loutre de mer'
'peppermint' => 'menthe poivrée'
'plush giraffe' => 'girafe en peluche'"

Request A: [System Prompt] + "'cheese' =>"
Request B: [System Prompt] + "'I love you' =>"

┌────────────────────────────────────┐
│   Shared Prefix Blocks (预缓存)     │
│   Block 0, 1, 2, ..., N            │
└──────────┬──────────────┬──────────┘
           │              │
       Request A      Request B
     只需处理         只需处理
     "'cheese'"    "'I love you'"
```

**实现**：

```python
# 服务启动时预计算
def cache_system_prompt(prompt_text):
    tokens = tokenize(prompt_text)
    physical_blocks = allocator.allocate_permanent(
        ceil(len(tokens) / BLOCK_SIZE)
    )

    # 计算 KV Cache
    kv_cache = model.compute_kv_cache(tokens)
    write_to_blocks(physical_blocks, kv_cache)

    return SharedPrefixHandle(physical_blocks)

# 处理请求时直接引用
def process_request_with_prefix(prefix_handle, user_input):
    block_table = prefix_handle.blocks.copy()
    block_table[-1].mark_copy_on_write()  # 最后一块可能被修改

    # 只处理用户输入
    process_tokens(user_input, block_table)
```

**性能提升**：
- 1-shot prefix: 1.67× 吞吐量
- 5-shot prefix: 3.58× 吞吐量

### 3.3 调度与抢占机制

#### 调度策略

**FCFS (First-Come-First-Serve)**：
- 保证公平性
- 防止请求饥饿
- 抢占时优先保留早到的请求

#### 抢占机制

当 GPU 内存不足时的两种策略：

**策略 A: Swapping (换出到 CPU)**

```python
def swap_out_request(request_id):
    blocks = block_table[request_id]

    # 分配 CPU 内存
    cpu_blocks = cpu_allocator.allocate(len(blocks))

    # 传输数据: GPU -> CPU
    for gpu_block, cpu_block in zip(blocks, cpu_blocks):
        copy_device_to_host(gpu_block, cpu_block)

    # 释放 GPU 内存
    gpu_allocator.free(blocks)

    # 记录映射
    swapped_table[request_id] = cpu_blocks

def swap_in_request(request_id):
    cpu_blocks = swapped_table[request_id]

    # 分配 GPU 内存
    gpu_blocks = gpu_allocator.allocate(len(cpu_blocks))

    # 传输数据: CPU -> GPU
    for cpu_block, gpu_block in zip(cpu_blocks, gpu_blocks):
        copy_host_to_device(cpu_block, gpu_block)

    # 恢复映射
    block_table[request_id] = gpu_blocks
```

**优点**：
- 可以恢复完整状态
- 适合大 block size

**缺点**：
- PCIe 带宽限制
- 增加延迟

**策略 B: Recomputation (重新计算)**

```python
def evict_request(request_id):
    blocks = block_table[request_id]

    # 保存 token 序列（轻量级）
    token_sequence = save_tokens(request_id)
    evicted_tokens[request_id] = token_sequence

    # 直接释放内存
    gpu_allocator.free(blocks)
    del block_table[request_id]

def restore_request(request_id):
    tokens = evicted_tokens[request_id]

    # 重新计算 KV Cache（作为一个大的 prefill）
    block_table[request_id] = allocate_and_compute(tokens)
```

**优点**：
- 无数据传输开销
- 适合小 block size
- 可以利用 GPU 计算能力

**缺点**：
- 需要重新计算
- 增加计算开销

**性能对比（图 19）**：

| Block Size | Swapping 开销 | Recomputation 开销 | 推荐策略 |
|-----------|--------------|-------------------|---------|
| 1-8       | 高           | 低                | Recomputation |
| 16-64     | 中等         | 中等              | 两者相当 |
| 128-256   | 低           | 高                | Swapping |

### 3.4 分布式执行

#### Tensor Parallelism

vLLM 采用 Megatron-LM 风格的张量并行：

```
单个 Attention Layer 的分布式计算:

GPU 0:  ┌────────────┐
        │ Heads 0-3  │  <- 处理部分 attention heads
        └────────────┘
             ↓
        All-Reduce
             ↓
GPU 1:  ┌────────────┐
        │ Heads 4-7  │  <- 处理其他 attention heads
        └────────────┘
             ↓
        All-Reduce
             ↓
         合并结果
```

#### 统一的 KV Cache 管理

关键观察：
- 所有 GPU 处理相同的输入 tokens
- 因此需要相同位置的 KV Cache
- 每个 GPU 只存储自己负责的 attention heads 的 KV

```python
# 调度器广播消息
def schedule_step(batch_requests):
    # 准备输入
    input_tokens = [req.next_token for req in batch_requests]

    # 准备 block tables（所有 GPU 共享）
    block_tables = {
        req.id: block_table[req.id]
        for req in batch_requests
    }

    # 广播给所有 workers
    for worker in gpu_workers:
        worker.execute_step(input_tokens, block_tables)

# GPU Worker 执行
def execute_step(input_tokens, block_tables):
    # 根据 block tables 读取 KV Cache
    kv_cache = []
    for req_id in input_tokens.keys():
        blocks = block_tables[req_id]
        # 只读取本 GPU 负责的 heads
        kv_cache.append(
            read_blocks(blocks, head_slice=my_head_range)
        )

    # 执行模型计算（含 All-Reduce）
    outputs = model.forward(input_tokens, kv_cache)

    # 返回结果给调度器
    return outputs
```

**优势**：
- 简化分布式逻辑
- 减少通信开销（只在必要时同步）
- 保持单节点的灵活性

---

## 四、实验结果与性能分析

### 4.1 实验设置

#### 模型配置

| 模型 | 参数量 | GPU 配置 | 总显存 | KV Cache 可用 |
|-----|-------|---------|-------|--------------|
| OPT-13B | 13B | 1× A100 | 40 GB | 12 GB |
| OPT-66B | 66B | 4× A100 | 160 GB | 21 GB |
| OPT-175B | 175B | 8× A100-80GB | 640 GB | 264 GB |

#### 数据集

**ShareGPT** (真实对话数据):
- 来源: ChatGPT 用户分享的对话
- 平均输入长度: 161 tokens
- 平均输出长度: 338 tokens
- 特点: 长序列，高方差

**Alpaca** (指令数据):
- 来源: GPT-3.5 生成的指令数据
- 平均输入长度: 19 tokens
- 平均输出长度: 58 tokens
- 特点: 短序列，低方差

#### Baseline 系统

1. **FasterTransformer**
   - NVIDIA 官方推理引擎
   - 优化目标：低延迟
   - 动态 batching，固定 batch size 上限

2. **Orca (Max)**
   - 预留最大长度 (2048 tokens)
   - 最保守，浪费最多

3. **Orca (Pow2)**
   - 预留到 2 的幂次（最多 2× 实际长度）
   - 例: 实际 25 tokens，预留 32

4. **Orca (Oracle)**
   - 假设知道真实输出长度
   - 理论上界，实际不可行

#### 评估指标

**Normalized Latency** (归一化延迟):
```
normalized_latency = mean(request.latency / request.output_length)
```

单位: 秒/token，越低越好

**Throughput** (吞吐量):
- 系统能维持低延迟的最大请求速率
- 单位: requests/second

### 4.2 单序列生成性能

#### ShareGPT 数据集（长序列）

| 模型 | 请求率 (req/s) | vLLM vs Orca (Oracle) | vLLM vs FasterTransformer |
|-----|---------------|----------------------|--------------------------|
| OPT-13B | 2.0 | **2.2×** | **22×** |
| OPT-66B | 1.0 | **2.7×** | **11×** |
| OPT-175B | 2.5 | **1.7×** | **8×** |

#### Alpaca 数据集（短序列）

| 模型 | 请求率 (req/s) | vLLM vs Orca (Oracle) | vLLM vs FasterTransformer |
|-----|---------------|----------------------|--------------------------|
| OPT-13B | 30 | **2.5×** | **15×** |
| OPT-66B | 20 | **2.1×** | **12×** |
| OPT-175B | 20 | **1.3×** | **6×** |

> 📊 **注**: OPT-175B 在 Alpaca 上提升较小，因为此配置下成为计算瓶颈而非内存瓶颈

#### Batch Size 分析

**OPT-13B @ ShareGPT (2 req/s)**:

```
平均并发请求数:
┌──────────────┬───────────────┐
│ 系统          │ Batch Size    │
├──────────────┼───────────────┤
│ Orca (Max)   │  7.00         │
│ Orca (Pow2)  │  9.81  (↑40%) │
│ Orca (Oracle)│ 13.62  (↑95%) │
│ vLLM         │ 30.42  (↑335%)│
└──────────────┴───────────────┘
```

**OPT-13B @ Alpaca (30 req/s)**:

```
平均并发请求数:
┌──────────────┬───────────────┐
│ 系统          │ Batch Size    │
├──────────────┼───────────────┤
│ Orca (Max)   │   7.00        │
│ Orca (Pow2)  │  43.24 (↑518%)│
│ Orca (Oracle)│  72.75 (↑939%)│
│ vLLM         │ 132.44 (↑1792%)│
└──────────────┴───────────────┘
```

> 💡 **关键洞察**: vLLM 能够批处理 **4-19× 更多请求**，充分利用 GPU 计算能力

### 4.3 复杂解码算法性能

#### Parallel Sampling

测试配置: OPT-13B, Alpaca 数据集

| 并行数 | vLLM vs Orca (Oracle) | 内存节省 |
|-------|----------------------|---------|
| 2     | 1.5× 吞吐量           | 6.09%   |
| 4     | 1.8× 吞吐量           | 8.53%   |
| 6     | 2.1× 吞吐量           | 9.79%   |

#### Beam Search

测试配置: OPT-13B, Alpaca 数据集

| Beam Width | vLLM vs Orca (Oracle) | 内存节省 |
|-----------|----------------------|---------|
| 2         | 1.6× 吞吐量           | 37.56%  |
| 4         | 2.0× 吞吐量           | 53.13%  |
| 6         | **2.3× 吞吐量**       | **55.16%** |

> 📈 **趋势**: Beam width 越大，vLLM 的优势越明显

#### ShareGPT 数据集对比

在更长的序列上，内存共享效果更显著：

| 解码方法 | 内存节省 (Alpaca) | 内存节省 (ShareGPT) |
|---------|-----------------|-------------------|
| Parallel Sampling (n=6) | 9.79% | 30.5% |
| Beam Search (width=6) | 55.16% | **66.3%** |

### 4.4 Shared Prefix 性能

测试配置: LLaMA-13B, WMT16 翻译任务

**1-shot prefix (80 tokens)**:
- vLLM 吞吐量: 42 req/s
- Orca (Oracle) 吞吐量: 25 req/s
- **提升: 1.67×**

**5-shot prefix (341 tokens)**:
- vLLM 吞吐量: 43 req/s
- Orca (Oracle) 吞吐量: 12 req/s
- **提升: 3.58×**

> 💡 **发现**: Prefix 越长，vLLM 的优势越显著

### 4.5 聊天机器人场景

测试配置: OPT-13B, ShareGPT 对话数据

特点:
- 长上下文 (最多 1024 tokens)
- 连续多轮对话
- Prompt 占比大

**结果**:

```
最大可持续请求率:
┌──────────────┬───────────────┐
│ Orca (Max)   │ 0.4 req/s     │
│ Orca (Pow2)  │ 0.4 req/s     │
│ Orca (Oracle)│ 0.4 req/s     │
│ vLLM         │ 0.8 req/s     │
└──────────────┴───────────────┘

提升: 2× 吞吐量
```

原因: 长 prompt 导致 Orca 的 Buddy allocator 浪费严重

---

## 五、关键实现细节

### 5.1 Block Size 选择

#### 消融实验

测试配置: OPT-13B, 固定请求率

**ShareGPT 数据集**:

```
Block Size vs Normalized Latency:
   1 ████████████ 12.1 s/token
   2 ████████     8.5  s/token
   4 ███████      7.2  s/token
   8 ████         4.8  s/token
  16 ██           2.1  s/token  ← 最优
  32 ██           2.3  s/token
  64 ███          3.1  s/token
 128 ████         4.2  s/token
 256 ████████     8.7  s/token
```

**Alpaca 数据集**:

```
Block Size vs Normalized Latency:
   1 ██████       6.2 s/token
   2 ████         4.1 s/token
   4 ███          3.2 s/token
   8 ██           2.5 s/token
  16 █            1.8 s/token  ← 最优
  32 █            1.9 s/token  ← 次优
  64 ████         4.2 s/token
 128 ██████████  10.5 s/token  (序列太短)
 256 ████████████15.2 s/token  (严重浪费)
```

#### 设计权衡

**Block size 太小 (1-8)**:
- ❌ 无法充分利用 GPU 并行性
- ❌ Block table 查找开销大
- ❌ Kernel launch 次数多

**Block size 太大 (128-256)**:
- ❌ 内部碎片严重
- ❌ 共享粒度粗
- ❌ 对短序列特别不友好

**最优选择: 16**:
- ✅ 平衡并行性和碎片率
- ✅ 适合大多数序列长度
- ✅ vLLM 的默认配置

### 5.2 Kernel 优化

#### 三大融合内核

**1. Fused Reshape + Block Write**

传统方法:
```cuda
// 3 次 kernel launch
reshape_kernel<<<...>>>(kv_cache);
transpose_kernel<<<...>>>(kv_cache);
write_kernel<<<...>>>(kv_cache, block_table);
```

vLLM 优化:
```cuda
// 1 次 kernel launch
fused_reshape_write_kernel<<<...>>>(
    kv_cache, block_table, block_size
);
```

**2. Fused Block Read + Attention**

传统方法:
```cuda
// 分离的读取和计算
for (int block_id : block_table) {
    read_block_kernel<<<...>>>(block_id, kv_buffer);
    __syncthreads();
    attention_kernel<<<...>>>(query, kv_buffer, output);
}
```

vLLM 优化:
```cuda
// 融合读取和计算
paged_attention_kernel<<<...>>>(
    query, block_table, block_size, output
) {
    // 在 kernel 内部:
    // 1. 按需读取块
    // 2. 计算 attention
    // 3. 累加结果
}
```

关键技术:
- **每个 warp 处理一个块**，保证 coalesced memory access
- **支持变长序列**，无需 padding
- **On-the-fly 计算**，减少中间结果存储

**3. Fused Block Copy**

用于 Copy-on-Write:

传统方法:
```cuda
// 多次小数据传输
for (int block_id : blocks_to_copy) {
    cudaMemcpyAsync(dst[block_id], src[block_id],
                     block_size, stream);
}
// 问题: launch overhead 高
```

vLLM 优化:
```cuda
// 批量复制 kernel
batched_block_copy_kernel<<<...>>>(
    src_blocks, dst_blocks, num_blocks, block_size
);
```

#### 性能开销分析

**Attention Kernel 微基准测试** (图 18a):

| Batch Size | Context Length | vLLM 延迟 | FasterTransformer 延迟 | 开销 |
|-----------|---------------|----------|----------------------|------|
| 8         | 64            | 45 μs    | 36 μs                | +25% |
| 8         | 128           | 78 μs    | 64 μs                | +22% |
| 8         | 256           | 142 μs   | 112 μs               | +27% |
| 32        | 64            | 52 μs    | 42 μs                | +24% |
| 32        | 128           | 91 μs    | 73 μs                | +25% |
| 32        | 256           | 168 μs   | 135 μs               | +24% |

**开销来源**:
1. Block table 查找 (~5%)
2. 非连续内存访问 (~10%)
3. 分支开销 (~5%)
4. 变长序列处理 (~5%)

**但是**:
- Attention 只占模型总计算的 ~15%
- 端到端开销: < 4%
- 换来 2-4× 的吞吐量提升

### 5.3 Swapping vs Recomputation

#### 微基准测试

测试配置: OPT-13B, 1024 tokens

**Block Size 对开销的影响**:

| Block Size | Swap Out | Swap In | Swap (总) | Recompute |
|-----------|---------|---------|----------|-----------|
| 1         | 85 ms   | 82 ms   | 167 ms   | 45 ms ✓   |
| 16        | 18 ms   | 16 ms   | 34 ms    | 45 ms     |
| 64        | 9 ms    | 8 ms    | 17 ms ✓  | 45 ms     |
| 256       | 7 ms    | 6 ms    | 13 ms ✓  | 45 ms     |

> 💡 **关键洞察**:
> - Recomputation 开销与 block size 无关
> - Swapping 随 block size 增大而减少（PCIe 带宽利用更好）

#### 端到端性能

测试配置: OPT-13B, ShareGPT, 同等请求率

**Normalized Latency**:

| Block Size | Recomputation | Swapping |
|-----------|--------------|----------|
| 1         | 1.2 s/token  | 2.1 s/token |
| 16        | 1.8 s/token  | 1.7 s/token |
| 64        | 2.0 s/token  | 1.5 s/token |
| 256       | 2.2 s/token  | 1.4 s/token |

#### 推荐策略

```python
def choose_eviction_policy(block_size, gpu_compute_power, pcie_bandwidth):
    if block_size <= 16:
        return "recomputation"  # 小块时避免传输开销
    elif block_size >= 128:
        return "swapping"       # 大块时传输效率高
    else:
        # 根据硬件特性动态选择
        compute_cost = block_size * gpu_compute_power
        transfer_cost = block_size * pcie_bandwidth
        return "recomputation" if compute_cost < transfer_cost else "swapping"
```

### 5.4 支持多种解码算法

#### 核心接口设计

vLLM 提供三个基础操作:

```python
class SequenceManager:
    def fork(self, parent_id: int) -> int:
        """创建新序列，共享父序列的 KV Cache"""
        child_id = allocate_sequence_id()

        # 复制 block table（浅拷贝）
        block_table[child_id] = block_table[parent_id].copy()

        # 增加引用计数
        for block in block_table[child_id]:
            block.ref_count += 1

        return child_id

    def append(self, seq_id: int, token: int, kv: Tensor):
        """添加新 token 的 KV Cache"""
        last_block = block_table[seq_id][-1]

        if last_block.is_full():
            # 分配新块
            new_block = allocator.allocate(1)
            block_table[seq_id].append(new_block)
            last_block = new_block

        # 写入（触发 CoW）
        if last_block.ref_count > 1:
            last_block = copy_on_write(last_block)

        write_to_block(last_block, kv)

    def free(self, seq_id: int):
        """删除序列，释放 KV Cache"""
        for block in block_table[seq_id]:
            block.ref_count -= 1
            if block.ref_count == 0:
                allocator.free(block)

        del block_table[seq_id]
```

#### 解码算法实现

**Parallel Sampling**:

```python
def parallel_sampling(prompt, num_samples):
    # 1. Prefill
    base_seq_id = process_prompt(prompt)

    # 2. Fork multiple samples
    sample_ids = [seq_manager.fork(base_seq_id) for _ in range(num_samples)]

    # 3. Independent generation
    while not all_finished(sample_ids):
        for seq_id in sample_ids:
            if not is_finished(seq_id):
                token, kv = generate_token(seq_id)
                seq_manager.append(seq_id, token, kv)

    # 4. Cleanup
    seq_manager.free(base_seq_id)
    return [get_output(sid) for sid in sample_ids]
```

**Beam Search**:

```python
def beam_search(prompt, beam_width, max_len):
    # 1. Prefill
    base_seq_id = process_prompt(prompt)

    # 2. Initialize beams
    beams = [seq_manager.fork(base_seq_id) for _ in range(beam_width)]
    scores = [0.0] * beam_width

    # 3. Beam expansion
    for step in range(max_len):
        candidates = []

        # Expand each beam
        for beam_id, score in zip(beams, scores):
            top_k_tokens, probs = get_top_k(beam_id, vocab_size)

            for token, prob in zip(top_k_tokens, probs):
                child_id = seq_manager.fork(beam_id)
                kv = compute_kv(child_id, token)
                seq_manager.append(child_id, token, kv)

                candidates.append((child_id, score + log(prob)))

        # Select top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        new_beams = [cand[0] for cand in candidates[:beam_width]]
        new_scores = [cand[1] for cand in candidates[:beam_width]]

        # Free old beams
        for old_beam in beams:
            if old_beam not in new_beams:
                seq_manager.free(old_beam)

        beams = new_beams
        scores = new_scores

    # 4. Return best
    best_idx = argmax(scores)
    return get_output(beams[best_idx])
```

---

## 六、核心贡献与影响

### 6.1 理论贡献

#### 1. 系统性分析 LLM 推理的内存瓶颈

论文首次量化了 KV Cache 内存管理的重要性：

**内存浪费分解**:
```
总内存浪费 = 预留内存 + 内部碎片 + 外部碎片
          = 13.3-26.8% + 13.6-17.9% + 25.2-41.6%
          = 60-80%
```

**对吞吐量的影响**:
- 内存利用率每提升 10%
- Batch size 增加 15-25%
- 吞吐量提升 20-30%

#### 2. 跨领域技术迁移的范例

论文证明了 OS 技术可以有效应用于 AI 系统：

| OS 技术 | AI 系统应用 | 效果 |
|--------|-----------|------|
| 虚拟内存 | KV Cache 逻辑视图 | 简化编程模型 |
| 分页 | Block-level 管理 | 消除碎片 |
| Page Table | Block Table | 灵活映射 |
| Copy-on-Write | KV Cache 共享 | 节省 30-66% 内存 |
| Swapping | GPU ↔ CPU | 支持超大 batch |

#### 3. 提出分块 Attention 的理论框架

**数学等价性证明**:

传统 Attention:
```
o_i = Σ_{j=1}^i softmax(q_i^T k_j / √d) · v_j
```

PagedAttention (块级):
```
o_i = Σ_{b=1}^⌈i/B⌉ softmax(q_i^T K_b / √d) · V_b
```

证明: 两者在数值上完全等价（忽略浮点误差）

**通用性**:
- 适用于所有 Transformer 架构
- 支持各种 attention 变体（Multi-head, Multi-query, Grouped-query）
- 可扩展到 Cross-attention

### 6.2 工程贡献

#### 1. 近零内存浪费

**对比分析**:

| 系统 | 有效利用率 | 浪费率 | 改进 |
|-----|----------|--------|------|
| FasterTransformer | ~20% | ~80% | - |
| Orca (Max) | 20.4% | 79.6% | - |
| Orca (Pow2) | 38.2% | 61.8% | 1.9× |
| Orca (Oracle) | 57.3% | 42.7% | 2.8× |
| **vLLM** | **96.3%** | **3.7%** | **4.7×** |

#### 2. 显著的吞吐量提升

**不同场景的提升**:

| 场景 | 模型 | 数据集 | vLLM vs Orca | vLLM vs FasterTransformer |
|-----|------|-------|-------------|--------------------------|
| 单序列 | OPT-13B | ShareGPT | 2.2× | 22× |
| 单序列 | OPT-175B | Alpaca | 1.3× | 6× |
| Parallel (n=6) | OPT-13B | Alpaca | 2.1× | - |
| Beam (w=6) | OPT-13B | Alpaca | 2.3× | - |
| Shared Prefix | LLaMA-13B | WMT16 | 3.6× | - |
| Chatbot | OPT-13B | ShareGPT | 2.0× | - |

#### 3. 统一的编程接口

vLLM 提供了简洁的 API，隐藏了复杂的内存管理：

```python
from vllm import LLM, SamplingParams

# 创建模型
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 各种解码算法使用相同的接口

# 1. 基础采样
outputs = llm.generate(prompts, SamplingParams(
    temperature=0.8,
    top_p=0.95,
))

# 2. Beam search
outputs = llm.generate(prompts, SamplingParams(
    use_beam_search=True,
    best_of=5,
))

# 3. Parallel sampling
outputs = llm.generate(prompts, SamplingParams(
    n=5,  # 生成 5 个候选
    temperature=1.0,
))

# 底层自动处理内存共享和管理
```

### 6.3 实际影响

#### 开源社区

**GitHub 统计** (截至论文发表):
- ⭐ Stars: 20,000+
- 🔱 Forks: 2,000+
- 💬 Issues: 1,500+
- 👥 Contributors: 100+

#### 工业采用

**使用 vLLM 的公司/项目**:
- Anyscale (Ray 生态)
- LMSys (Chatbot Arena)
- HuggingFace (Text Generation Inference)
- OpenAI (内部测试)
- Anthropic (内部测试)

#### 后续工作启发

**直接影响的系统**:
1. **TensorRT-LLM** (NVIDIA)
   - 借鉴 PagedAttention 的思想
   - 实现了类似的分块管理

2. **Text Generation Inference** (HuggingFace)
   - 集成 vLLM 作为后端
   - 支持更多模型架构

3. **LMDeploy** (商汤)
   - 参考 vLLM 的调度策略
   - 优化国产 GPU 适配

**学术影响**:
- SOSP 2023 Best Paper Nominee
- 被 50+ 论文引用
- 成为 LLM 推理系统的标准 baseline

---

## 七、局限与未来方向

### 7.1 当前局限

#### 1. 适用场景限制

**适用场景** ✅:
- 在线服务 (动态请求到达)
- 内存受限环境
- 长序列生成
- 需要高吞吐的应用

**不适用场景** ❌:
- 离线批处理 (已知所有输入)
- 计算瓶颈场景 (如 INT4 量化推理)
- 非常短的序列 (< 32 tokens)
- 非 Transformer 架构

#### 2. Kernel 开销

**Attention Kernel**:
- 比 FasterTransformer 慢 20-26%
- 原因: Block table 查找、非连续访存

**端到端影响**:
- Attention 占总时间 ~15%
- 实际开销 < 4%
- 被吞吐量提升抵消

#### 3. 有限的跨请求共享

**当前实现**:
- ✅ 请求内共享 (Parallel Sampling, Beam Search)
- ✅ 相同 Prefix 共享 (需手动配置)
- ❌ 自动检测相似 Prefix
- ❌ 部分 Prefix 匹配

**潜在改进空间**:
- Automatic Prefix Caching
- Semantic-based Sharing

#### 4. CPU-GPU 通信开销

**Swapping 性能**:
- PCIe 3.0: ~16 GB/s
- PCIe 4.0: ~32 GB/s
- 限制了 swap 速度

**可能改进**:
- NVLink 支持
- GPU Direct RDMA
- 更智能的 swap 策略

### 7.2 未来研究方向

#### 方向 1: 更细粒度的内存管理

**Token-level Paging**:
```python
# 当前: Block size = 16 tokens
Block = [token_1, ..., token_16]  # 固定大小

# 未来: Variable-size blocks
Block = [token_1, ..., token_k]   # k ∈ [1, 32]
```

**优势**:
- 进一步减少内部碎片
- 更灵活的共享粒度

**挑战**:
- Block table 开销增加
- Kernel 实现复杂度

#### 方向 2: 跨请求的智能缓存

**Automatic Prefix Caching**:

```python
class PrefixCache:
    def __init__(self):
        self.cache = {}  # prefix -> KV blocks
        self.lru = LRUCache(capacity=1000)

    def get_or_compute(self, prompt):
        # 查找最长公共前缀
        prefix = self.find_longest_prefix(prompt)

        if prefix in self.cache:
            # 命中缓存
            blocks = self.cache[prefix]
            remaining = prompt[len(prefix):]
            return blocks, remaining
        else:
            # 未命中，计算并缓存
            blocks = compute_kv(prompt)
            self.cache[prompt] = blocks
            return blocks, []
```

**应用场景**:
- 多轮对话（共享历史）
- Few-shot learning（共享示例）
- RAG（共享检索结果）

#### 方向 3: 与其他优化技术结合

**量化 (Quantization)**:

```
FP16 KV Cache:     800 KB / token
INT8 KV Cache:     400 KB / token  (2× 内存节省)
INT4 KV Cache:     200 KB / token  (4× 内存节省)

vLLM + INT4:       8× 内存节省 + 96% 利用率
                   = 批量处理 32× 更多请求
```

**稀疏化 (Sparsity)**:

```python
# H2O: Heavy Hitters Oracle
# 保留重要的 KV Cache，丢弃不重要的

def compress_kv_cache(kv_cache, keep_ratio=0.5):
    # 计算重要性分数
    importance = compute_attention_scores(kv_cache)

    # 保留 top-k
    k = int(len(kv_cache) * keep_ratio)
    important_indices = topk(importance, k)

    return kv_cache[important_indices]
```

**结合效果**:
- vLLM (内存管理) + 量化 (容量) + 稀疏 (冗余)
- 理论上可支持 **100×+ 吞吐量提升**

#### 方向 4: 多模态模型扩展

**挑战**:
- 图像 tokens 数量巨大 (576 for ViT-L)
- 不同模态的 cache 大小不同
- Cross-attention 的内存管理

**可能方案**:

```python
class MultimodalBlockTable:
    text_blocks: List[Block]      # 文本 KV Cache
    vision_blocks: List[Block]    # 视觉 KV Cache
    cross_blocks: List[Block]     # Cross-attention Cache

    def share_vision_blocks(self, requests):
        # 图像可以在多个请求间共享
        if same_image(requests):
            for req in requests:
                req.vision_blocks = shared_blocks
```

#### 方向 5: 推测解码 (Speculative Decoding)

**结合 vLLM**:

```python
def speculative_decoding_with_vllm(prompt):
    # 1. 用小模型快速生成候选
    draft_model = vLLM(model="small-llm")
    candidates = draft_model.generate(prompt, n=5)

    # 2. 用大模型验证
    target_model = vLLM(model="large-llm")
    # 共享 prompt 的 KV Cache
    verified = target_model.verify(candidates, share_prefix=prompt)

    return verified
```

**优势**:
- 减少大模型的 decode 步数
- vLLM 高效管理两个模型的 KV Cache

#### 方向 6: 硬件协同设计

**专用硬件支持**:

1. **Sparse Block Indexing**
   - 硬件加速 Block Table 查找
   - 类似 TLB (Translation Lookaside Buffer)

2. **KV Cache Compression**
   - 硬件支持的无损/有损压缩
   - 减少带宽需求

3. **CXL Memory**
   - 扩展 GPU 可访问内存
   - 更大的 KV Cache 容量

**示例架构**:

```
┌──────────────────────────────────────────┐
│  GPU Compute Units                        │
├──────────────────────────────────────────┤
│  Block Table Cache (Hardware TLB)        │  ← 新增
├──────────────────────────────────────────┤
│  HBM (High Bandwidth Memory)             │
├──────────────────────────────────────────┤
│  CXL Memory Expansion                    │  ← 新增
└──────────────────────────────────────────┘
```

---

## 八、总结

### 8.1 核心创新回顾

vLLM 通过三个层次的创新解决了 LLM 推理的内存瓶颈：

**1. 算法层 (PagedAttention)**:
- 将 Attention 计算分解为块级操作
- 支持非连续内存访问
- 数学等价性保证

**2. 系统层 (Memory Management)**:
- 虚拟内存抽象
- 按需分配，动态增长
- Copy-on-Write 共享机制

**3. 实现层 (GPU Kernels)**:
- 融合内存操作和计算
- 优化内存访问模式
- 批处理减少开销

### 8.2 关键数据总结

| 指标 | 改进幅度 |
|-----|---------|
| 内存利用率 | 20% → 96% (**4.8×**) |
| 吞吐量 (单序列) | **2-4× vs Orca**, **6-22× vs FasterTransformer** |
| 吞吐量 (Beam Search) | **2.3× vs Orca (Oracle)** |
| 内存节省 (Beam Search) | **55-66%** |
| Batch Size | **4-19× more requests** |

### 8.3 设计哲学

vLLM 的成功源于几个核心设计原则：

1. **跨领域迁移** (Cross-domain Transfer)
   - 借鉴成熟的 OS 技术
   - 适应 AI 系统的特点

2. **分离关注点** (Separation of Concerns)
   - 逻辑视图 vs 物理布局
   - 编程接口 vs 内存管理

3. **性能与通用性平衡** (Performance-Generality Tradeoff)
   - 支持多种解码算法
   - 保持高性能

4. **实用主义** (Pragmatism)
   - 接受 20-26% 的 kernel 开销
   - 换取 2-4× 的端到端提升

### 8.4 对 AI Infra 的启示

**1. 内存管理是关键瓶颈**:
- GPU 计算能力 > 内存带宽 > 内存容量
- 优化内存管理比优化计算更重要

**2. OS 技术可以有效迁移**:
- 虚拟内存、分页、CoW、Swapping
- 60 年的系统研究成果可以应用

**3. 系统优化需要全栈思维**:
- 算法 + 系统 + 硬件
- 端到端的性能分析

**4. 开源是推动创新的关键**:
- vLLM 的快速迭代
- 社区的广泛采用和改进

---

## 九、参考资源

### 论文

- **原始论文**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
- **ArXiv**: https://arxiv.org/abs/2309.06180

### 开源代码

- **官方仓库**: https://github.com/vllm-project/vllm
- **文档**: https://docs.vllm.ai/

### 相关工作

**LLM Serving Systems**:
1. Orca (OSDI 2022): Iteration-level scheduling
2. FasterTransformer (NVIDIA): Kernel optimizations
3. FlexGen (ICML 2023): Offloading for limited memory
4. AlpaServe (OSDI 2023): Statistical multiplexing

**Memory Optimizations**:
1. FlashAttention (NeurIPS 2022): IO-aware attention
2. ZeRO-Offload (ATC 2021): Training memory optimization
3. Megatron-LM (SC 2019): Model parallelism

**System Techniques**:
1. Virtual Memory (Kilburn et al., 1962): Original OS paper
2. Copy-on-Write: UNIX fork() optimization

---

## 附录：关键代码片段

### A. PagedAttention 核心算法

```python
def paged_attention(
    query: Tensor,           # [batch, heads, d_head]
    block_tables: List[List[int]],  # [batch][num_blocks]
    context_lens: List[int], # [batch]
    block_size: int,
    num_kv_heads: int,
) -> Tensor:
    """
    PagedAttention 的简化实现

    参数:
        query: 查询向量
        block_tables: 每个序列的 block table
        context_lens: 每个序列的上下文长度
        block_size: 每个块的大小
        num_kv_heads: KV 头的数量

    返回:
        output: attention 输出
    """
    batch_size = len(block_tables)
    num_query_heads = query.shape[1]
    head_dim = query.shape[2]

    output = torch.zeros_like(query)

    for i in range(batch_size):
        # 处理第 i 个序列
        num_blocks = (context_lens[i] + block_size - 1) // block_size

        # 累加 attention 结果
        attn_sum = torch.zeros(num_query_heads, head_dim)
        exp_sum = torch.zeros(num_query_heads)

        for block_idx in range(num_blocks):
            # 获取物理块 ID
            physical_block = block_tables[i][block_idx]

            # 读取块数据
            k_block = load_key_block(physical_block, num_kv_heads, block_size, head_dim)
            v_block = load_value_block(physical_block, num_kv_heads, block_size, head_dim)

            # 计算 attention scores (块级)
            scores = torch.einsum('qhd,khd->qhk', query[i], k_block) / math.sqrt(head_dim)
            scores = torch.exp(scores)  # [num_query_heads, num_kv_heads, block_size]

            # 累加
            exp_sum += scores.sum(dim=-1)  # [num_query_heads, num_kv_heads]
            attn_sum += torch.einsum('qhk,khd->qhd', scores, v_block)

        # 归一化
        output[i] = attn_sum / exp_sum.unsqueeze(-1)

    return output
```

### B. Block Allocator

```python
class BlockAllocator:
    """管理物理 KV 块的分配器"""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))  # 空闲块列表
        self.ref_counts = [0] * num_blocks           # 引用计数

    def allocate(self, num_blocks: int) -> List[int]:
        """分配指定数量的块"""
        if len(self.free_blocks) < num_blocks:
            raise OutOfMemoryError(f"Cannot allocate {num_blocks} blocks")

        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop()
            self.ref_counts[block_id] = 1
            allocated.append(block_id)

        return allocated

    def free(self, block_ids: List[int]):
        """释放块（减少引用计数）"""
        for block_id in block_ids:
            assert self.ref_counts[block_id] > 0
            self.ref_counts[block_id] -= 1

            if self.ref_counts[block_id] == 0:
                self.free_blocks.append(block_id)

    def increase_ref(self, block_ids: List[int]):
        """增加引用计数（用于共享）"""
        for block_id in block_ids:
            self.ref_counts[block_id] += 1

    def get_num_free_blocks(self) -> int:
        """获取空闲块数量"""
        return len(self.free_blocks)
```

### C. Sequence Manager

```python
class SequenceManager:
    """管理序列和它们的 block tables"""

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_tables: Dict[int, List[int]] = {}
        self.next_seq_id = 0

    def create_sequence(self, prompt_len: int) -> int:
        """创建新序列"""
        seq_id = self.next_seq_id
        self.next_seq_id += 1

        # 分配 prompt 需要的块
        num_blocks = (prompt_len + self.allocator.block_size - 1) // self.allocator.block_size
        blocks = self.allocator.allocate(num_blocks)

        self.block_tables[seq_id] = blocks
        return seq_id

    def fork_sequence(self, parent_id: int) -> int:
        """Fork 序列（用于 parallel sampling / beam search）"""
        if parent_id not in self.block_tables:
            raise ValueError(f"Sequence {parent_id} does not exist")

        # 创建子序列
        child_id = self.next_seq_id
        self.next_seq_id += 1

        # 共享父序列的块
        parent_blocks = self.block_tables[parent_id]
        self.block_tables[child_id] = parent_blocks.copy()

        # 增加引用计数
        self.allocator.increase_ref(parent_blocks)

        return child_id

    def append_token(self, seq_id: int) -> Optional[int]:
        """
        为序列添加新 token
        返回新分配的块 ID（如果有）
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} does not exist")

        blocks = self.block_tables[seq_id]
        last_block = blocks[-1]

        # 检查最后一个块是否已满
        # （这里简化处理，实际需要维护每个块的填充状态）
        if self._is_block_full(last_block):
            # Copy-on-Write（如果需要）
            if self.allocator.ref_counts[last_block] > 1:
                new_block = self.allocator.allocate(1)[0]
                self._copy_block(last_block, new_block)

                self.allocator.free([last_block])
                blocks[-1] = new_block

            # 分配新块
            new_block = self.allocator.allocate(1)[0]
            blocks.append(new_block)
            return new_block

        return None

    def delete_sequence(self, seq_id: int):
        """删除序列"""
        if seq_id not in self.block_tables:
            return

        blocks = self.block_tables[seq_id]
        self.allocator.free(blocks)
        del self.block_tables[seq_id]
```

---

**一句话总结**: vLLM 通过将操作系统的虚拟内存和分页技术应用到 LLM 的 KV Cache 管理，实现了近零内存浪费（96.3% 利用率）和 2-4 倍的吞吐量提升，成为大模型推理系统的里程碑式工作。

---

**作者**: Claude (AI Assistant)
**日期**: 2025-10-14
**版本**: v1.0
