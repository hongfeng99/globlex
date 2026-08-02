# Globex Agent

Globex Agent 是一个面向 Amazon、Shopee、AliExpress 和 eBay 的跨境购物 Agent 项目。后端基于 Python、LangChain 与 FastAPI，前端基于 React + Vite，并包含商品召回、RAG 品类洞察、跨平台比价、关税运费估算、长期记忆、上下文压缩、Harness 安全控制、评测进化和生产部署配置。

> 下方结构列出所有项目自有文件，并逐项说明用途。`.git`、`.venv`、`node_modules`、前端构建产物和运行时生成内容属于工具或运行环境自动生成的数据，因此只标注目录职责，不逐一展开其内部文件。

## 项目结构

```text
globex-agent/
├── README.md                              # 项目说明、目录结构与文件职责文档
├── .env                                   # 本地真实环境变量；包含密钥，不应提交仓库
├── .env.example                           # 项目环境变量模板和本地配置示例
├── .gitignore                             # Git 忽略规则，排除密钥、缓存、构建与运行时文件
├── pyproject.toml                         # Python 项目信息、依赖、打包和 pytest 配置
├── uv.lock                                # uv 锁定的 Python 完整依赖版本，保证环境可复现
│
├── .git/                                  # Git 仓库元数据，由 Git 自动维护
├── .venv/                                 # uv 创建的 Python 虚拟环境，不属于项目源码
├── .pytest_cache/                         # pytest 自动生成的测试缓存
│
├── app/                                   # 后端业务代码主包
│   ├── __init__.py                        # 声明 app 为 Python 包
│   │
│   ├── agent/                             # AgentLoop、模型、提示词、fork 与工具注册
│   │   ├── __init__.py                    # 声明 Agent 子包
│   │   ├── llm.py                         # 创建主模型、轻量模型、最小模型和 Judge 模型
│   │   ├── prompts.py                     # 加载并校验 prompts.yml，提供各类提示词读取函数
│   │   ├── main_agent.py                  # 组装并运行主 AgentLoop，注入记忆、预算和 Harness
│   │   ├── dispatch_tool.py               # fork 同质子 AgentLoop 的元工具及子任务超时控制
│   │   ├── fork_guard.py                  # 通过 ContextVar 限制递归 fork 深度
│   │   ├── dynamic_fork.py                # 根据各平台历史成功率动态选择 fork 平台
│   │   ├── middleware.py                  # 工具结果截断和重复工具调用检测
│   │   ├── tool_registry.py               # 注册 9 个核心工具、dispatch_tool 和终结性工具
│   │   └── registry.py                    # 兼容早期章节导入路径的工具注册表转发模块
│   │
│   ├── api/                               # FastAPI、WebSocket、请求上下文和事件监控
│   │   ├── __init__.py                    # 声明 API 子包
│   │   ├── context.py                     # 保存 thread_id 与 session_dir 的 ContextVar
│   │   ├── connection.py                  # 管理 thread_id 到 WebSocket 的连接路由
│   │   ├── monitor.py                     # 推送 tool_start、tool_end、fork 和 task_result 事件
│   │   └── server.py                      # FastAPI 入口；提供任务、取消、上传、下载和 WS 接口
│   │
│   ├── tools/                             # Agent 可以调用的核心业务工具
│   │   ├── __init__.py                    # 声明工具子包
│   │   ├── planner.py                     # 将复杂购物需求拆成严格的结构化意图
│   │   ├── chat_fallback.py               # 处理非购物问题和普通闲聊
│   │   ├── web_search.py                  # 通过 Tavily 检索评测、榜单和外部资料
│   │   ├── category_insight.py            # 基于 OpenSearch、Hybrid 与 Rerank 的品类洞察
│   │   ├── item_search.py                 # 单平台商品召回及语义/个性化双通道合流
│   │   ├── item_picker.py                 # 按硬约束、价格带、时效和偏好精挑候选商品
│   │   ├── price_compare.py               # 汇率归一后进行跨平台商品比价
│   │   ├── shipping_calc.py               # 估算运费、关税、到手价和配送时间
│   │   └── shopping_summary.py            # 终结性工具；生成最多 3 件商品的最终购物清单
│   │
│   ├── recall/                            # 向量召回、品类知识库和价格辅助服务
│   │   ├── __init__.py                    # 声明召回子包
│   │   ├── towers.py                      # User、Query、Item 三塔向量服务统一异步客户端
│   │   ├── ann.py                         # Faiss ANN 索引和商品元数据访问
│   │   ├── category_kb.py                 # CategoryCard 品类知识卡片数据模型
│   │   ├── category_norm.py               # 品类名称别名归一化
│   │   ├── reranker.py                    # 远程 Cross-Encoder 重排服务客户端
│   │   ├── reranker_service.py            # 基于 BGE Reranker 的独立 FastAPI 推理服务
│   │   ├── fx.py                          # 多币种兑换和人民币价格归一
│   │   ├── duty.py                        # 跨境商品关税档位与金额估算
│   │   └── shipping.py                    # 各平台运费和配送时效估算
│   │
│   ├── memory/                            # 用户长期偏好、经验策略和失败教训
│   │   ├── __init__.py                    # 声明记忆子包
│   │   ├── store.py                       # PreferenceEntry 与 StrategyEntry 的读写和相关召回
│   │   ├── injector.py                    # 将长期偏好和历史成功策略注入 system prompt
│   │   ├── strategy.py                    # 可复用成功策略的数据模型
│   │   └── lesson.py                      # 失败教训及规避提示的数据模型
│   │
│   ├── compress/                          # Cache Breakpoint 上下文压缩
│   │   ├── __init__.py                    # 声明压缩子包
│   │   ├── breakpoint.py                  # 根据上下文长度计算需要压缩的消息边界
│   │   └── compressor.py                  # 用 LLM 将边界前消息压缩为事实摘要
│   │
│   ├── budget/                            # Token 消耗统计、模型路由和预算降级
│   │   ├── __init__.py                    # 导出 TokenBudget 和预算上下文接口
│   │   ├── token_budget.py                # ContextVar Token 预算及 main/lite/minimal/fallback 四级路由
│   │   ├── limits.py                      # free、standard、premium 用户预算上限
│   │   ├── accounting.py                  # 记录 LLM 与工具返回造成的 Token 消耗
│   │   ├── middleware.py                  # 预算提示注入、模型选择和预算感知压缩
│   │   └── fallback.py                    # Token 耗尽后不调用 LLM 的确定性兜底回答
│   │
│   ├── resilience/                        # 外部工具熔断、请求排队和幂等保护
│   │   ├── __init__.py                    # 声明韧性控制子包
│   │   ├── circuit_breaker.py             # Closed/Open/HalfOpen 三态异步熔断器
│   │   ├── breakers.py                    # 为平台检索、Reranker 和 Tower 创建独立熔断器
│   │   ├── request_queue.py               # 普通/重型请求的优先队列、并发量与动态再平衡
│   │   └── dedup.py                       # 在时间窗口内识别并拒绝重复用户请求
│   │
│   ├── observability/                     # LangFuse Trace 与运行时告警
│   │   ├── __init__.py                    # 导出当前 Trace 上下文接口
│   │   ├── trace_ctx.py                   # 用 ContextVar 保存当前 LangFuse Trace 和 Span
│   │   ├── langfuse_handler.py            # 为 LangGraph 调用创建 LangFuse CallbackHandler
│   │   └── alerts.py                      # 工具 P99 延迟窗口、告警规则和 Webhook 通知
│   │
│   ├── security/                          # Prompt Injection、泄露和日志脱敏防护
│   │   ├── __init__.py                    # 声明安全子包
│   │   ├── tool_whitelist.py              # 从注册表建立工具白名单并校验工具名
│   │   ├── content_filter.py              # 清洗工具返回中的间接 Prompt Injection
│   │   ├── output_guard.py                # 审核并脱敏商品 ID、线程 ID、密钥和内部地址
│   │   └── log_sanitizer.py               # 对 query、user_id、密钥和偏好做日志脱敏
│   │
│   ├── harness/                           # Agent 行为控制、验证与 Hook Pipeline
│   │   ├── __init__.py                    # 导出 HarnessMiddleware 和 Hook 注册接口
│   │   ├── middleware.py                  # 六阶段 Hook 的注册、优先级、执行及异常隔离
│   │   ├── setup.py                       # 导入并注册所有内置 Hook
│   │   ├── phase_machine.py               # PLANNING 到 CONCLUDING 的对话阶段状态机
│   │   ├── tool_filter.py                 # 根据当前阶段生成动态工具集
│   │   ├── tool_risk.py                   # 标记只读、写入和资源密集型工具
│   │   ├── user_tool_filter.py            # 按用户等级叠加工具使用限制
│   │   └── hooks/                         # 内置 Harness Hook 实现
│   │       ├── __init__.py                # 声明 Hook 子包
│   │       ├── tool_whitelist.py          # pre_tool_call：拒绝不在白名单中的工具
│   │       ├── phase_check.py             # pre_tool_call：检查当前阶段是否允许调用工具
│   │       ├── user_tier_check.py         # pre_tool_call：检查用户等级权限
│   │       ├── sequencing.py              # pre_tool_call：验证工具调用的前置依赖顺序
│   │       ├── content_filter.py          # post_tool_call：过滤工具返回中的注入内容
│   │       ├── truncate.py                # post_tool_call：截断过长工具返回
│   │       ├── step_validator.py          # post_tool_call：用 Pydantic 校验工具返回 Schema
│   │       ├── loop_detector.py           # post_reflect：检测重复工具调用并注入收敛提示
│   │       ├── drift_detector.py          # post_reflect：检测 Silent Drift 并注入方向校正
│   │       ├── assertion_handler.py       # post_reflect：汇总单步校验失败并反馈给 Agent
│   │       └── phase_transition.py        # post_reflect：根据进展切换对话阶段
│   │
│   ├── evolution/                         # Bad Case、Prompt 和策略自进化闭环
│   │   ├── __init__.py                    # 声明进化子包
│   │   ├── collector.py                   # 收集低于 Rubric 门槛的 Bad Case 完整轨迹
│   │   ├── dedup.py                       # 按需求模式限制每天重复 Bad Case 数量
│   │   ├── router.py                      # 将 Bad Case 分流为 P0、P1、P2
│   │   ├── p0_fixer.py                    # 从安全类 P0 Case 自动生成泄露拦截规则
│   │   ├── audit.py                       # 从训练样本中抽取人工审计子集
│   │   ├── prompt_versions.py             # Prompt 语义化版本、状态、变更记录与文件存储
│   │   ├── ab_router.py                   # 按 user_id 稳定 Hash 分配 Prompt A/B 版本
│   │   ├── auto_prompt.py                 # 使用 Judge 模型分析 Bad Case 并提出 Prompt 改进
│   │   ├── cache_migration.py             # Prompt 升级期间的缓存预热和迁移策略
│   │   ├── strategy_extractor.py          # 从高 Rubric 轨迹中提炼可复用成功策略
│   │   ├── strategy_lifecycle.py          # 计算策略随时间与引用次数变化的置信度
│   │   ├── strategy_feedback.py           # 根据后续 Rubric 结果强化或衰减策略
│   │   └── fork_optimizer.py              # 统计平台成功率，为动态 fork 提供排序
│   │
│   ├── eval/                              # 召回与 Agent 质量评测
│   │   ├── __init__.py                    # 声明评测子包
│   │   └── recall_metrics.py              # 计算 Recall@K、MRR 和 NDCG@K
│   │
│   ├── prompt/                            # Prompt 资产和动态提醒
│   │   ├── prompts.yml                    # 完整 system prompt 及 Planner、摘要、压缩、Judge 元提示词
│   │   └── runtime.py                     # 构造不破坏静态缓存前缀的 system-reminder
│   │
│   └── utils/                             # 通用路径和请求上下文辅助函数
│       ├── __init__.py                    # 声明工具函数子包
│       ├── path_utils.py                  # 安全解析项目、上传、输出和会话目录
│       └── thread_ctx.py                  # 绑定、读取和恢复 thread ContextVar
│
├── frontend/                              # React + TypeScript + Vite 前端
│   ├── index.html                         # Vite HTML 入口和 React 根节点
│   ├── package.json                       # 前端依赖及开发、构建命令
│   ├── pnpm-lock.yaml                     # pnpm 锁定的前端依赖版本
│   ├── pnpm-workspace.yaml                # pnpm 工作区与允许构建的依赖配置
│   ├── tsconfig.json                      # TypeScript 严格模式和 JSX 配置
│   └── src/
│       ├── main.tsx                       # 创建 React Root 并挂载 App
│       ├── App.tsx                        # 任务提交、进度事件和最终结果界面
│       ├── useAgentEvents.ts              # 连接任务 WebSocket 并维护事件列表的 Hook
│       └── style.css                      # 前端页面基础视觉样式
│
├── docker/                                # 本地全栈容器编排
│   ├── .env.example                       # Compose 服务间地址和构建目标示例
│   ├── docker-compose.yml                 # Agent、OpenSearch、Redis、vLLM、Reranker 编排
│   ├── Dockerfile                         # Agent dev/prod 多阶段镜像
│   └── Dockerfile.reranker                # 独立 GPU Reranker 服务镜像
│
├── k8s/                                   # Kubernetes 生产部署配置
│   ├── deployment.yaml                    # Agent Deployment、Service、探针与优雅退出
│   ├── ingress.yaml                       # HTTP/WebSocket Ingress 和 thread_id 粘性路由
│   └── canary.yaml                        # 10% 流量的灰度发布 Ingress
│
├── scripts/                               # 数据生产、评测、初始化和预热脚本
│   ├── setup_pipeline.sh                  # 创建 OpenSearch Hybrid Search Pipeline
│   ├── warmup_vllm.py                     # 用代表性请求预热 vLLM KV Cache
│   ├── etl/
│   │   ├── extract_card.py                # 用 Judge 模型从原始资料抽取 CategoryCard 字段
│   │   └── admit.py                       # 对知识卡片做 Schema、置信度、长度与抽审准入
│   └── eval/
│       └── run_category_recall.py          # 读取标注集并运行品类召回指标评测
│
├── examples/                              # 按早期章节组织的最小可运行示例
│   ├── 02_context_demo.py                 # ContextVar 基础设置与读取示例
│   ├── 03_child_context_override_demo.py  # 子任务覆盖并恢复父线程上下文示例
│   ├── 04_monitor_demo.py                 # Monitor 事件上报示例
│   ├── 05_path_utils_demo.py              # 会话目录和安全路径解析示例
│   └── 06_llm_smoke_test.py               # LLM 配置和最小调用冒烟测试
│
├── tests/                                 # 单元测试和跨模块回归测试
│   ├── test_project_setup.py              # Python 版本、依赖和基础目录结构测试
│   ├── test_thread_context.py             # ContextVar 设置、读取和恢复测试
│   ├── test_path_utils.py                 # 路径规范化、目录创建和越界防护测试
│   ├── test_connection.py                 # WebSocket 连接覆盖、断开和路由测试
│   ├── test_monitor.py                    # AGUI 监控事件结构和发送测试
│   ├── test_llm.py                        # 模型环境变量、缓存和初始化测试
│   ├── test_prompts.py                    # Prompt 加载、注入、校验和缓存测试
│   ├── test_chapter11_prompt.py           # ItemSearch 单平台与多平台 fork 规则测试
│   ├── test_tool_registry.py              # 核心工具和 dispatch_tool 注册完整性测试
│   ├── test_dispatch_tool.py              # 子 Agent 上下文、结果和派发行为测试
│   ├── test_fork_guard.py                 # fork 最大深度和 ContextVar 恢复测试
│   ├── test_towers.py                     # 三塔 HTTP 客户端请求和响应校验测试
│   ├── test_ann.py                        # ANN 检索、平台过滤和元数据测试
│   ├── test_item_search.py                # ItemSearch 双路召回、去重和结构测试
│   ├── test_category_insight.py           # 品类卡片 Hybrid 召回、抽取和输出测试
│   ├── test_recall_metrics.py             # Recall、MRR、NDCG 数学实现测试
│   ├── test_fx.py                         # 汇率换算及异常币种测试
│   ├── test_price_compare.py              # 跨币种比价、排序和每平台最低价测试
│   ├── test_shipping_calc.py               # 运费、关税、到手价和时效测试
│   ├── test_price_shipping_pipeline.py    # 比价到运费关税估算的组合链路测试
│   ├── test_item_picker.py                # 硬约束过滤、打分和最多三件限制测试
│   └── test_chapters_16_19.py             # 预算、熔断、Harness、安全、策略和完整 Prompt 测试
│
├── data/                                  # 索引、Trace、Prompt 版本等持久化数据目录
│   └── .gitkeep                           # 保留空数据目录；实际数据被 Git 忽略
├── output/                                # 按 thread_id 保存报告和任务输出
│   ├── .gitkeep                           # 保留空输出目录
│   └── thread-demo/                       # 示例运行产生的会话输出目录
├── uploaded/                              # 按 thread_id 保存用户上传文件
│   ├── .gitkeep                           # 保留空上传目录
│   └── thread-demo/                       # 示例运行产生的上传目录
│
└── globex_agent.egg-info/                 # 本地安装生成的 Python 包元数据
    ├── PKG-INFO                           # 构建后的包名称、版本、依赖和描述
    ├── SOURCES.txt                        # Python 分发包包含的源文件清单
    ├── requires.txt                       # 分发包依赖列表
    ├── dependency_links.txt               # 旧式外部依赖链接元数据
    └── top_level.txt                      # 分发包顶层 Python 包名
```

## 核心运行链路

```text
用户请求
  → FastAPI 创建任务并绑定 thread_id/session_dir
  → 主 AgentLoop 注入长期偏好、历史策略和 Token 预算
  → Planner / 单工具 / dispatch_tool 子 AgentLoop
  → CategoryInsight + ItemSearch 获取候选
  → PriceCompare + ShippingCalc 计算真实到手成本
  → ItemPicker 按约束与偏好精挑
  → ShoppingSummary 生成最终清单
  → Monitor 通过 WebSocket 推送过程和结果
```

## 本地启动

### 1. 安装后端依赖

```bash
uv sync --all-extras
```

复制并填写环境变量：

```bash
cp .env.example .env
```

至少需要配置模型服务：

```dotenv
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_API_KEY=your-key
LLM_MAIN=your-model
```

启动 FastAPI：

```bash
uv run uvicorn app.api.server:app --reload
```

### 2. 启动前端

```bash
cd frontend
pnpm install
pnpm dev
```

### 3. 运行测试

```bash
uv run pytest -q
```

### 4. 使用 Docker Compose

```bash
cd docker
cp .env.example .env
docker compose up --build
```

该方式会统一启动 Agent、OpenSearch、Redis、vLLM 和 Reranker。vLLM 与 Reranker 的 GPU 配置需要根据本机显存和模型大小调整。

## 运行时目录说明

- `data/`：保存向量索引、评测数据、Trace 和 Prompt 版本等持久数据。
- `output/<thread_id>/`：保存某个任务生成的报告和结果文件。
- `uploaded/<thread_id>/`：保存某个任务上传的原始文件。
- `.env`：保存真实密钥，只能本地使用。
- `frontend/dist/`：前端生产构建产物，由 Vite 自动生成。

