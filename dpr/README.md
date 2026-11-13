# DPR（EMNLP 2020）复现说明

目标论文：[Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550/)（[PDF](https://aclanthology.org/2020.emnlp-main.550.pdf)）。本目录保留本地核心算法 smoke test、官方预计算产物/官方 checkpoint 核验，以及论文规模从头训练的路线；末尾另记录 Deeprank Lightning/AML 的 NQ 训练与 NFCorpus 域适配实验。这些证据与实验指标不能互相替代。

## 代码边界

- [`official/`](official/) 是上游 [`facebookresearch/DPR`](https://github.com/facebookresearch/DPR/tree/d7ba973ab2d862d1b44c2402b59b1ca2a362ee96) 在提交 `d7ba973ab2d862d1b44c2402b59b1ca2a362ee96`（2020-06-30）的固定快照，不跟随当前 `main`；来源、Git tree 和许可证见 [`UPSTREAM.md`](UPSTREAM.md)。
- [`core.py`](core.py) 是便于阅读和离线验证的本地最小实现，并非官方代码；[`verify_core.py`](verify_core.py) 使用微型随机编码器测试双塔、损失和指标。
- [`scripts/`](scripts/) 是下载、环境、核验和训练辅助脚本；[`outputs/`](outputs/) 保存可审计的本地结果。
- 后续上游 checkpoint、配置或 Hydra 接口不得混入 2020 路线，否则版本和指标口径会漂移。

## 方法与数据

DPR 使用两个参数独立的 `bert-base-uncased` 编码器：问题编码器 $E_Q$ 和段落编码器 $E_P$。取 `[CLS]` 的 768 维向量，以内积

$$
\operatorname{sim}(q,p)=E_Q(q)^\top E_P(p)
$$

打分。段落向量离线计算并建立 FAISS 索引，查询时执行最大内积检索（MIPS）。训练样本包含一个正段落、一个 BM25 hard negative，并把同一 mini-batch 中其他问题的正段落作为 in-batch negatives：

$$
\mathcal{L}_i=-\log\frac{\exp(\operatorname{sim}(q_i,p_i^+))}{\exp(\operatorname{sim}(q_i,p_i^+))+\sum_j\exp(\operatorname{sim}(q_i,p_{i,j}^-))}.
$$

端到端系统再用 extractive reader 联合预测段落相关性及答案 span 的起止位置。论文语料是 2018-12-20 英文 Wikipedia，清洗后切成 21,015,324 个不重叠的 100-word passages，并在正文前加入页面标题；本路线以 Natural Questions（NQ）为主。更完整的数据格式和参数说明见 [`official/README.md`](official/README.md)。

## 验收口径

论文 NQ test 的目标值为：

| 指标 | 论文目标 | 含义 |
|---|---:|---|
| Top-20 retrieval accuracy | 78.4% | 前 20 个段落中至少一个包含任一标准答案 |
| Top-100 retrieval accuracy | 85.4% | 前 100 个段落中至少一个包含任一标准答案 |
| End-to-end Exact Match | 41.5 | reader 最终答案与标准答案规范化后完全一致 |

严格复现必须固定数据划分、Wikipedia 版本、answer matching/tokenizer、checkpoint 或训练配置，并分别报告 retriever 与 reader。Top-$k$ answer containment 不是“命中指定 gold passage”，也不是端到端 EM。

### 已完成的本地证据

官方托管的 `nq-test.json.gz` 预计算检索结果已经在本机对 3,610 个问题实际核验；完整记录见 [`outputs/nq-test-precomputed-metrics.json`](outputs/nq-test-precomputed-metrics.json)：

| Top-$k$ | 1 | 5 | 20 | 100 |
|---|---:|---:|---:|---:|
| 本地核验 | 46.34% | 68.28% | 80.06% | 86.09% |

缓存的 `has_answer` 与按论文时代 DrQA `SimpleTokenizer` 规则重新分词匹配，在全部 361,000 个候选 context 上一致，并得到相同的四个 Top-$k$ 汇总值。这些是“发布的预计算产物”的核验值，Top-20/100 并非论文表中的 78.4/85.4；它没有重新运行编码、FAISS 检索或 reader，因此**不能据此声称严格重现论文结果**。

可重复执行（在仓库根目录，PowerShell）：

```powershell
python -m venv dpr\.venv
dpr\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu
dpr\.venv\Scripts\python.exe dpr\scripts\download_nq_resources.py --group precomputed --execute
dpr\.venv\Scripts\python.exe dpr\scripts\verify_precomputed.py --output dpr\outputs\nq-test-precomputed-metrics.json
```

同一安装命令也记录在 [`requirements-smoke.txt`](requirements-smoke.txt)；若网络代理不能解析 PyTorch 的依赖，请先修复 pip/PyPI 代理，而不要复用论文旧环境。

## 路线 0：CPU smoke test

使用上节创建的 DPR 专用 CPU 环境，从仓库根目录运行：

```powershell
dpr\.venv\Scripts\python.exe dpr\verify_core.py
```

预期包含 `DPR core verification passed`。该测试不下载 BERT 或 NQ，只验证独立双塔、`[CLS]` pooling、内积、in-batch NLL、一次小型优化及 Top-$k$ 计算。即使通过，也只说明 [`core.py`](core.py) 的数学和张量流程自洽，**不等于复现论文 checkpoint、检索指标或 EM**。

## 运行条件

论文 retriever 与 reader 都是在单机 `8 × 32 GB` GPU 上训练。全量 NQ 检索建议至少约 `120 GB RAM`；仅 21M 个 `float32 × 768` 向量已约 64.6 GB（十进制），FAISS、passage 文本、pickle 和 Python 对象还会增加占用。官方 checkpoint 全量检索组下载约 `72 GB`，解压和输出需要额外磁盘空间。

当前机器未检测到 NVIDIA/CUDA GPU，物理内存为 64 GB；它适合 smoke test 和流式预计算结果核验，但不足以安全执行全量 FAISS 检索，更无法按论文设置训练。请转到 Linux GPU 服务器，或在 WSL2 中使用已配置的 NVIDIA CUDA。

## 环境：Linux/WSL，锁定旧依赖

`official/` 使用 2020 年 `transformers.modeling_bert` 等旧接口。建议 Python 3.7/3.8；提供的脚本固定 Python 3.8，并锁定 PyTorch 1.7.1、Transformers 2.11.0、NumPy 1.19.5 和 FAISS 1.7.2。FAISS 1.7.2 是可获得性较好的替代版本；若能安装上游时代的 1.6.x，应记录实际版本。不要直接用现代 Python/Transformers 安装 `official/setup.py`。

```bash
cd /mnt/q/repos/ranking_retrieval/dpr
bash scripts/prepare_paper_env.sh
source paper-env/bin/activate
```

脚本见 [`scripts/prepare_paper_env.sh`](scripts/prepare_paper_env.sh)。为可追溯性，应保存 `pip freeze`、GPU/驱动/CUDA 版本、随机种子和命令输出。

## 路线 1：官方 checkpoint 核验

先做 dry run 查看约 72 GB 的资源清单，再显式下载；下载器支持 `.part` 断点续传和文件大小校验。

```bash
cd /mnt/q/repos/ranking_retrieval/dpr
source paper-env/bin/activate
python scripts/download_nq_resources.py --group paper-eval
python scripts/download_nq_resources.py --group paper-eval --execute
python scripts/extract_nq_archives.py --resource wiki --execute
bash scripts/run_nq_retrieval.sh
```

最后一条命令使用官方 NQ biencoder checkpoint、官方 50 份 Wikipedia embeddings 和完整 passage TSV，重新编码问题、建立 flat FAISS 索引并计算 Top-100；脚本见 [`scripts/run_nq_retrieval.sh`](scripts/run_nq_retrieval.sh)。它要求 CUDA，并在 Linux 上检查约 100 GiB 可用内存。本机尚未运行这条全量命令。

若还要核验 41.5 EM，需要下载官方 reader checkpoint 与由同一 NQ retriever 产生的 reader test 输入，再执行 reader 推理：

```bash
cd /mnt/q/repos/ranking_retrieval/dpr/official
../paper-env/bin/python data/download_data.py --resource checkpoint.reader.nq-single.hf-bert-base --output_dir ../downloads
../paper-env/bin/python data/download_data.py --resource data.reader.nq.single.test --output_dir ../downloads
../paper-env/bin/python train_reader.py \
  --dev_file ../downloads/data/reader/nq/single/test.pkl \
  --model_file ../downloads/checkpoint/reader/nq-single/hf-bert-base.cp \
  --prediction_results_file ../outputs/nq-reader-predictions.json \
  --eval_top_docs 10 20 40 50 80 100 \
  --dev_batch_size 8 --passages_per_question_predict 100 --sequence_length 350
```

`--dev_batch_size` 可按显存调整；验收读取日志中的 `n=100 EM`。这一步也尚未在当前机器执行。

## 路线 2：NQ 从头训练

下载训练集并解压，然后按快照 README 的 8-GPU 配置训练 40 epochs：

```bash
cd /mnt/q/repos/ranking_retrieval/dpr
source paper-env/bin/activate
python scripts/download_nq_resources.py --group paper-train --execute
python scripts/extract_nq_archives.py --resource train --resource dev --execute
WORLD_SIZE=8 bash scripts/train_nq.sh
```

训练命令见 [`scripts/train_nq.sh`](scripts/train_nq.sh)：每 GPU batch 16，跨 8 卡有效 batch 128，每个问题使用一个 BM25 hard negative，epoch 30 后启用 average-rank validation。该快照的“best settings”命令使用学习率 `2e-5`，而论文正文写 `1e-5`；必须把这一差异写入实验记录，分别报告结果，不能挑选更接近目标的数字后称为严格复现。

训练后的 passage encoder 与官方预计算 embeddings 不匹配，**不得复用路线 1 的 50 份 embeddings**。应以新 checkpoint 对 21M passages 重新分片编码，再检索：

```bash
cd /mnt/q/repos/ranking_retrieval/dpr
source paper-env/bin/activate
MODEL=/absolute/path/to/runs/nq-paper/dpr_biencoder.CHOSEN_CHECKPOINT
SHARD_ID=0  # 在 GPU 节点上分别提交 0..49
cd official
python generate_dense_embeddings.py \
  --model_file "$MODEL" \
  --ctx_file ../downloads/data/wikipedia_split/psgs_w100.tsv \
  --shard_id "$SHARD_ID" --num_shards 50 --batch_size 128 \
  --out_file ../runs/nq-paper/wiki_passages
cd ..
MODEL="$MODEL" EMBEDDINGS="$PWD/runs/nq-paper/wiki_passages_*" \
  OUTPUT="$PWD/outputs/nq-test-from-scratch.json" bash scripts/run_nq_retrieval.sh
```

要复现端到端 EM，还需用新 retriever 生成 train/dev/test Top passages，预处理 reader 数据，按 [`official/README.md`](official/README.md) 的 reader 配置训练约 16–18 epochs（约 60k updates），再在 NQ test 上评估。这部分成本高于只训练 retriever。

## Windows 常见问题

- 原生 Windows 对旧版 `faiss-cpu`、NCCL、Apex/FP16 和 `torch.distributed.launch` 支持不可靠；论文路径使用 Linux/WSL2。
- 快照的 `dense_retriever.py` 在问题编码处直接调用 `.cuda()`，`--no_cuda` 不能把全量检索变成 CPU 路径。
- `.sh`、Linux glob 和 `$VAR` 必须在 Bash 中运行；`Q:\repos\...` 在默认 WSL 挂载下对应 `/mnt/q/repos/...`。
- 下载与解压前先确认磁盘空间；不要把 `.part` 文件手工改名，重跑下载器即可续传。
- 记录 WSL、CUDA、驱动与依赖版本；不同 Transformers/FAISS 版本和 tokenizer 细节都可能改变指标。

截至本说明更新，本机已完成 CPU smoke 与官方预计算 `nq-test` 产物核验；另外在 deeprank 的 AML 环境完成了 NQ retriever 训练及 NFCorpus 适配实验，见下节。仍未用新训练的 NQ encoder 对完整 Wikipedia 重新编码并计算论文口径的 NQ Top-$k$，也没有运行 reader EM。因此，官方预计算产物的指标与本次训练模型的指标应分别解读。

## Deeprank Lightning / SMILE 迁移与 NFCorpus 迭代

本目录的 `official/` 固定快照和 `core.py` 保留上述各自用途。实际 GPU 训练实现位于 `Q:/repos/deeprank`：Lightning 提供数据加载、双塔与训练逻辑，SMILE 提供 AML pipeline 和参数配置。`deeprank_repro/` 按原目录层级保存这部分代码的镜像，便于在本目录查看和追溯；它不包含完整 deeprank 运行环境、训练数据或大型 checkpoint，**不能单独作为 AML 项目提交**。AML 提交仍从 `Q:/repos/deeprank/smile` 执行。镜像中的主要入口为：

- `deeprank_repro/lightning/src/smile_lightning/apps/retrieval/dpr/`：NQ DPR 双塔、数据和训练模块。
- `deeprank_repro/lightning/src/smile_lightning/apps/retrieval/nfcorpus/`：NFCorpus 微调、负例与完整 dev 检索评估。
- `deeprank_repro/lightning/conf/apps/retrieval/` 和 `deeprank_repro/lightning/scripts/retrieval/`：Hydra 配置及独立评估脚本。
- `deeprank_repro/smile/src/smile/pipelines/retrieval/`：NQ 训练与 NFCorpus 微调的 SMILE pipeline，包括 `default` 点积和 `cosine` 两种配置。

### 实验顺序

1. **NQ 双塔训练。** 在 AML 上用 `bert-base-uncased` 初始化两个参数独立的 BERT 编码器，以 NQ train/dev 训练 DPR。根 Run [`7947a6ce-47aa-40bb-acad-c1d35e85bb31`](https://ml.azure.com/runs/7947a6ce-47aa-40bb-acad-c1d35e85bb31?wsid=/subscriptions/07c3f5a2-0880-4a37-b678-e29ade7b1036/resourcegroups/singularitywestus3/workspaces/wxtcssingwestus3) 的选用 checkpoint 对应 epoch 39、step 18400；后续微调配置引用其子运行 `1942a01b-de55-4f9a-8efe-6cb5cb4e2a20` 下的 `best.ckpt`。尚未用这个模型完成上文所述的 21M Wikipedia passages 检索及 reader 评估。
2. **跨域诊断。** 直接把 NQ checkpoint 用于 NFCorpus test，原始标题与摘要分块口径的 Recall@10 / nDCG@10 为 `0.074246 / 0.167528`，低于同一数据集上的 BM25 和已用 NFCorpus train 微调的 SBERT。NQ 与医学文档的领域、查询类型和监督目标均不同；这是进一步做域内训练的动机，而不是单独证明哪项差异造成下降。
3. **NFCorpus 点积微调。** 根 Run [`d2f31491-2e27-4666-a93c-62fe0be1bb25`](https://ml.azure.com/runs/d2f31491-2e27-4666-a93c-62fe0be1bb25?wsid=/subscriptions/07c3f5a2-0880-4a37-b678-e29ade7b1036/resourcegroups/singularitywestus3/workspaces/wxtcssingwestus3) 从 NQ checkpoint 加载两塔权重，重建优化器，仅用 NFCorpus train 微调，以完整 dev Recall@10 选取 epoch 2、step 5172 的 checkpoint。Test 提升至 `0.151900 / 0.316789`。这是**顺序域适配**，并未把 NQ 与 NFCorpus 样本混合进同一训练阶段。
4. **固定权重诊断。** 对上述点积 checkpoint 的同一组 dev 向量，直接把推理打分由点积改为 cosine：Recall@10 从 `0.150851` 降到 `0.142634`，nDCG@10 从 `0.311722` 降到 `0.302102`。仅替换推理公式没有改善模型；该诊断不使用 test 选 checkpoint。
5. **归一化向量重训。** 根 Run [`c70c1303-4d7b-4afb-9100-cacde1863193`](https://ml.azure.com/runs/c70c1303-4d7b-4afb-9100-cacde1863193?wsid=/subscriptions/07c3f5a2-0880-4a37-b678-e29ade7b1036/resourcegroups/singularitywestus3/workspaces/wxtcssingwestus3) 再次从同一 NQ checkpoint 独立开始 NFCorpus 微调，分别归一化查询和文档块的 `[CLS]` 向量，并在训练损失中使用 $0.05$ 温度（分数乘 20）。以 dev Recall@10 选出的 checkpoint 是 epoch 4、step 8620；Test 达到 `0.174801 / 0.354231`。它不是接续点积微调 checkpoint 训练。

### NFCorpus test 对照

下表共用 `test.2-1-0.qrel`、3,162 个 test 文档 ID、325 个查询 ID；323 条有正例的查询参与宏平均，另 2 条跳过。Recall@10 是每条查询前 10 个文档的正例命中数除以该查询的全部正例数，再对查询平均；nDCG@10 使用等级 1/2 的**线性增益**。DPR 使用原始查询、原始文档标题与摘要、256-token 分块与 64-token overlap，并取各块最高分作为文档分数。SBERT V3 也使用原始文本分块和同一 qrels/指标，但其共享编码器、mean pooling 与具体分块方式不同；BM25 使用原始标题和摘要文本。下表是同一检索任务与指标口径的比较，并非完全相同的模型输入处理。

| 模型 / 训练来源 | Recall@10 | nDCG@10 |
|---|---:|---:|
| NQ-only DPR，未用 NFCorpus train | 0.074246 | 0.167528 |
| BM25，原始标题＋摘要 | 0.153696 | 0.316185 |
| SBERT V3，NLI 初始化＋NFCorpus train | 0.160806 | 0.327137 |
| DPR，NQ 初始化＋NFCorpus 点积微调 | 0.151900 | 0.316789 |
| **DPR，NQ 初始化＋NFCorpus cosine 微调** | **0.174801** | **0.354231** |

原始结果保存在 `Q:/repos/deeprank/lightning/outputs/dpr_nfcorpus_7947/raw_chunks_test.json`、`Q:/repos/deeprank/lightning/outputs/dpr_nfcorpus_d2f31491/nfcorpus_test_raw_chunks.json`、同目录 `nfcorpus_dev_dot_cosine.json` 与 `Q:/repos/deeprank/lightning/outputs/dpr_nfcorpus_cosine_c70c1303/nfcorpus_test_raw_chunks.json`。BM25 和 SBERT V3 的结果分别在 `Q:/repos/ranking_retrieval/sbert/runs/nfcorpus-recall-raw-chunks/bm25-test.json` 与同目录 `nfcorpus-test-results.json`。DPR dev 诊断使用 3,193 篇文档、325 个查询，其中 324 条参与指标计算；**不可与上表 test 数字混列**。

### 如何解读

- 本次 cosine DPR 相比点积微调，Test Recall@10 提高 `0.022901`、nDCG@10 提高 `0.037442`。但 cosine 实验同时改变了归一化和训练 logit 尺度，不能把全部增益归因于归一化本身，也不能把与 SBERT 的差异单独归因于双塔参数更多。
- 在这一次 Test 中，cosine DPR 高于 SBERT V3；对 323 条查询做配对重采样，DPR−SBERT 的 Recall@10 差值 `+0.013995`，95% 区间 `[-0.007764, +0.035886]`；nDCG@10 差值 `+0.027094`，区间 `[-0.001592, +0.055135]`。两区间都跨零，不足以声称稳定领先。
- 有 4 条 Test 查询原文与 Train 查询重复；DPR 和 SBERT 的训练都保留了对应样本。排除这 4 条后，319 条有正例查询上的 cosine DPR 为 `0.171806 / 0.349990`，SBERT 为 `0.158578 / 0.324434`。该检查未消除 Test 在迭代中已被查看的限制；后续模型选择应以 dev 为准。
