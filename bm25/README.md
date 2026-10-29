# BM25：NFCorpus Recall@10 与 nDCG@10

纯 Python 标准库实现，无须安装额外依赖、训练模型或使用训练集。默认参数为 `k1=1.2`、`b=0.75`，未在测试集上调参。

在仓库根目录运行：

```powershell
python .\bm25\run_bm25.py
```

脚本默认根据自身位置寻找 `..\nfcorpus`，也可以从任意目录运行。需要时可指定路径和参数：

```powershell
python .\bm25\run_bm25.py --data-dir Q:\repos\ranking_retrieval\nfcorpus --k1 1.2 --b 0.75 --output .\bm25\results.json
```

## 数据与评估口径

- 检索范围：`nfcorpus/test.docs`，共 3162 篇文档。
- 测试查询：`nfcorpus/test.titles.queries`，共 325 条标题查询。
- 相关性标注：`nfcorpus/test.2-1-0.qrel`，相关度大于 0 即为正例；数据集自带 README 指明该版本用于原实验。
- 323 条查询有正例，逐条计算 `前 10 篇中命中的正例数 / 该查询全部正例数`，再取宏平均。
- nDCG@10 保留标注中的相关度等级 1 和 2，按下述公式逐条计算，再对同样的 323 条查询取宏平均。未标注文档的相关度视为 0。
- `PLAIN-1078` 和 `PLAIN-860` 没有正例，剔除并报告，不计入平均。

结果写入 `bm25/results.json`：包含配置、整体 Recall@10 与 nDCG@10、查询数量与跳过 ID，以及每条有效查询的命中数、正例数、两项指标和前 10 个文档 ID。

nDCG 使用与 [trec_eval 的 ndcg_cut](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c) 一致的线性增益，即直接使用相关度等级：

$$
\operatorname{DCG}@10=\sum_{i=1}^{10}\frac{\operatorname{rel}_i}{\log_2(i+1)},\qquad
\operatorname{nDCG}@10=\frac{\operatorname{DCG}@10}{\operatorname{IDCG}@10}.
$$

IDCG 将该查询全部正例按相关度降序排列后取前 10 篇计算；正例不足 10 篇时使用全部正例。

## 实现约定

输入采用本地数据集提供的文本，再统一小写并用 `[a-z0-9]+` 分词；不新增停用词过滤或词干化。倒排表仅累加查询词对应的文档分数，重复查询词只计一次。所有文档都参与前 10 排名，同分按文档 ID 升序；不足 10 个正分文档时由零分文档补齐。

使用非负 IDF 的 BM25：

$$
\operatorname{BM25}(q,d)=\sum_{t\in\operatorname{unique}(q)}\ln\left(1+\frac{N-df_t+0.5}{df_t+0.5}\right)\frac{tf_{t,d}(k_1+1)}{tf_{t,d}+k_1(1-b+b|d|/avgdl)}.
$$

这里的结果依赖该分词、查询版本和 3162 篇测试文档范围；与使用完整语料或不同分析器的 BEIR / Elasticsearch 结果不直接等价。
