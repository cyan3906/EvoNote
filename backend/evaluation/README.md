# Evonote Evaluation

评估目录按层次组织：

- `layer1_l1_l3_claim`: 评估从 L1 原文抽取 L2 摘要、L3 相关实体和 claims 的质量。
- `layer2_rag_retrieval`: 评估 RAG 阶段相似文章检索质量，按 ES、Milvus、RRF 三个通道分别计算 note-level 指标。
- `layer3_human_review`: 预留给人工验证与抽检流程。

## Layer 1

第一层指标输出到 `backend/evaluation/results/layer1_l1_l3_claim_metrics.json`。

```powershell
python backend\evaluation\layer1_l1_l3_claim\evaluate_layer1_l1_l3_claim.py `
  --gold backend\testdata\408_notes_claims_fixture.json `
  --pred path\to\prediction.json `
  --out backend\evaluation\results\layer1_l1_l3_claim_metrics.json `
  --judge heuristic
```

如果不传 `--pred`，脚本会用 gold 文件作为 prediction 跑 smoke baseline。

推荐 prediction 结构：

```json
{
  "notes": [
    {
      "id": "note_001",
      "predicted": {
        "l2": "模型生成的摘要",
        "l3": ["模型抽取的实体"],
        "claims": [
          {
            "claim": "模型抽取的原子断言",
            "evidence": "模型给出的原文证据"
          }
        ]
      }
    }
  ]
}
```

第一层当前记录这些指标：

- `l2_faithfulness_rate`: L2 摘要是否忠实于 L1。
- `l2_coverage_score`: L2 摘要覆盖 gold L2 核心信息的程度。
- `l3_precision`: 预测 L3 实体中有多少匹配 gold L3。
- `l3_recall`: gold L3 实体中有多少被预测覆盖。
- `claim_precision`: 预测 claims 中有多少匹配 gold claims。
- `claim_recall`: gold claims 中有多少被预测覆盖。
- `claim_evidence_support_rate`: 预测 claims 中被 L1/evidence 完全支持的比例。
- `claim_hallucination_rate`: 预测 claims 中不被 L1 支持的比例。
- `claim_partial_support_rate`: 预测 claims 中部分支持、过度泛化或证据不足的比例。


## Layer 2

第二层使用 `backend/testdata/408_note_retrieval_queries_fixture.json` 作为 query gold 数据集，评估对象是整篇 note，不是 claim。

离线评估已有检索结果：

```powershell
python backend\evaluation\layer2_rag_retrieval\evaluate_layer2_note_retrieval.py `
  --queries backend\testdata\408_note_retrieval_queries_fixture.json `
  --results path\to\note_retrieval_results.json `
  --out backend\evaluation\results\layer2_note_retrieval_metrics.json
```

直接调用当前后端 ES、Milvus、RRF 检索：

```powershell
python backend\evaluation\layer2_rag_retrieval\evaluate_layer2_note_retrieval.py --live
```

如果只是验证评估器本身，可以跑 oracle smoke baseline：

```powershell
python backend\evaluation\layer2_rag_retrieval\evaluate_layer2_note_retrieval.py --smoke-oracle
```

推荐检索结果结构：

```json
{
  "queries": [
    {
      "query_id": "retrieval_001",
      "results": {
        "es": [{ "note_id": "note_001", "score": 12.3 }],
        "milvus": [{ "note_id": "note_001", "score": 0.87 }],
        "rrf": [{ "note_id": "note_001", "score": 0.032 }]
      }
    }
  ]
}
```

第二层记录这些指标：

- `recall_at_k`: gold 相关 note 有多少被 topK 召回。
- `precision_at_k`: topK 结果中有多少是 gold 相关 note。
- `hit_rate_at_k`: topK 中至少命中一个相关 note 的 query 占比。
- `mrr`: 第一个相关 note 排名越靠前分数越高。
- `ndcg_at_k`: 按相关性等级计算排序质量。
