# Evonote Evaluation

评估目录按层次组织：

- `layer1_l1_l3_claim`: 评估从 L1 原文抽取 L2 摘要、L3 相关实体和 claims 的质量。
- `layer2_rag_retrieval`: 预留给 RAG 检索召回评估。
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

