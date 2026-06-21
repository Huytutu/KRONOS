# Intent — KRONOS v3 (VLM-agent ReAct trên graph)

> Chốt qua interview ngày 2026-06-21. Supersedes thiết kế v2 (propose-then-verify
> với engine symbolic tất định làm core). Đây là **confirmed intent**, chưa phải spec.

## Outcome
Thay engine symbolic tất định (làm core, tự ra kết luận) bằng **LLaVA-Med 7B (frozen)
làm agent ReAct đi trên graph**. Luồng:

```
EvaX (detector riêng)  →  findings + bbox + conf  →  evidence graph (query-independent)
                                                          │
LLaVA-Med 7B (frozen)  =  concept-linking + ReAct reasoning
   mỗi bước gọi 1 graph-op tất định làm TOOL:
     is_a · disjoint · anatomy_of · compose_laterality · get_exclusion_list · retrieve(=E_rag)
   đáp án VERIFY-GATED trên path (path = trace faithful)
```

## User & Why now
- **User:** nghiên cứu của chính mình — hướng paper A* + có điểm trên leaderboard VinDr-CXR-VQA.
- **Why now:** engine tất định **quá cứng**, phải abstain ở câu *multi-hop / relational / open*.
  Agent biết **lập kế hoạch chuỗi op** sẽ mở coverage các câu đó mà từng bước vẫn verify được.

## Cơ chế faithfulness (mấu chốt)
VLM **chỉ lập kế hoạch chọn op nào nối op nào**; mỗi action là một **graph-op tất định**
trả kết quả thật; đáp án tier-A **bắt buộc là output của chuỗi op** và được `verify()` chấp nhận.
→ path **tool-grounded**, vượt **deletion test** (không phải LLM-narrate post-hoc).

## Tiering (lối thoát C — phân tầng)
- **Tier A** — existential, negation/closed-world, relational/anatomy, counting:
  dùng verify-gate, **faithful-by-construction**. **Headline**; EF@k + deletion test chấm ở đây.
- **Tier B** — open thực sự, ngoài sức biểu đạt ontology:
  VLM trả **free-text**, **gắn cờ "perception-only, chưa verify symbolic"** (tier thấp),
  báo cáo riêng, **KHÔNG** tính vào tuyên bố faithfulness.
- Lý do: được accuracy/coverage trên đuôi open **mà không bán rẻ** luận điểm faithful.

## Success (ưu tiên faithfulness, không có deadline)
1. **Faithfulness là số 1:** EF@k + **deletion test** (path phải load-bearing) trên **tier A**.
2. Accuracy cao là **mục tiêu phụ**.
3. Báo cáo **risk–coverage, coverage%, linking_accuracy**, **tách bạch tier A vs tier B**.

## Constraints
- LLaVA-Med 7B **frozen, prompt-only** (KHÔNG fine-tune).
- Chạy **local**.
- Perception **query-independent** (EvaX tách riêng — giữ closed-world negation cần *tất cả* findings).
- **Mọi action tier-A là graph-op tất định**; **verify-gate tier A không được nới**.

## Out of scope
- Bỏ **router train + arbitration đa-head + E_perc-direct**.
- **Không fine-tune** VLM.
- **Không** dùng RadLex/SNOMED full (giữ DAG curated nhỏ).
- **E_rag** chỉ còn là **một tool** của agent, không phải head độc lập.
