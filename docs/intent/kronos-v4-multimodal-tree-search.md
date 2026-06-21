# Intent — KRONOS v4 (Verifier-Guided Multimodal Tree Search)

> Chốt qua interview ngày 2026-06-21. Tiến hóa của [[kronos-v3-vlm-agent]]:
> giữ graph + verify-gate + tier A/B, **thêm** tree search (kiểu LATS) + visual actions.
> Target: **MICCAI** (faithfulness-first, không deadline). Đây là confirmed intent, chưa phải spec.

## Ý tưởng một câu
Tác tử VLM tìm kiếm trên **cây** các hành động **đan xen thị giác + ký hiệu**, dẫn bởi
**verifier tất định** (KHÔNG phải LLM self-eval); **path root→leaf thắng cuộc = trace
bằng chứng audit được + nhân quả** (deletion test). `re_detect` cho phép reasoning
**nhìn lại ảnh để vá lỗi perception**.

## Lấy gì từ LATS (và vứt gì)
- **LẤY:** backtracking + cây (thay ReAct tuyến tính của v3); self-reflection **chỉ làm
  heuristic dẫn search**, không vào trace.
- **VỨT:** value function = LLM tự chấm (giết faithfulness). Thay bằng **verifier-as-value**.
- **CẨN TRỌNG:** full MCTS rollout có thể thừa với graph nhỏ → chỉ dùng nếu best-first
  không đủ, và **phải chứng minh bằng số**.

## Ba đóng góp
- **C1 — Verifier-guided multimodal tree search:** reward = *closure-progress của verifier*
  (tồn tại: có witness path chưa; phủ định: tỉ lệ exclusion-list đã kiểm; quan hệ:
  anatomy/laterality đã resolve chưa) + kiểm tra nhất quán (không vi phạm disjoint).
  Terminal: verify-gate PASS (Hạng A) scale theo min-confidence; Hạng B trung bình + cờ;
  FAIL/abstain = 0. Cây bù cho 7B-planner-kém; nhánh thắng tự khắc faithful.
- **C2 — Faithfulness đa phương thức:** ký hiệu = verify-gate *sound*; thị giác =
  *evidence-traceable + deletion-test* (KHÔNG claim provably-sound cho neural). Mỗi visual
  step cite vùng bác sĩ audit được.
- **C3 — Reasoning vá perception:** `re_detect` bắt lại nốt nhỏ EvaX bỏ sót → multimodal
  reasoning kéo *cả accuracy*, không chỉ faithfulness. (Lý lẽ "hiệu quả" mạnh nhất cho MICCAI.)

## Pipeline
```
EvaX quét query-independent  →  evidence NỀN (giữ closed-world cho phủ định)
        ↓
graph bằng chứng (findings ∪ ontology DAG)
        ↓
LLaVA-Med 7B (frozen) SEARCH CÂY, mỗi node = trạng thái suy luận:
  • visual actions:  inspect(bbox)=LLaVA-Med trên crop · re_detect(region)=EvaX trên sub-region · compare(r1,r2)
  • symbolic actions: is_a · disjoint · get_exclusion_list · anatomy_of · compose_laterality · retrieve(=E_rag)
  reward = verifier closure-progress (KHÔNG phải LLM self-eval)
  visual action chỉ THÊM/tinh chỉnh fact (vùng+nhãn+conf), fold lại vào graph — KHÔNG thay nền
        ↓
verify-gate + phân Hạng A/B → path thắng = trace
```

| Visual action | Ai chạy | Trả về | Visual verifier |
|---|---|---|---|
| `inspect(bbox)` / zoom | LLaVA-Med trên crop | xác nhận/bác finding + conf | cite vùng đã nhìn; deletion-test |
| `re_detect(region)` | EvaX trên sub-region | finding mới + bbox + conf | bbox audit được; IoU |
| `compare(r1, r2)` | LLaVA-Med / so đặc trưng | quan hệ (vd trái đậm hơn phải) | hai vùng cite được |

## Faithfulness (định nghĩa chính xác)
- **Hạng A** = verify-gate ký hiệu trên path → *sound* cho phần ký hiệu.
- **Phần thị giác** = *evidence-traceable* (cite vùng) + *deletion-test* (xóa vùng/evidence →
  đáp án đổi). **Hạ chuẩn** từ "provably-sound" xuống "tra được + nhân quả" cho neural — đây
  là claim trung thực nhất, không hứa hão.
- EvaX nền giữ closed-world; visual query-conditioned chỉ được thêm, không thay nền →
  không phá phủ định, không tái lập shortcut "câu hỏi mồi nhận thức".

## Đánh giá
- **Datasets:** VinDr-CXR-VQA (bbox gold) → PadChest-GR (negation) → GEMeX (scale).
- **CHẠY TRƯỚC:** *reasoning-need diagnostic* — % câu thực sự cần reasoning vs pure perception.
  Nếu thấp → thu hẹp claim về subset cần reasoning.
- **Metrics:** EF@k + **deletion test** (headline faithfulness), selective-risk/coverage,
  accuracy/F1, linking-acc, + **audit định tính/bác sĩ** (MICCAI cần).
- **Ablation:** vs linear ReAct (bỏ cây) · vs LLM-value (LATS gốc) · vs no-visual (chỉ ký hiệu)
  · vs no-verify-gate · vs no-`re_detect`.

## Ràng buộc
- Frozen hoàn toàn, **KHÔNG train**: LLaVA-Med 7B + EvaX đóng băng (chỉ prompt); engine CPU.
- Chạy local.

## Out of scope
- Không fine-tune.
- **Không** claim provably-sound cho bước neural.
- **Không** full MCTS rollout nếu best-first đủ (chứng minh bằng số).
- **Không** thay evidence nền bằng visual (giữ closed-world).

## Positioning (novelty delta)
- vs **LATS**: value = verifier tất định, không phải LLM self-eval; trace faithful.
- vs **KRONOS v3**: có tree search + backtracking + visual actions (v3 tuyến tính, thuần ký hiệu).
- vs **visual-MCTS** (Mulberry/AR-MCTS): reward tất định + deletion-test nhân quả, không reward học/LLM.
