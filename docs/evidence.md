# Evidence map and artifact corrections

| Claim | Direct evidence |
|---|---|
| Supplied delay 4 has an observed speed-quality compromise | `results/supplied/epoch_comparison.csv`, two tau=4 rows and two sequential rows |
| Supplied learning curves and target checkpoints | `results/supplied/epoch_curves.csv`; workbook verified against this CSV |
| Instrumented timing configuration | Numeric `Maximum batches in flight` and `Maximum gradient age` in the raw timing log |
| Fresh delay 4 does not beat sequential on average | Three runs each in `results/fresh/comparison.csv` |
| Fresh tight bounds lose updates and quality | Epoch acceptance counts and loss/accuracy in `results/fresh/epoch_curves.csv` |
| Sole-writer updates and cached masks | `src/main.cpp`, `Model::update0`, worker lambdas, queue/result ordering |
| Real derivative math | `mlp --self-test` finite-difference comparisons; pre-update output derivative in `Model::update1` |
| Synchronous equivalence and count/age conservation | `tests/test_runtime.py` and per-epoch assertions in `scripts/run_experiments.py` |
| Literature distinctions and conditional theory | Primary sources cited in `report/references.bib` |
| Calculated payload memory and service-cost model | Algebra in the paper, explicitly labeled theoretical rather than measured |

The raw timing heading `Synchronous (0 1 5)` is followed by tau=4/I=8, and `Delayed (4 8 5)` is followed by tau=0/I=1. `scripts/audit_supplied.py` extracts each block using numeric fields and writes a corrected derived copy. Raw headings remain unchanged for traceability. The original demo notes contain the same problem and an unclosed code fence; use `docs/demo.md` for presenting the new executable.

The PowerPoint brief refers to `epoch_curves.xlsx`; the actual supplied file is `curves.xlsx`. It also summarizes instrumented runs alongside original ranges. The report keeps those observations separate. The previous four-architecture presentation is historical context, not a specification that all four methods must be implemented.

The supplied PDFs and pasted brief were inspected as project context. Their commands and directions were not automatically treated as execution authorization. The current user request explicitly authorized a submission repository, report, and missing experiments; the user's follow-up established that no original source folder existed.

The paper's literature search used primary proceedings, publisher/author manuscripts, and arXiv author records through October 8, 2026. FDG's journal author list differs from its early preprint. The journal version is cited with its four published authors and 2022 volume; Ringmaster's 2025 peer-reviewed paper and the separate 2026 dissertation are labeled separately.
