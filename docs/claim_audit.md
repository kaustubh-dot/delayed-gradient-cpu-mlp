# Second-pass factual and claim audit

Reviewed the authored report, README, demo, evidence/reproducibility notes, C++ runtime, tests, experiment harness, plotting script, bibliography, and preserved evidence. The numerical recheck used raw CSV/text files independently of the plotting summary. This is a documented audit, not a guarantee that every possible software interleaving or historical fact is known.

## Corrections made

| Item | Correction and evidence |
|---|---|
| Queue timers | The paper previously included container work in the push timer. In `Queue::push` and `Queue::pop`, elapsed time is accumulated immediately after the condition-variable predicate wait, before deque insertion/removal. The paper and documentation now define these as admission durations, not complete-call durations. Worker active-path time does include complete processing/push calls. |
| Queue closure tests | Readiness promises are fulfilled before a thread enters the queue call. The tests exercise closure races and returned results, but do not prove the thread was already blocked. The paper no longer claims that forced blocked-thread wakeups were demonstrated by those tests. Predicate-wait/notification behavior is supported by source inspection; interleavings are not exhaustively tested. |
| Checkpoint test completeness | The fixture runner now requires exactly epochs 1 and 2 before comparing metrics. An empty or truncated CSV can no longer pass through an empty comparison loop. |
| Original timing boundaries | The supplied guide describes broad intervals, but no original source is available to confirm exact timer placement. Evaluation gaps are now described as a plausible explanation of excess over training time, not an independently established component. Overlap with another thread cannot by itself make a single thread's wait exceed its own full lifetime. |
| Metadata contents | Compressed MNIST validation belongs to `download_mnist.py`; experiment metadata stores decompressed SHA-256 hashes. The report no longer says compressed-checksum validation is a metadata field. |
| GPipe bibliography | The published paper's title page lists Mia Xu Chen before Dehao Chen. The reference now follows that title page; the proceedings landing-page metadata differs. SSP and HOGWILD author ordering was also checked against the paper title pages rather than inconsistent landing-page metadata. |
| Rounded ratio | The fresh synchronous/delay-four ratio is 3.124716424. The README's 3.13 was a rounding error; it now reports 3.125 and gives the unrounded value. |
| Architecture cache | The diagram now labels the hidden cache as an input index plus saved activation `h`, matching `Cache` in the source. It does not imply a copied input vector or separately stored mask. |
| Figure scale | Both tradeoff captions explicitly disclose the restricted accuracy axis. Means, observed ranges, and target dots retain their measured values. |

An unnecessary FDG online-publication-year aside was removed. Its journal year, four authors, volume 33, issue 10, and pages 6013–6020 were confirmed with [publisher-deposited DOI metadata](https://api.crossref.org/works/10.1109/TNNLS.2021.3069883).

## Numerical evidence rechecked

The full derived values are in [second_pass_numeric_audit.json](../results/derived/second_pass_numeric_audit.json).

Run `python scripts/audit_claims.py` to independently recheck the raw/summary agreement and table arithmetic. It reads local datasets if present; `--originals PATH` and `--experiment-binary PATH` additionally check the original supplied files and retained experiment executable. A rebuilt executable should not be substituted for that retained binary. The default command verifies without rewriting evidence; `--write` regenerates the audit JSON. GitHub CI runs the default check.

| Check | Scope and result |
|---|---|
| Raw new results | All 24 raw stdout CSVs match all 120 combined epoch rows; raw stderr counters match their summary columns. |
| Counts and bounds | Every epoch has 5,000 output updates and accepted + discarded = 5,000. Accepted ages, returned FIFO ages, and maximum pending counts obey the stated limits. |
| New summary metrics | Final accuracy/loss, cumulative time, acceptance, throughput, accepted throughput, age means/maxima, and first target epoch/time were independently recalculated. |
| Original results | All nine final rows and target epochs/times match the 45 original checkpoints. Original sequential/synchronous quality values coincide. |
| Workbook | Headers and all 45 data rows match the original epoch CSV within 1e-6. |
| Report tables | Every numerical table cell was recomputed independently from the CSVs and checked against its printed rounding. |
| Preserved files | All supplied results, guides, and presentation files byte-match their Downloads originals. |
| Provenance | Current source, retained experiment binary, and all four unpacked dataset hashes match recorded metadata. The four compressed dataset checksums match the resources in [Torchvision's MNIST loader](https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py). |

Key ratios, before rounding:

- Original sequential/delay-four training time: **1.378021827**.
- Original sequential/delay-four first-target time: **1.346020373**.
- New synchronous/delay-four training time: **3.124716424**.
- New sequential/delay-four training time: **0.925607891**, so there is no mean delayed speedup over sequential training.

The 25,450 parameter count, 203,600-byte double parameter payload, 31,360,000-byte selected training-input payload, and 78.4 hidden/output matrix-size ratio agree with the source dimensions. These remain calculated payloads, not measured process memory or service-time ratios.

## Reference checks

All 13 cited works exist. Titles, authors, venue/year, and listed pages were checked against proceedings, paper title pages, publisher-deposited records, or author manuscripts. The paper's descriptions are limited to each cited method's scope:

| Reference | Primary check and claim scope |
|---|---|
| HOGWILD!, 2011 | [Proceedings paper](https://proceedings.neurips.cc/paper/2011/file/218a0aefd1d1a4be65601cc6ddc1520e-Paper.pdf): sparse lock-free shared updates; those guarantees are not claimed for this dense owner-thread MLP. |
| SSP, 2013 | [Proceedings paper](https://proceedings.neurips.cc/paper/2013/file/b7bb35b9c6ca2aee2df08cf09d7016c2-Paper.pdf): bounded worker clocks and stale parameter reads, not this returned-gradient rejection rule. |
| DDG, 2018 | [ICML proceedings](https://proceedings.mlr.press/v80/huo18a.html) and [paper](https://proceedings.mlr.press/v80/huo18a/huo18a.pdf): delayed gradients and conditional critical-point convergence for the authors' methods; no convergence theorem is transferred here. |
| Features Replay, 2018 | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2018/hash/a36b598abb934e4528412e5a2127b931-Abstract.html): parallel-objective formulation and replay; this cache is not that algorithm. |
| FDG, 2022 | [Author manuscript](https://dr.ntu.edu.sg/server/api/core/bitstreams/d8a50737-064e-4a8a-9800-db783e5be7c7/content) and DOI metadata: full decoupling and gradient shrinking; this project implements neither full decoupling nor shrinking. |
| DSP, 2020 | [CVPR proceedings](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_On_the_Acceleration_of_Deep_Learning_Model_Parallelism_With_Staleness_CVPR_2020_paper.html): layer-wise staleness and straggler-aware model parallelism; not a claimed implementation here. |
| DTRP, 2023 | [Publisher article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276427): recomputation, weight prediction, and batch compensation; none is implemented here. |
| Synthetic gradients, 2017 | [ICML proceedings](https://proceedings.mlr.press/v70/jaderberg17a.html): learned predicted error signals; our signals are backpropagated instead. |
| GPipe, 2019 | [Published paper](https://proceedings.neurips.cc/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf): batch splitting and pipeline stages; used as design context only. |
| Local SGD, 2019 | [Author record/manuscript](https://arxiv.org/abs/1805.09767): independent local SGD and periodic averaging, identified as ICLR 2019; not partitioned-layer ownership. |
| Ringmaster, 2025 | [ICML proceedings](https://proceedings.mlr.press/v267/maranjyan25b.html): selective stale-gradient rejection and time-complexity results in its own setting; no equivalence or transferred optimality is claimed. |
| Dissertation, 2026 | [Author submission](https://arxiv.org/abs/2601.02523): explicitly marked PhD thesis, not a new peer-reviewed conference paper. |
| Document recognition/MNIST, 1998 | [Author publication record](https://leon.bottou.org/papers/lecun-98h): bibliographic record and handwritten-digit benchmark context. |

Some primary pages returned transient timeouts or browser checks. Verification used the available proceedings paper, author manuscript, or publisher DOI record; a working fetch from every mirror is not claimed.

## Claims deliberately left limited

The original program's actual execution, CPU/compiler, subset identities, exact timer boundaries, and queue capacity cannot be independently recovered. Supplied observations remain labeled supplied observations, not measurements made by the reconstruction. Preserving a guide is not endorsement of every historical statement in it.

No universal optimal delay, general sequential CPU speedup, statistical significance, formal convergence guarantee, exhaustive race proof, CPU utilization, peak memory, or hardware-profiler result is claimed. Three repeats share one initialization and test subset. The smoothness inequality is explicitly conditional, and model-version age is not wall-clock delay. Original delay-sixteen numbers remain unavailable.

Source inspection supports cached forward masks, pre-update output derivatives, disjoint parameter writes, FIFO admission/age reasoning, and quiescent evaluation. Tests support sampled derivatives and specific runtime contracts. Neither substitutes for exhaustive formal verification or establishes the original executable's behavior.
