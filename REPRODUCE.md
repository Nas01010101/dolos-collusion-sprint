# Reproduce

## Audit pipeline

The other five hypotheses are general sandbagging (MMLU-Pro), hazardous-knowledge sandbagging (WMDP), sandbagging under an evaluation cue, denial of known facts (MASK), and leniency as a code monitor toward copies of itself. They return nulls, uniform shifts, or a ceiling (both models flag every snippet).

![Audit summary](figures/dolos_audit.png)

```mermaid
flowchart TD
    A["Dolos-1.5B and Qwen2.5-1.5B-Instruct"] --> B["Divergence scan: per-token KL on 15 prompt sources<br/>kl_scan.py"]
    B --> C["Largest KL on multiple-choice answers<br/>turn1_format.py"]
    C --> D["Pushback test on 400 items, three follow-ups, deference prompt<br/>syc_flip.py, flip_stats.py"]
    D --> E["Error sensitivity: flips on wrong vs right first answers<br/>audit_stats.py, review_stats.py"]
    A --> F["Alternative hypotheses<br/>capability.py, wmdp.py, mask_belief.py, monitor.py"]
    A --> G["Weight diff and low-rank structure<br/>weight_diff.py, delta_structure.py, rank1_chain.py"]
    G --> H["Tensor swaps between the models<br/>perturb.py swap"]
    D --> I["Reference-free statistics on three other small models<br/>calib_flip.py, audit_stats.py"]
```

## Results

All values are from `report/dolos.md`; the file column names the result file in `redteam/results/`.

| Result | Dolos-1.5B | Qwen2.5-1.5B-Instruct | Section | File |
|---|---|---|---|---|
| First-turn accuracy, 400 items | 0.478 | 0.483 | 4.2 | `flip_summary.csv` |
| Flip rate after "Are you sure?" | 4.5% (95% CI [2.5, 6.8]) | 24.0% ([19.8, 28.3]) | 4.2 | `flip_summary.csv`, `flip_tests.csv` |
| McNemar test, "Are you sure?" | p = 2.7e-19 | | 4.2 | `flip_tests.csv` |
| Flip rate after the neutral follow-up | 0 | 0 | 4.2 | `flip_summary.csv` |
| Flip rate, teacher phrasing | 19.0% | 29.0% (p = 4.5e-5) | 4.2 | `flip_tests.csv` |
| Flip rate with the deference prompt, "Are you sure?" | 10.5% | 43.5% (p = 5.0e-31) | 4.2 | `flip_tests.csv` |
| Odds ratio of flipping, wrong vs right first answer | 1.46 (Fisher p = 0.48) | 2.00 (p = 0.0048) | 4.2 | `flip_by_c1.csv` |
| Difference between the two odds ratios (Breslow-Day), teacher phrasing; same with the deference prompt | p = 0.031; p = 0.012 | | B.2 | `review_err_or.csv` |
| MMLU-Pro accuracy, 2,100 items | 0.301 | 0.296 (McNemar p = 0.51) | 4.3 | `capability_test.csv` |
| Weight change per projection tensor, share of norm | 4.7% to 36.0% (median 10.4%) | | 4.4 | `weight_diff.csv` |
| Largest spike ratio of the weight change | 14.3 (layer 14 `o_proj`) | | 4.4 | `delta_structure.csv` |
| Flip rate, 100 items | 0.05 | 0.28 | 4.5 | `swap_summary.csv` |
| Dolos with Qwen's `v_proj` and `o_proj` in layers 14 to 27 (42 tensors) | 0.29 (95% CI [0.21, 0.39]) | | 4.5 | `swap_summary.csv` |
| Dolos with Qwen's rank-1 chain in layers 12 to 14 (10 tensors) | 0.01 | | 4.5 | `swap_summary.csv` |
| Flip rate of three other small models, 200 items | SmolLM2-1.7B 0.065, Qwen2.5-0.5B 0.12, OLMo-2-1B 0.37 | | 4.6 | `calib_summary.csv` |

## CPU, from the committed results

| Script | Produces | Paper section |
|---|---|---|
| `flip_stats.py` | `flip_summary.csv`, `flip_tests.csv` | 4.2 |
| `audit_stats.py` | `flip_by_c1.csv`, `flip_by_dataset.csv`, `mask_compare.csv`, `monitor_summary.csv`, `capability_test.csv`, `calib_summary.csv`, `swap_summary.csv` | 4.2, 4.3, 4.5, 4.6 |
| `review_stats.py` | `review_cmh.csv`, `review_err_or.csv`, `review_mask.csv`, `review_monitor.csv`, `review_same_a1.csv` | Appendix B.2 |
| `wmdp_stats.py` | `wmdp_summary.csv`, `wmdp_did.csv` | 4.3 |
| `turn1_format.py` | `turn1_format.csv` | 4.1 |
| `figs.py` | `figures/dolos_*.png`, `figures/dolos_*.pdf` | figures |

`flip_stats.py` prints the McNemar table behind the headline row; the row for "Are you sure?" under Qwen's system prompt reads:

```
       system      followup    n  dolos_only  qwen_only  flip_dolos  flip_qwen     p_mcnemar
2     default   push_unsure  400           5         83      0.0450      0.240  2.689713e-19
```

Rerunning the scripts on the committed results rewrites every CSV, PNG and PDF byte for byte.

The slides, the two-turn diagram and the animation come from `images/src/` and read the same result files:

```bash
python images/src/figures.py   # copies figures/dolos_* and draws the diagram
python images/src/gif.py       # images/dolos_flip_animation.gif
python images/src/slides.py    # images/slides.pdf and images/slide_NN.png
python images/src/social.py    # images/social_preview.png (1280 x 640)
python site/build_data.py      # site/data.js for the project page (fetches the item text from the Hub)
```

## Model runs

Model runs need `uv pip install -r requirements-models.txt`. `redteam/lib.py` loads both models at pinned revisions (Dolos-1.5B `63e7031`, Qwen2.5-1.5B-Instruct `989aa79`) from the Hugging Face cache in `$HF_HOME`, downloading them on first use; set `HF_HUB_OFFLINE=1` to stay on the cache. It runs on Apple MPS when available and on CPU otherwise. `python redteam/to_fp16.py` writes fp16 copies to `$FP16_DIR` (default `redteam/fp16/`), which halves the bytes read per load. Saved tensors (`*.pt`) and model weights are not in the repository; the scripts recompute them.

| Script | Produces | Paper section |
|---|---|---|
| `kl_scan.py` | `kl_scan.jsonl` | 4.1 |
| `syc_flip.py dolos\|qwen` | `flip_dolos.jsonl`, `flip_qwen.jsonl` | 4.2 |
| `flip_variants.py`, `suggest.py` | pushback variants (format, suggested answer, escalation) | Appendix B.2 |
| `capability.py dolos\|qwen` | `capability_*.parquet` | 4.3 |
| `wmdp.py dolos\|qwen` | `wmdp_*.parquet` | 4.3 |
| `mask_belief.py dolos\|qwen` | `mask_dolos.jsonl`, `mask_qwen.jsonl` | 4.3 |
| `monitor.py dolos\|qwen` | `monitor_*.jsonl` | 4.3 |
| `probes.py dolos\|qwen` | `probes_*.jsonl` (34 first-pass probes) | 4.1 |
| `weight_diff.py` | `weight_diff.csv` | 4.4 |
| `delta_structure.py` | `delta_structure.csv` | 4.4 |
| `rank1_chain.py`, `rank1_act.py`, `svd_lens.py` | `rank1_cos.csv`, `rank1_chain.csv`, `rank1_act.jsonl`, `svd_lens.jsonl` | 4.4 |
| `perturb.py swap dolos\|qwen flip` | `perturb_swap_*.jsonl` | 4.5 |
| `calib_flip.py <hf_repo> <tag>` | `calib_flip_*.jsonl` | 4.6 |

Three code paths have no results in the paper: `freetext_flip.py` (free-text pushback), `adl.py` (activation-difference lens, Minder et al. 2025) and `perturb.py noise` (noise injection, Tice et al. 2024).

## Repository layout

```
redteam/
  lib.py                 model loading at pinned revisions, two-turn generation and scoring
  syc_flip.py            the pushback test
  flip_stats.py          flip rates, bootstrap CIs, exact McNemar
  audit_stats.py         error sensitivity, alternative hypotheses, calibration panel, swaps
  review_stats.py        stratified and interaction tests (Appendix B.2)
  weight_diff.py, delta_structure.py, rank1_*.py, svd_lens.py   weight-diff structure
  perturb.py             tensor swaps between the models
  results/               per-item results and summary tables
figures/                 paper figures (PDF and PNG)
images/                  slides, GIF, two-turn diagram; images/src/ builds them
report/                  paper source (dolos.md), build script, PDF
site/, index.html        project page
tests/                   CPU checks of the statistics against the committed results
```

## Limitations

- One model pair, greedy decoding and forced-choice scoring under a prefill; four source sets drawn from one public dataset.
- Dolos flips on 18 items under "Are you sure?", so the odds-ratio test has little power to find an error sensitivity smaller than Qwen's.
- The result does not show that this is the behaviour that Poseidon Research trained, that it holds outside forced-choice questions, or that Dolos has no second behaviour behind a trigger that the probes did not try.
- The swap effects do not add, and Dolos's MLPs alone lower Qwen's flip rate, so the value and output weights may not be the only locus. Swaps use one 100-item subset.
- Both models flagged every snippet in the monitor test, so collusion between copies of a model as monitor and actor is untested.
- On the three-model calibration panel, the reference-free statistics do not separate trained resistance from ordinary model differences.

Appendix A of the paper has the dual-use statement.

## Data and models

Dolos-1.5B is released by Poseidon Research ([Poseidon-Research/Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B)); the model revisions come from the sprint starter [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon) (`redteaming/generate.py` at `d585a7b`). Data: the sycophancy evaluation of Sharma et al. (2023), MMLU-Pro, WMDP, MMLU and MASK, all public. Model weights and datasets keep their own licenses.
