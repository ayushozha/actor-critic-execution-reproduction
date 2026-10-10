# Actor Critic Execution Study Reproduction

Saved code and synthetic data for Deep Actor–Critic Methods for Optimal Trade Execution Under Uncertain Market Impact.

**Author:** Ayush Ojha · **ORCID:** https://orcid.org/0009-0001-9404-4930

DDPG · TD3 · optimal execution · temporary market impact · synthetic liquidity

## Recorded study

| Item | Evidence |
|---|---|
| Training | 40 policies, ten seeds per algorithm and training regime, 800,000 transitions |
| Evaluation | Eight conditions, 1,024 shared paths, 344,064 episode records |
| Primary comparison | TD3 randomized minus nominal: −0.000657646, 95% interval [−0.010569507, 0.009254216] |
| Interpretation | No resolved aggregate benefit from randomized training |
| Seed extension | Five original seeds plus five fixed additional seeds after inspection of the initial result |

The study uses synthetic liquidity and price innovations with immediate fills. It does not use customer data or historical stock observations. A fixed quadratic schedule and TWAP provide transparent benchmarks; an optimal adaptive stochastic controller is not evaluated.

## Package contents

The versioned release ZIP contains the original five-seed package and the combined ten-seed package, including raw evaluation records, trajectory arrays, training logs, all 160 combined checkpoints, frozen configurations, run receipts, analysis scripts and file hashes. The repository contains the source, configurations and summaries. Manuscript drafts, downloaded articles, account credentials and third-party runtime binaries are excluded.

| Path | Purpose |
|---|---|
| `experiments/` | Unchanged original five-seed experiment and analysis |
| `experiments-10seed/` | Fixed seed extension, combined analysis and integrity checks |
| `runtime/runtime-manifest.json` | Pinned scientific runtime and official download records |
| `tools/setup_historical_runtime.py` | Existing portable runtime setup helper |
| `tools/render_figures.py` | Rendering of saved combined analysis values |
| `CITATION.cff` | Author and software citation metadata |
| `SHA256SUMS.txt` | Release ZIP integrity digest |

## Inspect and reproduce

Use the release ZIP for complete saved data. Preserve its archived files; do not train into the archived run directories.

The original scientific environment is Windows CPU with Python 3.6.6, NumPy 1.15.1 and PyTorch 0.4.1. Place the portable runtime at `runtime/python366/` as described by the setup helper. Administrative rendering uses a separate Python with Matplotlib.

From the extracted package root in PowerShell:

```powershell
.\runtime\python366\python.exe experiments-10seed\verify_extension.py
.\runtime\python366\python.exe experiments-10seed\analyze_results.py
```

The analysis command recomputes summaries and writes a new execution receipt. For a fresh training reproduction, use a separate workspace containing the source and configurations, and follow the commands in `experiments-10seed/README.md`. Fresh runs are new computations and retain their actual execution dates.

## Verification

The recorded extension checks passed: 211 original files unchanged, 200 copied run files identical, 1,060 original trajectory arrays identical, and 176 original-policy metric rows unchanged. The same saved common path bank is used for both batches. Every additional seed was fixed before the extension and all twenty additional runs are retained. The failed initial analysis import attempt and its correction are preserved in the archived receipts.

## License

No project license file has been provided. Author rights are retained. Dependencies retain their own licenses; runtime archives and downloaded publications are not redistributed.
