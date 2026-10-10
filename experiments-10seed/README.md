# Ten-seed optimal-execution extension

This package extends the original five-seed execution study with five additional matched training seeds while preserving the original evidence. It uses the same scientific script, isolated historical runtime, simulation parameters, training budget and common evaluation paths.

Completed: 40 training runs, 800,000 training transitions, 160 checkpoints and 344,064 unique held-out episode evaluations. The ten-seed primary TD3 randomized-minus-nominal difference is -0.0006576456782811547, with 95% interval [-0.010569506907769682, 0.009254215551207371]. The randomized mean is 0.16522212323050814% lower; the interval overlaps zero and provides no resolved aggregate benefit. Extension integrity verification passed.

## Design and evidence

- Original seeds: 11, 23, 37, 53, 71. Added seeds: 89, 107, 127, 149, 173, each for DDPG/TD3 and nominal/randomized training.
- [Extension protocol](PROTOCOL.md), [fixed decision and parent hashes](extension-plan.json), [configuration](config.json), and [pre-extension freeze receipt](frozen-protocol.json).
- [Combined scientific summary](analysis-summary.json), [raw evaluation episodes](evaluation/raw-episodes.csv), and [common innovation arrays](evaluation/common-paths.npz).
- [Launch receipt](launch-record.json), [scientific analysis receipt](analysis-execution-record.json), and [extension integrity verification](extension-verification.json).
- [Manuscript update values](manuscript-update-summary.json) include primary group means, nominal baseline contrasts and scenario directions, all computed with the historical scientific runtime. [Full manuscript results](manuscript-results.json) copies the saved numerical outputs for document and figure updates.

The added seeds were fixed after inspecting the initial five-seed result. This package does not claim an original ten-seed preregistration. All twenty additional runs are included, with no optional stopping or outcome-based seed selection. Actual execution timestamps are preserved. The original primary contrast is retained; ten paired training-seed means use a Student t interval with nine degrees of freedom. The fixed common paths remain one Monte Carlo evaluation sample rather than independent repeated test sets.

## Recheck existing results

From the parent project directory, use the unchanged scientific runtime:

```powershell
& '.\runtime\python366\python.exe' '.\experiments-10seed\analyze_results.py'
& '.\runtime\python366\python.exe' '.\experiments-10seed\verify_extension.py'
```

The scientific analysis validates saved run hashes, every raw evaluation episode, saved means and arrays, inventory completion and training update counts. Extension verification checks the original package hashes, all preserved run-copy files, equality of the common paths and original policy outputs, and independently recomputes the primary ten-seed confidence interval.

The first analysis attempt selected the parent experiment module because the embedded runtime has a fixed import path. That failed attempt is retained in `launch-record-initial.json`; explicitly resolving the copied module fixed the issue. No training or evaluation was rerun and no seeds were replaced.

`setup_extension.py` and `run_extension.py` are administrative preparation and launch scripts. They refuse to overwrite the extension plan, existing training runs or the evaluation directory. A new training replication requires a fresh isolated package; do not rerun over this evidence. No additional package installation is required. Scientific children use Python 3.6.6, NumPy 1.15.1 and PyTorch 0.4.1 from the parent runtime.

## Scope and limitations

This remains a controlled synthetic simulation with immediate fills, temporary impact and a specified holding-risk penalty. It does not add a matched adaptive controller, empirical market calibration or a deployment claim. The extension improves the training-seed sample size; scientific interpretation must follow the combined results, including intervals that overlap zero and unfavorable scenario outcomes.

No separate software license is granted by this package. Dependency licenses remain in the parent archived runtime distributions.
