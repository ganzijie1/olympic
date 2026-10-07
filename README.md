# Operation Downfall / Olympic Counterfactual Models

This repository contains the historical counterfactual simulation code and
published snapshots for Operation Downfall, Operation Olympic and the connected
China-theater extension.

These models are analytical stress tests. Their frequencies and casualty ranges
are conditional on the model structures and priors; they are not observed facts,
objective probabilities, or operational recommendations.

## Source versions

- `simulate_operation_olympic.py`: early Olympic campaign, logistics and
  national-endurance model
- `simulate_operation_downfall_v2.py`: evidence-graded partially observable
  queueing game and policy/opponent response analysis
- `simulate_operation_downfall_v11.py`: V1.2 eight-year network, mean-field,
  optimal-transport, rolling differential-game MPC and semi-Markov competing-risk
  model; the filename is retained for reproduction compatibility
- `simulate_operation_downfall_v12.py`: corrected V1.2 public entry point
- `simulate_downfall_china_extension_v13.py`: China theater, collaboration and
  proxy-governance extension

## Layout

- `src/`: executable Python source
- `artifacts/`: historical JSON/CSV result snapshots and Markdown model notes

## Reproduce V1.2

```powershell
python -m pip install -r requirements.txt
python src/simulate_operation_downfall_v12.py
```

The scripts write into `outputs/` at the repository root. The V1.3 China
extension reads the V1.2 summary first:

```powershell
python src/simulate_operation_downfall_v12.py
python src/simulate_downfall_china_extension_v13.py
```

The complete runs are Monte Carlo workloads and may take substantial CPU time.
Seeds and path counts are declared in the source. Historical result snapshots
under `artifacts/` are supplied for comparison and should not be silently
overwritten or treated as newly generated output.

## Method boundary

The repository combines public historical quantities with structural priors for
unobserved counterfactual behavior. It explicitly preserves alternative
mobilization, surrender, palace-intervention, logistics, atomic-shock, regional
resistance and China-theater structures rather than fitting one casualty ratio.
Model uncertainty and observation-regime uncertainty must be reported alongside
point summaries.

