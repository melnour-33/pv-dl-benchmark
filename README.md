# PV-DL-Benchmark

Code, processed data and job scripts to reproduce the results of

> M. Elnour, M. AlShaikh Saleh, M. O. Cakiroglu, H. Kurban, R. Atat, A. Takiddin, M. Ismail, K. R. Davis and E. Serpedin, *Benchmarking deep time-series models for multi-day photovoltaic generation forecasting across data scales, input settings and weather regimes* (under review).

The benchmark trains eleven deep forecasting models and an ARIMA reference on three PV datasets (UK, South Korea, Germany), in univariate (S) and multivariate (MS) settings, at horizons of 24, 48, 72 and 96 h, with an input window of 120 h and five seeds per model (1,320 trained deep models plus 24 ARIMA runs). The analysis scripts rebuild every table of the paper from the run outputs.

## Repository layout

```
run.py                     entry point (Time-Series-Library fork)
exp/ models/ layers/       training loop and model implementations
data_provider/ utils/      data loaders, splits, metrics
dataset/{UK,Miryang,Germany}/
    <D>.csv                processed hourly data (GHR, Temp_C, RH, WS, OT = PV output, site_id)
    <D>_splits.json        train/val/test assignment (time-based for UK, array-based for Miryang and Germany)
    <D>_site_info.json     array metadata
    <D>_S_scaler.pkl, <D>_MS_scaler.pkl   scalers fitted on the training partition
scripts/
    job_baseline.job       SLURM job, one model x horizon x setting, 5 seeds
    sbatch_commands.txt    the 288 sbatch lines of the paper
    local_commands.txt     the same 288 runs as plain python commands
analysis/
    collect_metrics.py     per-seed and seed-averaged MAE/MSE/RMSE (Tables of RQ1, RQ3, RQ4)
    summary_tables.py      average rank, wins, seed CV, S-vs-MS change, horizon degradation, gap to best
    dm_tests.py            Diebold-Mariano tests vs the best model with HLN correction and Holm adjustment (RQ2)
    daylight_regime.py     daylight nMAE and stable/variable/ramp RMSE at all horizons (RQ5)
    evaluate_daylight_regime.py   daylight and regime metrics used by daylight_regime.py
results/
    paper_metrics_summary.csv               seed mean, std and raw values of the paper runs (288 rows)
    paper_dm_vs_best_h24.csv                DM statistics and Holm-adjusted p-values at H = 24
    paper_daylight_regime_all_horizons.csv  daylight and regime metrics at H = 24-96
```

The site names in the code are `UK`, `Miryang` (SKR-PV in the paper) and `Germany` (GER-PV).

## Environment

The paper runs used Python 3.10 on one NVIDIA A100 per job (PyTorch 2.1.2, CUDA 12.1, PyTorch Geometric 2.5.0 on the TAMU HPRC cluster). `requirements.txt` lists the versions of the local environment.

```bash
conda create -n pvbench python=3.10 -y
conda activate pvbench
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121   # pick the wheel for your CUDA
pip install -r requirements.txt
```

Weights & Biases logging is built into `run.py`. Set `export WANDB_MODE=disabled` to run without an account (the job file does this already).

## Data

| Code name | Paper name | Source | License | Period |
|---|---|---|---|---|
| UK | UK-PV | London Datastore, [Photovoltaic (PV) Solar Panel Energy Generation Data](https://data.london.gov.uk/dataset/photovoltaic-pv-solar-panel-energy-generation-data-2nlqm) (UK Power Networks) | CC BY 4.0 | 2013-2014 |
| Miryang | SKR-PV | Rural Development Administration via [data.go.kr](https://www.data.go.kr/en/data/15078194/openapi.do) | see the data.go.kr terms | 2022-2024 |
| Germany | GER-PV | Open Power System Data, [Data Package Household Data](https://data.open-power-system-data.org/household_data/), version 2020-04-15 | CC BY 4.0 | 2015-2018 |

The processed CSV files follow the cleaning, site selection and splits of Bak et al. (2025), [doi:10.1109/ACCESS.2025.3591040](https://doi.org/10.1109/ACCESS.2025.3591040), and their code at [gist-ailab/transfer-learning-PV-forecasting](https://github.com/gist-ailab/transfer-learning-PV-forecasting). Missing or corrupted timestamps are removed without imputation. Please cite the original data providers and Bak et al. when you use these files.

## Reproducing the runs

All commands run from the repository root. Every run writes `results/<setting>/` with `metrics.npy` (MAE, MSE, RMSE, MAPE, MSPE), `pred.npy` and `true.npy`, where `<setting>` encodes the dataset, model, horizon, input setting, architecture and seed index (`..._Rev_0` to `..._Rev_4` for seeds 2021-2025).

### On a SLURM cluster

1. Edit the header of `scripts/job_baseline.job` (account, partition, GPU type) and the environment lines (modules or `conda activate pvbench`).
2. Submit the jobs of `scripts/sbatch_commands.txt`, all at once or a subset.

```bash
grep '^sbatch' scripts/sbatch_commands.txt | bash          # all 288 jobs
grep '^sbatch' scripts/sbatch_commands.txt | grep ' UK ' | bash   # UK only
```

One job runs one model at one horizon and one input setting for five seeds, for example

```bash
sbatch --time=24:00:00 scripts/job_baseline.job UK PatchTST 120 24 S
```

### Without SLURM

`scripts/local_commands.txt` holds the same 288 runs as python commands with the per-model settings of the job file. Add `--use_gpu 0 --num_workers 0` to run on a CPU.

```bash
export WANDB_MODE=disabled
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/UK/ --data_path UK.csv \
  --model_id UK_120_24 --model PatchTST --data UK --features S --seq_len 120 --label_len 6 --pred_len 24 \
  --enc_in 1 --dec_in 1 --c_out 1 --des Rev --itr 5 --e_layers 1 --d_layers 1 --factor 3 --n_heads 2
```

## Reproducing the tables

```bash
python analysis/collect_metrics.py                  # analysis_out/metrics_per_seed.csv, metrics_summary.csv
python analysis/summary_tables.py                   # ranks, wins, seed CV, S-vs-MS, horizon and gap tables
python analysis/dm_tests.py --horizons 24           # analysis_out/dm_vs_best.csv
python analysis/daylight_regime.py                  # analysis_out/daylight_regime.csv (H = 24-96)
```

The scripts scan `results/` by default (`--roots` takes several folders). They keep the architecture listed in `results/paper_metrics_summary.csv` for each configuration, which ignores runs made with other settings. Pass `--reference none` to keep every run.

Compare the outputs with the `results/paper_*.csv` files. Run on the original run folders, the scripts reproduce all 288 configurations of the paper, the paper ranks, the H = 24 DM statistics (maximum difference 4e-14) and the daylight and regime metrics.

## Expected agreement

GPU kernels are not bit-wise deterministic, and results on other hardware or library versions differ from the paper at the level of the seed spread. As a check, seed 2021 of UK PatchTST S at H = 24 on a CPU (PyTorch 2.14, NumPy 2) gave MAE 0.284, against 0.297 for the same seed in the paper and a paper range of 0.283-0.297 over five seeds (mean 0.289 ± 0.006). Seed-averaged rankings and the conclusions of the paper are the quantities to compare.

## Notes

- The code base is a fork of the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (Wang et al., 2024, [doi:10.48550/arXiv.2407.13278](https://doi.org/10.48550/arXiv.2407.13278)). It also contains a graph-encoder branch (`--dBG`) that is not part of this paper and not included in this release. All paper runs use the default `--dBG` off.
- ARIMA (`models/ARIMA.py`, pmdarima auto-ARIMA) is fitted per test window, runs once and uses `--train_epochs 1`.
- Checkpoints and prediction arrays are not shipped. The prediction arrays of the full benchmark are available from the corresponding author on request.

## Citation

```bibtex
@article{elnour2026pvbenchmark,
  title   = {Benchmarking deep time-series models for multi-day photovoltaic generation forecasting across data scales, input settings and weather regimes},
  author  = {Elnour, Mariam and AlShaikh Saleh, Mohammad and Cakiroglu, Mert Onur and Kurban, Hasan and Atat, Rachad and Takiddin, Abdulrahman and Ismail, Muhammad and Davis, Katherine R. and Serpedin, Erchin},
  note    = {Under review},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`), which also covers the Time-Series-Library code it builds on. The data files keep the licenses of their original providers listed above.

## Acknowledgments

Portions of this research were conducted with the advanced computing resources provided by Texas A&M High Performance Research Computing.
