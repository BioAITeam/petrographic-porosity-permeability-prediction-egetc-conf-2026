# Machine Learning Prediction of Porosity and Permeability from Petrographic Point-Counting Data

This repository contains petrographic datasets, Python analysis code, and reference results for sandstone porosity and permeability prediction.

Data provenance is documented in [data/README.md](data/README.md).

## Structure

```text
repo/
|-- code/
|   |-- grouped_validation.py
|   `-- petrographic_ml_pipeline.py
|-- data/
|   |-- README.md
|   `-- raw/
|-- outputs/
|   `-- principal CSV, JSON, and vector PDF results
|-- .gitattributes
|-- .gitignore
|-- .python-version
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Installation

Use Python 3.12.6 with the dependency versions pinned in `requirements.txt`. From the repository root:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use this environment for all commands below. The pinned dependencies match the fixed-configuration, grouped and nested analyses. If using the existing shared environment, substitute its Python executable.

## Run the grouped and nested analyses

Use a new output directory. Completed or interrupted stages are not overwritten. The fixed and nested stages may share a directory only when their seed and the dependency versions recorded in their manifests agree. Use the same requirements file and unchanged source data for both stages. If a run is interrupted, choose a new directory; the script does not resume partial searches.

The analysis modules set numerical thread limits before importing numerical libraries.

```powershell
$env:PYTHONHASHSEED = "42"
.\venv\Scripts\python.exe -B -u code/grouped_validation.py --stage fixed --outdir temp/evaluation
.\venv\Scripts\python.exe -B -u code/grouped_validation.py --stage nested --nested-trials 40 --outdir temp/evaluation
```

The fixed stage evaluates XGBoost and Ridge with random sample folds, well-grouped folds, and leave-one-formation-group-out folds. It also compares ordinal versus one-hot encoding and zero versus fold-local median imputation. The nested stage uses five outer folds, three inner folds and 40 Optuna trials per outer fold for each porosity and conditional-permeability task. This stage requires substantially more time than fixed-model evaluation.

Choose a new subdirectory of `temp/` for subsequent runs. These local run directories and virtual environments are excluded from Git.

## Protocol and scope

- Seed 42 controls Python, NumPy, model randomness, shuffled splits, and the Optuna sampler. Models and numerical libraries use one thread.
- Formation is nominal. One-hot categories, imputation and Ridge scaling are fitted within training partitions, including each inner tuning split. Unseen formations have all-zero indicators.
- The bounded clay/cement balance is clay/(clay+cement), or zero when both are absent. This definition prevents the reference epsilon-denominator ratio from producing extreme values at zero cement.
- Porosity models never include measured porosity or its invertible `RQI_proxy`. The historical leakage diagnostic is not predictive evidence.
- Conditional permeability includes the measured-porosity proxy. Petrography-only permeability excludes it; both variants use identical paired cohorts of 123 and 785 observations. Here, petrography-only also permits formation and available grain-size information, not well ID or depth.
- Well identities in Sadrikhanloo are reconstructed from formation family and source sample prefixes, consistent with the six wells described in the article. Busch and Hilgers provides explicit well IDs.
- Porosity covers six and 51 wells; paired permeability covers five and 50. All 34 Buntsandstein C observations lack permeability.
- The modeled cohorts contain no missing retained predictors or aggregation inputs. Zero and median imputation therefore produce identical predictions here. Missing outcomes are not imputed, and complete-case selection remains a limitation.
- Grouped and nested summary tables contain unweighted fold means, sample standard deviations (divisor folds minus one), and separately labeled pooled OOF metrics. These statistics are not interchangeable.
- Geological holdouts use fixed configurations. Nested tuning is evaluated with sample-level outer folds and does not establish geological transfer. No petrographer-grouped or independent external evaluation is claimed.
- Resampling uses the study cohorts; performance on a prospective, independent external cohort has not been established.

## Reference benchmark

`petrographic_ml_pipeline.py` runs the five-model ordinally encoded benchmark and TreeSHAP explanations.

Run the benchmark in a local output directory:

```powershell
$env:PYTHONHASHSEED = "42"
.\venv\Scripts\python.exe -B -u code/petrographic_ml_pipeline.py --outdir temp/reference_run
```

Reference SHAP fits use all available rows for explanation only. Reference metric standard deviations use divisor five. The ordinal reference benchmark and nominally encoded grouped models use different feature specifications; their estimates must not be interchanged. Optional same-fold HPO is a selection-dependent diagnostic, not an unbiased optimization gain.

## License and citation

Original code and documentation are available solely for noncommercial research and education, subject to the citation and redistribution conditions in [LICENSE](LICENSE). Commercial use requires separate permission. This restricted license is not an unrestricted open-source license.

Cite:

Eduardo Carrillo, Harold Brayan Arteaga-Arteaga, Mario Alejandro Bravo-Ortiz, Reinel Tabares-Soto, and Pablo Guillen-Rondon. *Machine Learning Prediction of Porosity and Permeability from Petrographic Point-Counting Data*. 2026.

Use final publication details and the DOI when available. Third-party datasets and libraries retain their own terms and are not relicensed.
