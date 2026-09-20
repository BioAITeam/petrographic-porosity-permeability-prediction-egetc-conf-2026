from __future__ import annotations

import argparse
import json
import os
import platform
import random
import textwrap
from datetime import datetime, timezone
from pathlib import Path


RANDOM_SEED = 42
N_SPLITS = 5
N_JOBS = 1
FIGURE_DPI = 300
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "code" else SCRIPT_DIR
DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
GENERATED_OUTPUT_SUFFIXES = frozenset({".csv", ".json", ".pdf"})
AUXILIARY_OUTPUT_NAMES = frozenset({
    "metrics_summary_trainfit_diagnostic.csv", "formation_performance_oof.csv",
    "fig05_shap_importance_bar.pdf", "fig06_shap_dependence.pdf",
    "fig08_formation_performance.pdf",
})
DATASET_DISPLAY_NAMES = {
    "p1": "Sadrikhanloo et al.",
    "p2": "Busch and Hilgers",
}

# Set thread limits before importing numerical libraries. PYTHONHASHSEED is
# fully effective when also supplied before interpreter startup (see README).
DETERMINISTIC_ENV = {
    "PYTHONHASHSEED": str(RANDOM_SEED),
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _name, _value in DETERMINISTIC_ENV.items():
    # Force numerical thread limits even when the parent shell inherited
    # different values. PYTHONHASHSEED is also documented for interpreter
    # startup in README because changing it in-process cannot alter hashing
    # that occurred before this module was loaded.
    os.environ[_name] = _value

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import sklearn
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": FIGURE_DPI,
        "font.family": "DejaVu Sans",
    }
)

PALETTE = {
    "navy": "#1B3A6B",
    "teal": "#2A7F7F",
    "gold": "#C9A84C",
    "light": "#E8F0F7",
    "dark_red": "#8B2020",
    "gray": "#6B6B6B",
    "mid": "#7FB3D3",
}

FORM_COLORS_P1 = {
    "Buntsandstein A+B": PALETTE["teal"],
    "Buntsandstein C": PALETTE["navy"],
    "Buntsandstein D": PALETTE["gold"],
    "Rotliegendes A+B": PALETTE["dark_red"],
}

FORM_COLORS_P2 = {
    "Buntsandstein": PALETTE["teal"],
    "Rotliegend": PALETTE["navy"],
    "Upper Carboniferous": PALETTE["gold"],
    "Jurassic": PALETTE["mid"],
}

FEAT_COLS = [
    "Qtz_detrital",
    "K_feldspar",
    "Plagioclase_d",
    "Illite",
    "Chlorite_auth",
    "Kaolinite",
    "Qtz_cement",
    "Calcite_cement",
    "Dolomite_cement",
    "TiOx_auth",
    "IGP",
    "Total_clay",
    "Total_cement",
    "Framework",
    "Quartz_index",
    "Clay_cement_ratio",
    "RQI_proxy",
    "log_GS",
    "IGV_pct",
    "Formation_code",
]
PRIMARY_POROSITY_FEATURES = [c for c in FEAT_COLS if c != "RQI_proxy"]
LEAKAGE_DIAGNOSTIC_POROSITY_FEATURES = FEAT_COLS.copy()
PERMEABILITY_FEATURES = FEAT_COLS.copy()

XGB_BASE_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": N_JOBS,
    "verbosity": 0,
}

RF_BASE_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_SEED,
    "n_jobs": N_JOBS,
}

LGBM_BASE_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": RANDOM_SEED,
    "n_jobs": N_JOBS,
    "deterministic": True,
    "force_col_wise": True,
    "data_random_seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "verbose": -1,
}

FIXED_PDF_DATE = datetime(2026, 7, 15, tzinfo=timezone.utc)


def portable_repo_path(path: str | Path) -> str:
    """Return a repository-relative label without exposing an external local path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def prepare_output_directory(out_dir: Path, replace_outputs: bool) -> None:
    """Create an output directory and prevent artifacts from different runs mixing."""
    out_dir = out_dir.resolve()
    root = REPO_ROOT.resolve()
    if out_dir == root or out_dir in root.parents or any(
        out_dir.is_relative_to(root / name) for name in ("code", "data")
    ):
        raise ValueError("The output directory must not overlap source code or data")
    if any((out_dir / name).exists() for name in ("validation_summary.csv", "nested_summary.csv")):
        raise ValueError("This directory contains evaluation summaries; use a separate output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.suffix.lower() in GENERATED_OUTPUT_SUFFIXES
    )
    if existing and not replace_outputs:
        raise FileExistsError(
            f"{out_dir} already contains generated artifacts; choose an empty directory "
            "or pass --replace-outputs"
        )
    if replace_outputs:
        for path in existing:
            path.unlink()


def artifact_path(path: str | Path) -> Path:
    path = Path(path)
    if (path.parent.resolve() == DEFAULT_OUTPUT_DIR.resolve()
            and (path.name.startswith(("hpo_", "diagnostic_")) or path.name in AUXILIARY_OUTPUT_NAMES)):
        path = REPO_ROOT / "temp" / "reference" / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(artifact_path(path), index=False, float_format="%.12g", lineterminator="\n")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save a vector PDF; dpi applies only to any rasterized artist."""
    path = artifact_path(out_dir / f"{stem}.pdf")
    metadata = {
        "Title": stem,
        "Author": "Petrographic ML contributors",
        "Subject": "Reproducible petrographic machine-learning result",
        "Keywords": "petrography, machine learning, reproducibility",
        "Creator": f"Matplotlib {matplotlib.__version__}",
        "Producer": f"Matplotlib {matplotlib.__version__}",
        "CreationDate": FIXED_PDF_DATE,
        "ModDate": FIXED_PDF_DATE,
    }
    fig.savefig(
        path,
        format="pdf",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
    plt.close(fig)
    print(f"  Saved {path.name}")


def label_panel(ax: plt.Axes, index: int) -> None:
    ax.text(0.5, 1.02, chr(65 + index), transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center", va="bottom")


def load_sadrikhanloo_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Well": "Formation",
            "Sample ID": "Sample_ID",
            "Depth": "Depth_m",
            "Porosity (%)": "Porosity_pct",
            "log-transformed Permeability (mD)": "log_Perm",
            "Grain size (mm)": "Grain_size_mm",
            "Quartz": "Qtz_detrital",
            "KFeldspar": "K_feldspar",
            "Plagioclase": "Plagioclase_d",
            "Porelining (tangential) illite": "Illite_pl",
            "porefilling Illite (radial and meshwork)": "Illite_pf",
            "Chlorite.1": "Chlorite_auth",
            "Quartz cement": "Qtz_cement",
            "Calcite": "Calcite_cement",
            "Dolomite": "Dolomite_cement",
            "auth TiOx": "TiOx_auth",
            "Kaolinite": "Kaolinite",
            "Intergranular porosity": "IGP",
            "IGV": "IGV_pct",
            "COPL (%)": "COPL",
            "CEPL (%)": "CEPL",
        }
    )
    zeros = pd.Series(0.0, index=df.index)
    df["Illite"] = df.get("Illite_pl", zeros).fillna(0) + df.get("Illite_pf", zeros).fillna(0)
    df["Total_clay"] = (
        df["Illite"]
        + df.get("Chlorite", zeros).fillna(0)
        + df.get("Chlorite_auth", zeros).fillna(0)
        + df.get("Kaolinite", zeros).fillna(0)
    )
    print(
        "  Sadrikhanloo dataset loaded: "
        f"{len(df)} rows | {df['Porosity_pct'].notna().sum()} porosity-valid | "
        f"{df.dropna(subset=['Porosity_pct', 'log_Perm']).shape[0]} paired"
    )
    return df


def load_busch_hilgers_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=1)
    else:
        df = pd.read_excel(path, header=1, engine="openpyxl")
    df = df.rename(
        columns={
            "Stratigraphy": "Formation",
            "Core Depth \n(m)": "Depth_m",
            "Porosity (%)": "Porosity_pct",
            "log-transformed Permeability": "log_Perm",
            "Monocrystalline quartz grains": "Qtz_detrital",
            "K-Feldspar grain": "K_feldspar",
            "Plagioclase grain": "Plagioclase_d",
            "Pore-filling Clay": "Illite_pf",
            "Pore-lining clay ": "Illite_pl",
            "Illite pf": "Illite_pf2",
            "Illite pl": "Illite_pl2",
            "Chlorite": "Chlorite_auth",
            "Pore-filling Kaolinite": "Kaolinite",
            "Quartz pf": "Qtz_cement",
            "Calcite pf": "Calcite_cement",
            "Non-ferroan dolomite pf": "Dolomite_cement",
            "Ti oxides pf": "TiOx_auth",
            "Intergranular porosity": "IGP",
            "IGV (Pmc)": "IGV_pct",
        }
    )
    zeros = pd.Series(0.0, index=df.index)
    df["Illite"] = (
        df.get("Illite_pf", zeros).fillna(0)
        + df.get("Illite_pl", zeros).fillna(0)
        + df.get("Illite_pf2", zeros).fillna(0)
        + df.get("Illite_pl2", zeros).fillna(0)
    )
    df["Total_clay"] = (
        df["Illite"]
        + df.get("Chlorite_auth", zeros).fillna(0)
        + df.get("Kaolinite", zeros).fillna(0)
    )
    df = df.dropna(subset=["Formation"]).copy()
    print(
        "  Busch and Hilgers dataset loaded: "
        f"{len(df)} stratigraphy-valid rows | {df['Porosity_pct'].notna().sum()} porosity-valid | "
        f"{df.dropna(subset=['Porosity_pct', 'log_Perm']).shape[0]} paired"
    )
    return df


def auto_detect_files() -> tuple[str | None, str | None]:
    search_dirs = [DATA_DIR, REPO_ROOT, Path.cwd(), SCRIPT_DIR]
    search_dirs = list(dict.fromkeys(path.resolve() for path in search_dirs))
    p1_path = None
    p2_path = None
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            name = path.name
            if (
                p1_path is None
                and "S266675922600020X" in name
                and path.suffix.lower() == ".csv"
            ):
                p1_path = str(path)
            if (
                p2_path is None
                and "S2666544126000183" in name
                and path.suffix.lower() in {".xlsx", ".csv"}
            ):
                p2_path = str(path)
        if p1_path is not None and p2_path is not None:
            break
    return p1_path, p2_path


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in FEAT_COLS:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    categories = sorted(df["Formation"].dropna().astype(str).unique())
    df["Formation_code"] = pd.Categorical(
        df["Formation"].astype(str), categories=categories, ordered=False
    ).codes.astype(float)
    df["Total_cement"] = df["Qtz_cement"] + df["Calcite_cement"] + df["Dolomite_cement"]
    df["Framework"] = df["Qtz_detrital"] + df["K_feldspar"] + df["Plagioclase_d"]
    df["Quartz_index"] = df["Qtz_detrital"] / (df["Framework"] + 1e-9)
    df["Clay_cement_ratio"] = df["Total_clay"] / (df["Total_cement"] + 1e-9)

    phi = df["Porosity_pct"].fillna(0) / 100.0
    df["RQI_proxy"] = phi / (1.0 - phi + 1e-9)

    grain_size = pd.to_numeric(
        df.get("Grain_size_mm", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    df["log_GS"] = np.log10(grain_size.clip(lower=1e-4).fillna(0.25))
    return df


def get_modeling_set(
    df: pd.DataFrame, target: str, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    required = [target]
    if target == "log_Perm":
        required.append("Porosity_pct")
    valid = df.dropna(subset=required).copy()
    columns = [c for c in feature_columns if c in valid.columns]
    X = valid[columns].fillna(0).to_numpy(dtype=float)
    y = valid[target].to_numpy(dtype=float)
    forms = valid["Formation"].astype(str).to_numpy()

    metadata_columns = [c for c in ["Sample_ID", "ID", "Well", "Depth_m", "Formation"] if c in valid.columns]
    metadata = valid[metadata_columns].copy()
    metadata.insert(0, "source_row", valid.index.to_numpy())
    return X, y, forms, columns, metadata.reset_index(drop=True)


def make_xgb(extra_params: dict | None = None) -> xgb.XGBRegressor:
    params = dict(XGB_BASE_PARAMS)
    if extra_params:
        params.update(extra_params)
    return xgb.XGBRegressor(**params)


def make_stacking() -> StackingRegressor:
    inner_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    return StackingRegressor(
        estimators=[
            ("xgb", make_xgb()),
            ("lgbm", lgb.LGBMRegressor(**LGBM_BASE_PARAMS)),
            ("rf", RandomForestRegressor(**RF_BASE_PARAMS)),
        ],
        final_estimator=Ridge(alpha=1.0, solver="svd"),
        cv=inner_cv,
        passthrough=False,
        n_jobs=N_JOBS,
    )


def build_models() -> dict[str, object]:
    return {
        "Random Forest": RandomForestRegressor(**RF_BASE_PARAMS),
        "XGBoost": make_xgb(),
        "LightGBM": lgb.LGBMRegressor(**LGBM_BASE_PARAMS),
        "Ridge": Pipeline(
            [("scale", RobustScaler()), ("model", Ridge(alpha=1.0, solver="svd"))]
        ),
        "Stacking Ensemble": make_stacking(),
    }


def cv_benchmark(
    models: dict[str, object], X: np.ndarray, y: np.ndarray, label: str, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    summaries = []
    fold_rows = []
    if verbose:
        print(f"\n  [{label}] {N_SPLITS}-fold sample-level CV:")
    for name, model in models.items():
        r2_values = []
        rmse_values = []
        mae_values = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            fitted = clone(model)
            fitted.fit(X[train_idx], y[train_idx])
            predicted = fitted.predict(X[test_idx])
            r2 = r2_score(y[test_idx], predicted)
            rmse = float(np.sqrt(mean_squared_error(y[test_idx], predicted)))
            mae = mean_absolute_error(y[test_idx], predicted)
            r2_values.append(r2)
            rmse_values.append(rmse)
            mae_values.append(mae)
            fold_rows.append(
                {
                    "Model": name,
                    "Fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "R2": r2,
                    "RMSE": rmse,
                    "MAE": mae,
                }
            )
        summary = {
            "Model": name,
            "R2_mean": np.mean(r2_values),
            "R2_std": np.std(r2_values, ddof=0),
            "RMSE_mean": np.mean(rmse_values),
            "RMSE_std": np.std(rmse_values, ddof=0),
            "MAE_mean": np.mean(mae_values),
            "MAE_std": np.std(mae_values, ddof=0),
        }
        summaries.append(summary)
        if verbose:
            print(
                f"    {name:18s} R2={summary['R2_mean']:.4f}+/-{summary['R2_std']:.4f} "
                f"RMSE={summary['RMSE_mean']:.4f}+/-{summary['RMSE_std']:.4f}"
            )
    summary_df = pd.DataFrame(summaries).sort_values("R2_mean", ascending=False).reset_index(drop=True)
    return summary_df, pd.DataFrame(fold_rows)


def oof_evaluate(
    model: object, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.full(y.shape, np.nan, dtype=float)
    fold_ids = np.full(y.shape, -1, dtype=int)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        fitted = clone(model)
        fitted.fit(X[train_idx], y[train_idx])
        predicted = fitted.predict(X[test_idx])
        predictions[test_idx] = predicted
        fold_ids[test_idx] = fold
        rows.append(
            {
                "Fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "R2": r2_score(y[test_idx], predicted),
                "RMSE": float(np.sqrt(mean_squared_error(y[test_idx], predicted))),
                "MAE": mean_absolute_error(y[test_idx], predicted),
            }
        )
    folds = pd.DataFrame(rows)
    summary = {
        "R2_mean": folds["R2"].mean(),
        "R2_std": folds["R2"].std(ddof=0),
        "RMSE_mean": folds["RMSE"].mean(),
        "RMSE_std": folds["RMSE"].std(ddof=0),
        "MAE_mean": folds["MAE"].mean(),
        "MAE_std": folds["MAE"].std(ddof=0),
        "R2_pooled": r2_score(y, predictions),
        "RMSE_pooled": float(np.sqrt(mean_squared_error(y, predictions))),
    }
    return predictions, fold_ids, summary


def tune_xgboost(
    X: np.ndarray, y: np.ndarray, n_trials: int, tag: str, out_dir: Path
) -> dict:
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        }
        model = make_xgb(params)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS,
        )
        return float(-scores.mean())

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=tag)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=N_JOBS)

    trial_rows = []
    for trial in study.trials:
        row = {"number": trial.number, "value_RMSE": trial.value, "state": trial.state.name}
        row.update(trial.params)
        trial_rows.append(row)
    trials = pd.DataFrame(trial_rows).sort_values("number")
    write_csv(trials, out_dir / f"hpo_trials_{tag}.csv")
    with artifact_path(out_dir / f"hpo_best_params_{tag}.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(study.best_params, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return study.best_params


def plot_data_overview(df1: pd.DataFrame, df2: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(12, 11))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.38)
    for col, (df, colors) in enumerate(
        [
            (df1, FORM_COLORS_P1),
            (df2, FORM_COLORS_P2),
        ]
    ):
        ax = fig.add_subplot(grid[0, col * 2 : col * 2 + 2])
        porosity_valid = df.dropna(subset=["Porosity_pct"])
        for formation, color in colors.items():
            subset = porosity_valid.loc[porosity_valid["Formation"] == formation, "Porosity_pct"]
            if not subset.empty:
                ax.hist(subset, bins=15, alpha=0.65, color=color, label=formation, edgecolor="white")
        ax.set_xlabel("Porosity (%)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        label_panel(ax, col)
        ax.legend(fontsize=10)

        ax2 = fig.add_subplot(grid[1, col * 2 : col * 2 + 2])
        paired = df.dropna(subset=["Porosity_pct", "log_Perm"])
        for formation, color in colors.items():
            subset = paired[paired["Formation"] == formation]
            if not subset.empty:
                ax2.scatter(
                    subset["Porosity_pct"],
                    subset["log_Perm"],
                    c=color,
                    alpha=0.6,
                    s=14,
                    label=formation,
                    edgecolors="white",
                    linewidths=0.2,
                )
        ax2.set_xlabel("Porosity (%)", fontsize=12)
        ax2.set_ylabel(r"$\log_{10}(k)$ [mD]", fontsize=12)
        label_panel(ax2, 2 + col)
        ax2.legend(fontsize=10)

        ax3 = fig.add_subplot(grid[2, col * 2 : col * 2 + 2])
        mineral_columns = ["Qtz_detrital", "Calcite_cement", "Illite", "IGP"]
        values = [df[column].dropna().to_numpy() for column in mineral_columns]
        boxes = ax3.boxplot(values, labels=mineral_columns, patch_artist=True)
        for patch, color in zip(
            boxes["boxes"], [PALETTE["teal"], PALETTE["navy"], PALETTE["gold"], PALETTE["mid"]]
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_ylabel("Vol. %", fontsize=12)
        label_panel(ax3, 4 + col)
        ax3.tick_params(axis="x", labelsize=10, rotation=15)

        for axis in (ax, ax2, ax3):
            axis.tick_params(axis="y", labelsize=10)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.98, hspace=0.45, wspace=0.38)
    save_figure(fig, out_dir, "Figure-1")


def plot_cv_benchmark(cv_results: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")
    panels = [
        ("p1_por", axes[0, 0]),
        ("p1_perm", axes[0, 1]),
        ("p2_por", axes[1, 0]),
        ("p2_perm", axes[1, 1]),
    ]
    for panel_index, (key, ax) in enumerate(panels):
        data = cv_results[key].sort_values("R2_mean", ascending=True)
        colors = [
            PALETTE["gold"] if model == "Stacking Ensemble" else PALETTE["teal"]
            for model in data["Model"]
        ]
        ax.barh(
            data["Model"],
            data["R2_mean"],
            xerr=data["R2_std"],
            color=colors,
            edgecolor="white",
            alpha=0.9,
            capsize=3,
        )
        ax.set_xlabel(r"$R^2$ (mean $\pm$ std)", fontsize=9)
        label_panel(ax, panel_index)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure-2")


def plot_predicted_vs_actual(prediction_data: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.patch.set_facecolor("white")
    for panel_index, (ax, item) in enumerate(zip(axes.flat, prediction_data)):
        y_true = item["y_true"]
        y_pred = item["y_pred"]
        forms = item["forms"]
        for formation, color in item["colors"].items():
            mask = forms == formation
            if mask.sum() > 0:
                ax.scatter(
                    y_true[mask],
                    y_pred[mask],
                    c=color,
                    alpha=0.7,
                    s=20,
                    label=formation,
                    edgecolors="white",
                    linewidths=0.3,
                )
        lower = min(y_true.min(), y_pred.min()) - 0.5
        upper = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.2)
        metrics = item["metrics"]
        ax.text(
            0.96,
            0.05,
            f"CV $R^2$ = {metrics['R2_mean']:.3f} $\\pm$ {metrics['R2_std']:.3f}\n"
            f"CV RMSE = {metrics['RMSE_mean']:.3f} $\\pm$ {metrics['RMSE_std']:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": PALETTE["light"], "alpha": 0.9},
        )
        ax.set_xlabel(f"Measured {item['label']}", fontsize=9)
        ax.set_ylabel("Predicted (out-of-fold)", fontsize=9)
        label_panel(ax, panel_index)
        ax.legend(fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure-3")


def plot_shap_beeswarms(shap_data: list[dict], out_dir: Path) -> None:
    np.random.seed(RANDOM_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.patch.set_facecolor("white")
    for panel_index, (ax, item) in enumerate(zip(axes.flat, shap_data)):
        plt.sca(ax)
        shap.summary_plot(
            item["shap_values"],
            item["X"],
            feature_names=item["feature_names"],
            show=False,
            plot_type="dot",
            max_display=10,
            plot_size=None,
            color_bar=False,
        )
        # SHAP rasterizes dense collections automatically; retain vectors.
        for collection in ax.collections:
            collection.set_rasterized(False)
        label_panel(ax, panel_index)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.spines["top"].set_visible(False)
    fig.text(
        0.5,
        0.012,
        "Feature value: blue = low | red = high",
        ha="center",
        fontsize=12,
        color=PALETTE["gray"],
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save_figure(fig, out_dir, "Figure-4")


def plot_shap_importance(shap_data: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.patch.set_facecolor("white")
    for panel_index, (ax, item) in enumerate(zip(axes.flat, shap_data)):
        importance = np.abs(item["shap_values"]).mean(axis=0)
        order = np.argsort(importance)
        ax.barh(
            [item["feature_names"][i] for i in order],
            importance[order],
            color=PALETTE["teal"],
            edgecolor="white",
            alpha=0.85,
        )
        ax.set_xlabel("Mean |SHAP|", fontsize=12)
        label_panel(ax, panel_index)
        ax.tick_params(axis="both", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ranking = pd.DataFrame(
            {"Feature": item["feature_names"], "SHAP_importance": importance}
        ).sort_values("SHAP_importance", ascending=False)
        write_csv(ranking, out_dir / f"shap_{item['tag']}.csv")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig05_shap_importance_bar")


def plot_shap_dependence(shap_data: list[dict], out_dir: Path) -> None:
    # Four rows keep all eight diagnostic panels legible.
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.patch.set_facecolor("white")
    for dataset_index, item in enumerate(shap_data):
        importance = np.abs(item["shap_values"]).mean(axis=0)
        top_four = np.argsort(importance)[::-1][:4]
        for feature_position, feature_index in enumerate(top_four):
            panel_index = dataset_index * 4 + feature_position
            ax = axes.flat[panel_index]
            ax.scatter(
                item["X"][:, feature_index],
                item["shap_values"][:, feature_index],
                c=item["shap_values"][:, feature_index],
                cmap="RdYlBu_r",
                alpha=0.7,
                s=24,
                edgecolors="white",
                linewidths=0.2,
            )
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel(item["feature_names"][feature_index], fontsize=12)
            ax.set_ylabel(f"SHAP ({item['short_label']})", fontsize=12)
            label_panel(ax, panel_index)
            ax.tick_params(axis="both", labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig06_shap_dependence")


def plot_residuals(residual_data: list[list[dict]], out_dir: Path) -> None:
    # Two rows per target keep eight diagnostic panels readable at standard
    # text width.
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.patch.set_facecolor("white")
    for target_index, groups in enumerate(residual_data):
        summary_row = target_index * 2
        dataset_row = summary_row + 1
        y_true = np.concatenate([group["y_true"] for group in groups])
        y_pred = np.concatenate([group["y_pred"] for group in groups])
        residual = y_true - y_pred
        label = groups[0]["target_label"]
        axes[summary_row, 0].scatter(
            y_pred,
            residual,
            c=PALETTE["navy"],
            alpha=0.5,
            s=22,
            edgecolors="white",
            linewidths=0.2,
        )
        axes[summary_row, 0].axhline(
            0, color=PALETTE["dark_red"], linestyle="--", linewidth=1.5
        )
        axes[summary_row, 0].set_xlabel(f"Predicted {label}", fontsize=12)
        axes[summary_row, 0].set_ylabel("Residual", fontsize=12)
        axes[summary_row, 1].hist(
            residual, bins=30, color=PALETTE["teal"], edgecolor="white", alpha=0.85
        )
        axes[summary_row, 1].axvline(
            0, color=PALETTE["dark_red"], linestyle="--", linewidth=1.5
        )
        axes[summary_row, 1].set_xlabel("Residual", fontsize=12)
        axes[summary_row, 1].set_ylabel("Count", fontsize=12)
        for col, group in enumerate(groups):
            group_residual = group["y_true"] - group["y_pred"]
            axes[dataset_row, col].scatter(
                group["y_pred"],
                group_residual,
                alpha=0.55,
                s=22,
                color=group["color"],
                edgecolors="white",
                linewidths=0.2,
            )
            axes[dataset_row, col].axhline(
                0, color=PALETTE["dark_red"], linestyle="--", linewidth=1.2
            )
            axes[dataset_row, col].set_xlabel(f"Predicted {label}", fontsize=12)
            axes[dataset_row, col].set_ylabel("Residual", fontsize=12)
    for panel_index, ax in enumerate(axes.flat):
        label_panel(ax, panel_index)
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir, "Figure-5")


def plot_formation_performance(performance_data: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")
    records = []
    for panel_index, (ax, item) in enumerate(zip(axes, performance_data)):
        formations = sorted(set(item["forms_por"]) | set(item["forms_perm"]))
        porosity_values = []
        permeability_values = []
        porosity_counts = []
        permeability_counts = []
        for formation in formations:
            porosity_mask = item["forms_por"] == formation
            permeability_mask = item["forms_perm"] == formation
            n_porosity = int(porosity_mask.sum())
            n_permeability = int(permeability_mask.sum())
            porosity_r2 = (
                r2_score(item["y_por"][porosity_mask], item["yhat_por"][porosity_mask])
                if n_porosity >= 3
                else np.nan
            )
            permeability_r2 = (
                r2_score(item["y_perm"][permeability_mask], item["yhat_perm"][permeability_mask])
                if n_permeability >= 3
                else np.nan
            )
            porosity_values.append(porosity_r2)
            permeability_values.append(permeability_r2)
            porosity_counts.append(n_porosity)
            permeability_counts.append(n_permeability)
            records.extend(
                [
                    {
                        "Dataset": item["title"],
                        "Formation": formation,
                        "Target": "Porosity",
                        "n": n_porosity,
                        "R2_OOF": porosity_r2,
                    },
                    {
                        "Dataset": item["title"],
                        "Formation": formation,
                        "Target": "Permeability",
                        "n": n_permeability,
                        "R2_OOF": permeability_r2,
                    },
                ]
            )

        x = np.arange(len(formations))
        width = 0.35
        bars_porosity = ax.bar(
            x - width / 2,
            porosity_values,
            width,
            label="Porosity",
            color=PALETTE["navy"],
            alpha=0.85,
        )
        bars_permeability = ax.bar(
            x + width / 2,
            permeability_values,
            width,
            label="Permeability",
            color=PALETTE["teal"],
            alpha=0.85,
        )
        for bars, values, counts in [
            (bars_porosity, porosity_values, porosity_counts),
            (bars_permeability, permeability_values, permeability_counts),
        ]:
            for bar, value, count in zip(bars, values, counts):
                if np.isnan(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.02,
                        f"N/A\n(n={count})",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                        color=PALETTE["gray"],
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value - 0.02,
                        f"n={count}",
                        ha="center",
                        va="top",
                        fontsize=8,
                        rotation=90,
                        color="white",
                        fontweight="bold",
                    )

        wrapped_labels = ["\n".join(textwrap.wrap(str(f), width=18)) for f in formations]
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_labels, fontsize=8, rotation=12, ha="right")
        ax.set_ylabel(r"Out-of-fold $R^2$", fontsize=10)
        label_panel(ax, panel_index)
        ax.axhline(0.8, color=PALETTE["gold"], linestyle="--", linewidth=1.2, label=r"$R^2=0.8$")
        finite = [v for v in porosity_values + permeability_values if np.isfinite(v)]
        lower = min(-0.1, min(finite) - 0.1) if finite else -0.1
        ax.set_ylim(lower, 1.05)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig08_formation_performance")
    metrics = pd.DataFrame(records)
    write_csv(metrics, out_dir / "formation_performance_oof.csv")


def assert_oof_matches_cv(oof_metrics: dict[str, float], cv_table: pd.DataFrame, tag: str) -> None:
    row = cv_table.loc[cv_table["Model"] == "XGBoost"].iloc[0]
    for metric in ["R2_mean", "R2_std", "RMSE_mean", "RMSE_std"]:
        if not np.isclose(oof_metrics[metric], row[metric], rtol=0, atol=1e-12):
            raise RuntimeError(
                f"OOF/CV mismatch for {tag} {metric}: {oof_metrics[metric]} vs {row[metric]}"
            )


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "lightgbm": lgb.__version__,
        "shap": shap.__version__,
        "optuna": optuna.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible petrographic ML pipeline")
    parser.add_argument(
        "--sadrikhanloo-data",
        default=None,
        help="Sadrikhanloo et al. CSV (auto-detected in data/raw by default)",
    )
    parser.add_argument(
        "--busch-hilgers-data",
        default=None,
        help="Busch and Hilgers CSV or XLSX (auto-detected in data/raw by default)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: outputs at the repository root)",
    )
    parser.add_argument(
        "--replace-outputs",
        action="store_true",
        help="Remove generated artifacts already present in the output directory",
    )
    parser.add_argument("--hpo", action="store_true", help="Run seeded XGBoost HPO comparisons")
    parser.add_argument("--hpo-trials", type=int, default=40, help="Trials per target when --hpo is used")
    args = parser.parse_args()
    if args.hpo_trials < 1:
        parser.error("--hpo-trials must be a positive integer")

    out_dir = Path(args.outdir).resolve() if args.outdir else DEFAULT_OUTPUT_DIR
    auto_p1, auto_p2 = auto_detect_files()
    p1_path = Path(args.sadrikhanloo_data or auto_p1 or "").resolve()
    p2_path = Path(args.busch_hilgers_data or auto_p2 or "").resolve()
    if not p1_path.is_file() or not p2_path.is_file():
        parser.error("Both source datasets are required and must exist")
    if any(path.is_relative_to(out_dir) for path in (p1_path, p2_path)):
        parser.error("The output directory must not contain either source dataset")

    print("=" * 78)
    print("REPRODUCIBLE PETROGRAPHIC ML PIPELINE")
    print(f"seed={RANDOM_SEED} | folds={N_SPLITS} | jobs={N_JOBS} | output={out_dir}")
    print(json.dumps(package_versions(), indent=2, sort_keys=True))
    print("=" * 78)

    print(f"\n[DATA] Sadrikhanloo et al.: {p1_path}")
    df1 = engineer_features(load_sadrikhanloo_data(p1_path))
    print(f"[DATA] Busch and Hilgers: {p2_path}")
    df2 = engineer_features(load_busch_hilgers_data(p2_path))
    datasets = {"p1": df1, "p2": df2}

    prepared: dict[str, dict] = {}
    dataset_rows = []
    for key, df in datasets.items():
        # The Busch and Hilgers dataset contains no grain-size field. Exclude
        # log_GS instead of presenting an imputed constant as an observed predictor.
        porosity_feature_set = [
            c for c in PRIMARY_POROSITY_FEATURES if not (key == "p2" and c == "log_GS")
        ]
        diagnostic_feature_set = [
            c
            for c in LEAKAGE_DIAGNOSTIC_POROSITY_FEATURES
            if not (key == "p2" and c == "log_GS")
        ]
        permeability_feature_set = [
            c for c in PERMEABILITY_FEATURES if not (key == "p2" and c == "log_GS")
        ]
        X_por, y_por, forms_por, features_por, metadata_por = get_modeling_set(
            df, "Porosity_pct", porosity_feature_set
        )
        X_por_rqi, y_por_rqi, _, features_por_rqi, _ = get_modeling_set(
            df, "Porosity_pct", diagnostic_feature_set
        )
        X_perm, y_perm, forms_perm, features_perm, metadata_perm = get_modeling_set(
            df, "log_Perm", permeability_feature_set
        )
        prepared[key] = {
            "X_por": X_por,
            "y_por": y_por,
            "forms_por": forms_por,
            "features_por": features_por,
            "metadata_por": metadata_por,
            "X_por_with_rqi": X_por_rqi,
            "y_por_with_rqi": y_por_rqi,
            "features_por_with_rqi": features_por_rqi,
            "X_perm": X_perm,
            "y_perm": y_perm,
            "forms_perm": forms_perm,
            "features_perm": features_perm,
            "metadata_perm": metadata_perm,
        }
        expected_counts = {"p1": (157, 123), "p2": (850, 785)}
        if (len(y_por), len(y_perm)) != expected_counts[key]:
            raise RuntimeError(
                f"Unexpected modeling counts for {key}: porosity={len(y_por)}, "
                f"permeability={len(y_perm)}; expected {expected_counts[key]}"
            )
        print(
            f"  [{key}] primary porosity={X_por.shape}; with-RQI diagnostic={X_por_rqi.shape}; "
            f"permeability={X_perm.shape}"
        )
        dataset_rows.append(
            {
                "Dataset": key,
                "rows_loaded": len(df),
                "porosity_n": len(y_por),
                "permeability_n": len(y_perm),
                "porosity_features_primary": X_por.shape[1],
                "porosity_features_with_rqi": X_por_rqi.shape[1],
                "permeability_features": X_perm.shape[1],
            }
        )
    # Preserve previous outputs until both source datasets and cohorts are valid.
    try:
        prepare_output_directory(out_dir, args.replace_outputs)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    write_csv(pd.DataFrame(dataset_rows), out_dir / "dataset_summary.csv")

    print("\n[FIGURE 1] Dataset overview")
    plot_data_overview(df1, df2, out_dir)

    print("\n[CV] Primary benchmarks")
    cv_results: dict[str, pd.DataFrame] = {}
    diagnostic_results: dict[str, pd.DataFrame] = {}
    for key, data in prepared.items():
        label = f"{key.upper()} Porosity (primary, RQI excluded)"
        cv_results[f"{key}_por"], fold_por = cv_benchmark(
            build_models(), data["X_por"], data["y_por"], label
        )
        write_csv(cv_results[f"{key}_por"], out_dir / f"cv_{key}_porosity.csv")
        write_csv(fold_por, out_dir / f"cv_folds_{key}_porosity.csv")

        label = f"{key.upper()} Permeability (porosity-derived RQI included)"
        cv_results[f"{key}_perm"], fold_perm = cv_benchmark(
            build_models(), data["X_perm"], data["y_perm"], label
        )
        write_csv(cv_results[f"{key}_perm"], out_dir / f"cv_{key}_permeability.csv")
        write_csv(fold_perm, out_dir / f"cv_folds_{key}_permeability.csv")

        label = f"{key.upper()} Porosity leakage diagnostic (RQI included)"
        diagnostic_results[key], fold_diag = cv_benchmark(
            build_models(), data["X_por_with_rqi"], data["y_por_with_rqi"], label
        )
        write_csv(
            diagnostic_results[key], out_dir / f"diagnostic_{key}_porosity_with_rqi.csv"
        )
        write_csv(fold_diag, out_dir / f"diagnostic_folds_{key}_porosity_with_rqi.csv")

    print("\n[FIGURE 2] Primary CV benchmark")
    plot_cv_benchmark(cv_results, out_dir)

    hpo_comparison_rows = []
    if args.hpo:
        trial_word = "trial" if args.hpo_trials == 1 else "trials"
        print(
            f"\n[HPO] Seeded XGBoost optimization "
            f"({args.hpo_trials} {trial_word} per target)"
        )
        for key in ["p1", "p2"]:
            data = prepared[key]
            for target, X, y in [
                ("porosity", data["X_por"], data["y_por"]),
                ("permeability", data["X_perm"], data["y_perm"]),
            ]:
                tag = f"{key}_{target}"
                print(f"  Tuning {tag}...")
                best_params = tune_xgboost(X, y, args.hpo_trials, tag, out_dir)
                tuned_summary, tuned_folds = cv_benchmark(
                    {"XGBoost HPO": make_xgb(best_params)}, X, y, f"{tag} HPO", verbose=False
                )
                write_csv(tuned_folds, out_dir / f"hpo_folds_{tag}.csv")
                baseline_table = cv_results[f"{key}_{'por' if target == 'porosity' else 'perm'}"]
                baseline = baseline_table.loc[baseline_table["Model"] == "XGBoost"].iloc[0]
                tuned = tuned_summary.iloc[0]
                hpo_comparison_rows.append(
                    {
                        "Dataset": key,
                        "Target": target,
                        "n": len(y),
                        "Baseline_R2_mean": baseline["R2_mean"],
                        "Baseline_R2_std": baseline["R2_std"],
                        "Baseline_RMSE_mean": baseline["RMSE_mean"],
                        "Baseline_RMSE_std": baseline["RMSE_std"],
                        "HPO_R2_mean": tuned["R2_mean"],
                        "HPO_R2_std": tuned["R2_std"],
                        "HPO_RMSE_mean": tuned["RMSE_mean"],
                        "HPO_RMSE_std": tuned["RMSE_std"],
                        "RMSE_change_pct": 100.0
                        * (tuned["RMSE_mean"] - baseline["RMSE_mean"])
                        / baseline["RMSE_mean"],
                        "best_params": json.dumps(best_params, sort_keys=True, separators=(",", ":")),
                    }
                )
        write_csv(pd.DataFrame(hpo_comparison_rows), out_dir / "hpo_xgboost_comparison.csv")

    print("\n[OOF] XGBoost predictions used by Figures 3 and 5 and formation diagnostics")
    oof: dict[str, dict] = {}
    prediction_data = []
    colors = {"p1": FORM_COLORS_P1, "p2": FORM_COLORS_P2}
    labels = {
        ("p1", "por"): "Porosity (%)",
        ("p1", "perm"): r"$\log_{10}(k)$ [mD]",
        ("p2", "por"): "Porosity (%)",
        ("p2", "perm"): r"$\log_{10}(k)$ [mD]",
    }
    for key in ["p1", "p2"]:
        data = prepared[key]
        yhat_por, folds_por, metrics_por = oof_evaluate(
            make_xgb(), data["X_por"], data["y_por"]
        )
        yhat_perm, folds_perm, metrics_perm = oof_evaluate(
            make_xgb(), data["X_perm"], data["y_perm"]
        )
        assert_oof_matches_cv(metrics_por, cv_results[f"{key}_por"], f"{key} porosity")
        assert_oof_matches_cv(metrics_perm, cv_results[f"{key}_perm"], f"{key} permeability")

        por_predictions = data["metadata_por"].copy()
        por_predictions["fold"] = folds_por
        por_predictions["y_true"] = data["y_por"]
        por_predictions["y_pred_oof"] = yhat_por
        por_predictions["residual"] = data["y_por"] - yhat_por
        write_csv(por_predictions, out_dir / f"oof_predictions_{key}_porosity.csv")

        perm_predictions = data["metadata_perm"].copy()
        perm_predictions["fold"] = folds_perm
        perm_predictions["y_true"] = data["y_perm"]
        perm_predictions["y_pred_oof"] = yhat_perm
        perm_predictions["residual"] = data["y_perm"] - yhat_perm
        write_csv(perm_predictions, out_dir / f"oof_predictions_{key}_permeability.csv")
        oof[key] = {
            "yhat_por": yhat_por,
            "yhat_perm": yhat_perm,
        }
        prediction_data.extend(
            [
                {
                    "y_true": data["y_por"],
                    "y_pred": yhat_por,
                    "forms": data["forms_por"],
                    "colors": colors[key],
                    "label": labels[(key, "por")],
                    "metrics": metrics_por,
                },
                {
                    "y_true": data["y_perm"],
                    "y_pred": yhat_perm,
                    "forms": data["forms_perm"],
                    "colors": colors[key],
                    "label": labels[(key, "perm")],
                    "metrics": metrics_perm,
                },
            ]
        )

    print("\n[FIGURE 3] OOF predicted vs measured")
    plot_predicted_vs_actual(prediction_data, out_dir)

    print("\n[SHAP] Full-data explanatory models (not used for performance claims)")
    shap_data = []
    for key in ["p1", "p2"]:
        data = prepared[key]
        for target, X, y, feature_names, title, tag in [
            (
                "porosity",
                data["X_por"],
                data["y_por"],
                data["features_por"],
                f"{DATASET_DISPLAY_NAMES[key]}: Porosity (RQI excluded)",
                f"{key}_por",
            ),
            (
                "permeability",
                data["X_perm"],
                data["y_perm"],
                data["features_perm"],
                f"{DATASET_DISPLAY_NAMES[key]}: Permeability (RQI included)",
                f"{key}_perm",
            ),
        ]:
            model = make_xgb()
            model.fit(X, y)
            values = shap.TreeExplainer(model).shap_values(X)
            shap_data.append(
                {
                    "dataset": key,
                    "target": target,
                    "X": X,
                    "feature_names": feature_names,
                    "shap_values": values,
                    "title": title,
                    "tag": tag,
                    "short_label": (
                        f"{DATASET_DISPLAY_NAMES[key]} "
                        f"{'Por.' if target == 'porosity' else 'Perm.'}"
                    ),
                    "model": model,
                }
            )

    print("\n[FIGURE 4] SHAP beeswarms and auxiliary explanation plots")
    plot_shap_beeswarms(shap_data, out_dir)
    plot_shap_importance(shap_data, out_dir)
    dependence_data = [
        next(item for item in shap_data if item["dataset"] == "p1" and item["target"] == "porosity"),
        next(item for item in shap_data if item["dataset"] == "p2" and item["target"] == "permeability"),
    ]
    plot_shap_dependence(dependence_data, out_dir)

    print("\n[FIGURE 5] OOF residual analysis")
    residual_data = []
    for target in ["por", "perm"]:
        target_groups = []
        for key, color in [("p1", PALETTE["navy"]), ("p2", PALETTE["teal"])]:
            data = prepared[key]
            target_groups.append(
                {
                    "y_true": data[f"y_{target}"],
                    "y_pred": oof[key][f"yhat_{target}"],
                    "target_label": "Porosity (%)" if target == "por" else "log10(k) [mD]",
                    "dataset_label": DATASET_DISPLAY_NAMES[key],
                    "color": color,
                }
            )
        residual_data.append(target_groups)
    plot_residuals(residual_data, out_dir)

    print("\n[AUXILIARY] Per-formation descriptive OOF performance")
    performance_data = []
    for key in ["p1", "p2"]:
        data = prepared[key]
        performance_data.append(
            {
                "y_por": data["y_por"],
                "y_perm": data["y_perm"],
                "yhat_por": oof[key]["yhat_por"],
                "yhat_perm": oof[key]["yhat_perm"],
                "forms_por": data["forms_por"],
                "forms_perm": data["forms_perm"],
                "title": DATASET_DISPLAY_NAMES[key],
            }
        )
    plot_formation_performance(performance_data, out_dir)

    print("\n[SUMMARY] Best primary CV model per target")
    summary_rows = []
    for key in ["p1", "p2"]:
        for target, suffix in [("Porosity", "por"), ("Permeability", "perm")]:
            best = cv_results[f"{key}_{suffix}"].iloc[0]
            summary_rows.append(
                {
                    "Dataset": DATASET_DISPLAY_NAMES[key],
                    "Target": target,
                    "Best_model": best["Model"],
                    "R2_CV_mean": best["R2_mean"],
                    "R2_CV_std": best["R2_std"],
                    "RMSE_CV_mean": best["RMSE_mean"],
                    "RMSE_CV_std": best["RMSE_std"],
                }
            )
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    write_csv(summary, out_dir / "metrics_summary_cv_best.csv")

    trainfit_rows = []
    for item in shap_data:
        data = prepared[item["dataset"]]
        suffix = "por" if item["target"] == "porosity" else "perm"
        y = data[f"y_{suffix}"]
        X = data[f"X_{suffix}"]
        prediction = item["model"].predict(X)
        trainfit_rows.append(
            {
                "Dataset": DATASET_DISPLAY_NAMES[item["dataset"]],
                "Target": item["target"].title(),
                "R2_trainfit": r2_score(y, prediction),
            }
        )
    write_csv(pd.DataFrame(trainfit_rows), out_dir / "metrics_summary_trainfit_diagnostic.csv")

    manifest = {
        "pipeline": portable_repo_path(SCRIPT_PATH),
        "seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "n_jobs": N_JOBS,
        "cv_design": "KFold(shuffle=True), sample-level; not grouped by well or formation",
        "porosity_primary": "RQI_proxy excluded",
        "porosity_diagnostic": "RQI_proxy included (target-leakage diagnostic only)",
        "permeability": "RQI_proxy included; measured porosity required at inference",
        "hpo_enabled": bool(args.hpo),
        "hpo_trials_per_target": args.hpo_trials if args.hpo else 0,
        "package_versions": package_versions(),
        "deterministic_environment": DETERMINISTIC_ENV,
        "datasets": {
            "sadrikhanloo": {"path": portable_repo_path(p1_path)},
            "busch_hilgers": {"path": portable_repo_path(p2_path)},
        },
        "feature_sets": {
            "sadrikhanloo": {
                "porosity_primary": prepared["p1"]["features_por"],
                "porosity_with_rqi_diagnostic": prepared["p1"]["features_por_with_rqi"],
                "permeability": prepared["p1"]["features_perm"],
            },
            "busch_hilgers": {
                "porosity_primary": prepared["p2"]["features_por"],
                "porosity_with_rqi_diagnostic": prepared["p2"]["features_por_with_rqi"],
                "permeability": prepared["p2"]["features_perm"],
            },
        },
        "model_parameters": {
            "xgboost_fixed_baseline": XGB_BASE_PARAMS,
            "random_forest_fixed_baseline": RF_BASE_PARAMS,
            "lightgbm_fixed_baseline": LGBM_BASE_PARAMS,
        },
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
