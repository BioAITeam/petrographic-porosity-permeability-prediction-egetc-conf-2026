"""Nominal formation encoding, geological holdouts and nested model tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import petrographic_ml_pipeline as base
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def datasets():
    first, second = base.auto_detect_files()
    require(first is not None and second is not None, 'Both input datasets are required')
    return {'p1': base.load_sadrikhanloo_data(first), 'p2': base.load_busch_hilgers_data(second)}


def well_ids(frame, dataset):
    if dataset == 'p2':
        require(frame.Well.notna().all(), 'Missing well identifiers')
        return frame.Well.astype(str).str.strip()
    formation = frame.Formation.astype(str).str.strip()
    prefix = frame.Sample_ID.astype(str).str.extract(r'^([A-Z]+)', expand=False)
    prefix = prefix.where(formation != 'Buntsandstein D', 'D')
    require(prefix.notna().all(), 'Unrecognized sample identifier')
    require(prefix.loc[formation.str.endswith('A+B')].isin(['A', 'B']).all(), 'Ambiguous paired-well identifier')
    return formation.str.replace(' A+B', '', regex=False) + ':' + prefix


def task_data(frame, dataset, task):
    require(dataset in ('p1', 'p2'), 'Unknown dataset')
    require(task in ('porosity', 'conditional', 'petrography'), 'Unknown prediction task')
    target = 'Porosity_pct' if task == 'porosity' else 'log_Perm'
    # Permeability ablations share exactly the same paired observations.
    needed = ['Porosity_pct'] if task == 'porosity' else ['log_Perm', 'Porosity_pct']
    valid = frame.dropna(subset=needed).copy()
    numeric = [c for c in base.FEAT_COLS if c not in ('Formation_code', 'RQI_proxy')
               and not (dataset == 'p2' and c == 'log_GS')]
    if task == 'conditional':
        numeric.append('RQI_proxy')
    require(task == 'conditional' or 'RQI_proxy' not in numeric, 'Target proxy in unconditional features')
    engineered = base.engineer_features(valid)
    X = engineered[numeric].copy()
    # Restore any observed predictor missingness before fold-local imputation.
    for column in numeric:
        if column in valid:
            X[column] = pd.to_numeric(valid[column], errors='coerce')
    X['Formation'] = valid.Formation.astype(str).str.strip()
    # A bounded balance avoids arbitrary magnitudes when cement is absent.
    clay, cement = X['Total_clay'], X['Total_cement']
    X['Clay_cement_ratio'] = np.divide(clay, clay + cement, out=np.zeros(len(X)), where=(clay + cement).to_numpy() != 0)
    y = valid[target].to_numpy(float)
    wells = well_ids(valid, dataset).to_numpy()
    return X.reset_index(drop=True), y, wells, valid.index.to_numpy(), numeric


def estimator(numeric, model='XGBoost', encoding='onehot', imputation='median', params=None):
    require(model in ('XGBoost', 'Ridge'), 'Unknown model')
    require(encoding in ('onehot', 'ordinal'), 'Unknown formation encoding')
    require(imputation in ('median', 'zero'), 'Unknown imputation strategy')
    imp = SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True) if imputation == 'zero' else SimpleImputer(strategy='median', keep_empty_features=True)
    categorical = OneHotEncoder(handle_unknown='ignore', sparse_output=False) if encoding == 'onehot' else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    preprocessing = ColumnTransformer([('numeric', imp, numeric), ('formation', categorical, ['Formation'])])
    predictor = base.make_xgb(params) if model == 'XGBoost' else Pipeline([('scale', RobustScaler()), ('ridge', Ridge(alpha=1, solver='svd'))])
    return Pipeline([('preprocessing', preprocessing), ('model', predictor)])


def splits(X, y, wells, scheme):
    require(scheme in ('sample', 'well', 'formation'), 'Unknown validation scheme')
    if scheme == 'sample':
        return list(KFold(5, shuffle=True, random_state=SEED).split(X, y))
    groups = wells if scheme == 'well' else X.Formation.to_numpy()
    splitter = GroupKFold(5) if scheme == 'well' else LeaveOneGroupOut()
    result = list(splitter.split(X, y, groups))
    for train, test in result:
        require(not set(groups[train]) & set(groups[test]), 'Training/test group overlap')
    return result


def score(y, predicted):
    return dict(R2=r2_score(y, predicted), RMSE=np.sqrt(mean_squared_error(y, predicted)), MAE=mean_absolute_error(y, predicted))


def evaluate(X, y, wells, source_rows, numeric, scheme, model, encoding='onehot', imputation='median'):
    prediction = np.full(len(y), np.nan)
    assignment = np.zeros(len(y), int)
    rows = []
    template = estimator(numeric, model, encoding, imputation)
    for fold, (train, test) in enumerate(splits(X, y, wells, scheme), 1):
        fit = clone(template).fit(X.iloc[train], y[train])
        prediction[test] = fit.predict(X.iloc[test])
        assignment[test] = fold
        rows.append(dict(fold=fold, n_train=len(train), n_test=len(test), **score(y[test], prediction[test])))
    require(np.isfinite(prediction).all(), 'Incomplete predictions')
    oof = pd.DataFrame(dict(source_row=source_rows, y_true=y, y_pred=prediction, fold=assignment,
                            well=wells, formation=X.Formation.to_numpy()))
    return oof, pd.DataFrame(rows)


def nested(X, y, wells, source_rows, numeric, out, tag, trials):
    outer = splits(X, y, wells, 'sample')
    predictions, records, trials_all = [], [], []
    for fold, (train, test) in enumerate(outer, 1):
        inner = list(KFold(3, shuffle=True, random_state=SEED).split(X.iloc[train]))
        def objective(trial):
            params = dict(n_estimators=trial.suggest_int('n_estimators', 100, 500),
                          max_depth=trial.suggest_int('max_depth', 3, 8),
                          learning_rate=trial.suggest_float('learning_rate', .01, .2, log=True),
                          subsample=trial.suggest_float('subsample', .6, 1),
                          colsample_bytree=trial.suggest_float('colsample_bytree', .5, 1),
                          reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 1, log=True))
            scores = []
            for a, b in inner:
                fit = estimator(numeric, params=params).fit(X.iloc[train[a]], y[train[a]])
                scores.append(score(y[train[b]], fit.predict(X.iloc[train[b]]))['RMSE'])
            return float(np.mean(scores))
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=trials, n_jobs=1)
        fit = estimator(numeric, params=study.best_params).fit(X.iloc[train], y[train])
        pred = fit.predict(X.iloc[test]).astype(float)
        predictions.append(pd.DataFrame(dict(source_row=source_rows[test], y_true=y[test], y_pred=pred,
                                              fold=fold, well=wells[test], formation=X.Formation.iloc[test].to_numpy())))
        records.append(dict(fold=fold, n_train=len(train), n_test=len(test), **score(y[test], pred)))
        for trial in study.trials:
            trials_all.append(dict(outer_fold=fold, trial=trial.number, inner_RMSE=trial.value, **trial.params))
        (out / f'params_{tag}_fold{fold}.json').write_text(json.dumps(study.best_params, indent=2) + '\n', encoding='utf-8')
        print(f'{tag} outer {fold}/5 complete', flush=True)
    pd.DataFrame(trials_all).to_csv(out / f'trials_{tag}.csv', index=False)
    return pd.concat(predictions).sort_values('source_row'), pd.DataFrame(records)


def summarize(tag, oof, folds, **labels):
    return dict(task=tag, n=len(oof), wells=oof.well.nunique(), formations=oof.formation.nunique(),
                folds=len(folds), **labels, **{f'{metric}_mean': folds[metric].mean() for metric in ('R2', 'RMSE', 'MAE')},
                **{f'{metric}_sd': folds[metric].std(ddof=1) for metric in ('R2', 'RMSE', 'MAE')},
                **{f'{metric}_pooled': value for metric, value in score(oof.y_true, oof.y_pred).items()})


def prepare_stage_directory(out, stage):
    """Reject incomplete runs and incompatible stages before writing results."""
    out = Path(out).resolve()
    require(stage in ('fixed', 'nested'), 'Unknown analysis stage')
    require(out != ROOT and out not in ROOT.parents and not any(out.is_relative_to(ROOT / p) for p in ['code', 'data']), 'Protected output directory')
    patterns = (['validation_summary.csv', 'manifest_fixed.json'] if stage == 'fixed' else
                ['nested_summary.csv', 'manifest_nested.json', '*_nested_XGBoost.csv', 'trials_*.csv', 'params_*.json'])
    existing = [path for pattern in patterns for path in out.glob(pattern)]
    if stage == 'fixed':
        existing.extend(path for pattern in ['oof_*.csv', 'folds_*.csv'] for path in out.glob(pattern)
                        if not path.name.endswith('_nested_XGBoost.csv'))
    require(not existing, 'Choose a new output directory; completed or interrupted stages are not overwritten')
    other = out / ('manifest_nested.json' if stage == 'fixed' else 'manifest_fixed.json')
    if other.exists():
        manifest = json.loads(other.read_text(encoding='utf-8'))
        require(manifest.get('seed') == SEED and manifest.get('versions') == base.package_versions(),
                'The existing stage uses a different seed or environment')
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'temp' / 'evaluation')
    parser.add_argument('--nested-trials', type=int, default=40)
    parser.add_argument('--stage', choices=['fixed', 'nested'], default='fixed')
    args = parser.parse_args()
    require(args.nested_trials > 0, 'Positive trial count required')
    out = prepare_stage_directory(args.outdir, args.stage)
    result_file = out / ('nested_summary.csv' if args.stage == 'nested' else 'validation_summary.csv')
    frames = datasets()
    summaries, audit = [], []
    for dataset, frame in frames.items():
        for task in ['porosity', 'conditional', 'petrography']:
            tag = f'{dataset}_{task}'
            X, y, wells, source_rows, numeric = task_data(frame, dataset, task)
            audit.append(dict(task=tag, n=len(y), wells=len(set(wells)), formations=X.Formation.nunique(),
                              numeric_features=len(numeric), missing_predictor_cells=int(X[numeric].isna().sum().sum()),
                              target_missing=int(frame[('Porosity_pct' if task == 'porosity' else 'log_Perm')].isna().sum())))
            if args.stage == 'fixed':
                for scheme in ['sample', 'well', 'formation']:
                    for model in ['XGBoost', 'Ridge']:
                        name = f'{tag}_{scheme}_{model}'
                        oof, folds = evaluate(X, y, wells, source_rows, numeric, scheme, model)
                        oof.to_csv(out / f'oof_{name}.csv', index=False)
                        folds.to_csv(out / f'folds_{name}.csv', index=False)
                        summaries.append(summarize(tag, oof, folds, scheme=scheme, model=model, encoding='onehot', imputation='median'))
                        if scheme == 'sample':
                            for encoding, imputation in [('ordinal', 'median'), ('onehot', 'zero')]:
                                extra = f'{name}_{encoding}_{imputation}'
                                other, other_folds = evaluate(X, y, wells, source_rows, numeric, scheme, model, encoding, imputation)
                                other.to_csv(out / f'oof_{extra}.csv', index=False)
                                other_folds.to_csv(out / f'folds_{extra}.csv', index=False)
                                summaries.append(summarize(tag, other, other_folds, scheme=scheme, model=model, encoding=encoding, imputation=imputation))
                print(f'{tag}: grouped and encoding analyses complete', flush=True)
            if args.stage == 'nested' and task != 'petrography':
                oof, folds = nested(X, y, wells, source_rows, numeric, out, tag, args.nested_trials)
                oof.to_csv(out / f'oof_{tag}_nested_XGBoost.csv', index=False)
                folds.to_csv(out / f'folds_{tag}_nested_XGBoost.csv', index=False)
                summaries.append(summarize(tag, oof, folds, scheme='nested', model='XGBoost', encoding='onehot', imputation='median'))
    pd.DataFrame(summaries).to_csv(result_file, index=False)
    pd.DataFrame(audit).to_csv(out / 'cohort_audit.csv', index=False)
    manifest = dict(seed=SEED, nested_trials=args.nested_trials, stage=args.stage, versions=base.package_versions(),
                    formation_encoding='fold-local one-hot, unseen levels all-zero',
                    clay_cement_balance='clay / (clay + cement), zero if both absent',
                    well_groups_p1='formation family plus source sample prefix, D for Buntsandstein D',
                    statistics='mean fold metrics and sample SD; pooled metrics reported separately')
    (out / f'manifest_{args.stage}.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
