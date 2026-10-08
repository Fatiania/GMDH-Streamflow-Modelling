import re
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from openpyxl.utils import get_column_letter


# =========================================================
# USER SETTINGS
# =========================================================
MAIN_INPUT_FOLDER = Path(r"D:\Morocco\Paper_catchment5\Dataset\5_catch\PHYSICAL\DATA")

# Excel file where each sheet contains:
# column 1 = point id
# column 2 = cluster number
CLUSTER_EXCEL_FILE = Path(
    r"D:\Morocco\Paper_catchment5\Dataset\5_catch\PHYSICAL\DATA\PerPoint_Clustering_Spatial_SOM\all_points_cluster.xlsx"
)

RUNOFF_KEYWORDS = ["runoff", "discharge", "streamflow", "flow", "q"]

MIN_ROWS_REQUIRED = 10
MIN_VALID_DATE_COLUMNS = 12

TEST_FRACTION = 0.15
MIN_TEST_ROWS = 3

OUTPUT_FOLDER_NAME = "Sensitivity_GMDH_results_Clusters"

FIGURE_DPI = 1200

# Runoff is in mm/month.
RUNOFF_UNIT_LABEL = r"mm/month"

# Set to 0 when using mm/month so axes are not divided by 10^3.
AXIS_SCALE_POWER = 0

# Fixed colors for subclusters.
# cluster 0 = blue, cluster 1 = red, cluster 2 = green, cluster 3 = orange.
CLUSTER_PLOT_COLORS = {
    0: "blue",
    1: "red",
    2: "green",
    3: "orange",
    4: "violet",
    5: "brown",
    6: "pink",
    7: "gray",
    8: "olive",
    9: "cyan",
}

USE_GMDH = True
GMDH_MAX_LAYERS = 4
GMDH_TOP_NEURONS_PER_LAYER = 6
GMDH_VALIDATION_FRACTION = 0.30
GMDH_MIN_VALIDATION_ROWS = 4
GMDH_MIN_IMPROVEMENT = 1e-6

# Minimum sample-size control for quadratic GMDH pair neurons.
# A pair neuron has 6 fitted terms: 1, x1, x2, x1^2, x2^2, x1*x2.
# Requiring several rows per fitted term reduces severe overfitting.
GMDH_PAIR_NEURON_N_TERMS = 6
GMDH_MIN_ROWS_PER_PAIR_TERM = 3
GMDH_MIN_PAIR_TRAIN_ROWS = GMDH_PAIR_NEURON_N_TERMS * GMDH_MIN_ROWS_PER_PAIR_TERM

# Standardize predictors before fitting polynomial GMDH neurons.
# This improves numerical stability when variables have different units/scales.
STANDARDIZE_PREDICTORS = True
STANDARDIZATION_EPS = 1e-12

RUN_FULL_MODEL = True
RUN_TWO_PARAMETER_COMBINATIONS = True
RUN_LEAVE_ONE_OUT = True

# Cluster Excel structure
CLUSTER_POINT_COLUMN_INDEX = 0
CLUSTER_LABEL_COLUMN_INDEX = 1

# Data Excel structure:
# column 1 = point id
# column 2 onward = monthly values
# For datasets with x/y coordinate columns preceding monthly observations,
# set FIRST_MONTHLY_COLUMN_INDEX = 3.
# For datasets without x/y coordinates, with monthly observations starting in column 2,
# set FIRST_MONTHLY_COLUMN_INDEX = 1.
DATA_POINT_COLUMN_INDEX = 0
FIRST_MONTHLY_COLUMN_INDEX = 3


# =========================================================
# PARAMETER ABBREVIATIONS
# =========================================================
PARAMETER_ABBREVIATIONS = {
    "tmax": "Tmax",
    "tmin": "Tmin",
    "temperature_max": "Tmax",
    "temperature_min": "Tmin",
    "maximum_temperature": "Tmax",
    "minimum_temperature": "Tmin",
    "potential_evaporation": "PE",
    "potentialevaporation": "PE",
    "potential_evapotranspiration": "PE",
    "potentialevapotranspiration": "PE",
    "evaporation": "PE",
    "evapotranspiration": "PE",
    "pet": "PE",
    "pe": "PE",
    "total_evaporation": "TE",
    "totalevaporation": "TE",
    "precipitation": "P",
    "precip": "P",
    "rainfall": "P",
    "ppt": "P",
    "p": "P",
    "shortwave_radiation": "SR",
    "shortwaveradiation": "SR",
    "solar_radiation": "SR",
    "solarradiation": "SR",
    "radiation": "SR",
    "sr": "SR",
    "soil_moisture": "SM",
    "soilmoisture": "SM",
    "soil_moisute": "SM",
    "soilmoisute": "SM",
    "sm": "SM",
}


# =========================================================
# BASIC HELPERS
# =========================================================
def normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text).lower())


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_point_id(value):
    if pd.isna(value):
        return None

    s = str(value).strip()

    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def get_cluster_plot_color(cluster_number):
    if cluster_number is None or pd.isna(cluster_number):
        return "black"
    cluster_number = int(cluster_number)
    return CLUSTER_PLOT_COLORS.get(cluster_number, "black")


def find_excel_files(folder: Path):
    files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    return [f for f in files if not f.name.startswith("~$")]


def detect_runoff_file(files):
    ordered_keywords = sorted(RUNOFF_KEYWORDS, key=lambda x: len(normalize_text(x)), reverse=True)
    for f in files:
        fname = normalize_text(f.stem)
        for kw in ordered_keywords:
            if normalize_text(kw) in fname:
                return f
    return None


def clean_parameter_name(file_path: Path, existing_names=None):
    name = file_path.stem.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-]", "_", name)

    if existing_names is None:
        return name

    base = name
    i = 2
    while name in existing_names:
        name = f"{base}_{i}"
        i += 1
    return name


def auto_abbreviate_parameter(name: str) -> str:
    lname = normalize_text(name)

    if "tmax" in lname:
        return "Tmax"
    if "tmin" in lname:
        return "Tmin"
    if "temperaturemax" in lname or ("temperature" in lname and "max" in lname):
        return "Tmax"
    if "temperaturemin" in lname or ("temperature" in lname and "min" in lname):
        return "Tmin"
    if "potentialevaporation" in lname or "potentialevapotranspiration" in lname or lname in {"pet", "pe"}:
        return "PE"
    if "totalevaporation" in lname:
        return "TE"
    if "precipitation" in lname or "rainfall" in lname or lname in {"ppt", "p"}:
        return "P"
    if "shortwaveradiation" in lname or "solarradiation" in lname or lname == "sr":
        return "SR"
    if "soilmoisture" in lname or "soilmoisute" in lname or lname == "sm":
        return "SM"

    parts = [p for p in re.split(r"[_\-\s]+", str(name)) if p]
    if len(parts) >= 2:
        return "".join(p[0] for p in parts).upper()

    token = parts[0] if parts else str(name)
    return token[:4].upper()


def build_parameter_symbol_map(parameter_names):
    symbol_map = {}
    used_symbols = set()
    normalized_manual = {normalize_text(k): v for k, v in PARAMETER_ABBREVIATIONS.items()}

    for name in parameter_names:
        key = normalize_text(name)
        symbol = normalized_manual[key] if key in normalized_manual else auto_abbreviate_parameter(name)

        base_symbol = symbol
        i = 2
        while symbol in used_symbols:
            symbol = f"{base_symbol}_{i}"
            i += 1

        symbol_map[name] = symbol
        used_symbols.add(symbol)

    return symbol_map


def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================================================
# CLUSTER LOADING
# =========================================================
def load_cluster_definitions(cluster_excel_file: Path):
    if not cluster_excel_file.exists():
        raise FileNotFoundError(f"Cluster Excel file does not exist:\n{cluster_excel_file}")

    xls = pd.ExcelFile(cluster_excel_file)
    cluster_jobs = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(cluster_excel_file, sheet_name=sheet_name)

        if df.shape[1] < 2:
            print(f"Skipped cluster sheet {sheet_name}: fewer than 2 columns")
            continue

        point_col = df.columns[CLUSTER_POINT_COLUMN_INDEX]
        cluster_col = df.columns[CLUSTER_LABEL_COLUMN_INDEX]

        tmp = df[[point_col, cluster_col]].copy()
        tmp.columns = ["point_id", "cluster_number"]

        tmp["point_id"] = tmp["point_id"].map(normalize_point_id)
        tmp["cluster_number"] = pd.to_numeric(tmp["cluster_number"], errors="coerce")

        tmp = tmp.dropna(subset=["point_id", "cluster_number"]).copy()
        tmp["cluster_number"] = tmp["cluster_number"].astype(int)

        for cluster_number in sorted(tmp["cluster_number"].unique()):
            point_ids = tmp.loc[tmp["cluster_number"] == cluster_number, "point_id"].tolist()
            point_ids = sorted(set(point_ids))

            if len(point_ids) == 0:
                continue

            cluster_jobs.append({
                "cluster_sheet": sheet_name,
                "cluster_number": int(cluster_number),
                "point_ids": point_ids,
                "cluster_label": f"{safe_name(sheet_name)}__cluster_{int(cluster_number)}",
                "n_points": len(point_ids),
            })

    if not cluster_jobs:
        raise ValueError("No valid cluster groups were found in the cluster Excel file.")

    return cluster_jobs


# =========================================================
# ROBUST NUMERIC CONVERSION
# =========================================================
def parse_maybe_number(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)

    s = str(value).strip()
    if s == "":
        return np.nan

    s_lower = s.lower()
    if s_lower in {"nan", "none", "na", "n/a", "null", "missing", "nodata", "--"}:
        return np.nan

    s = s.replace("\u2212", "-").replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("\u00A0", " ").strip()

    if re.fullmatch(r"\(.*\)", s):
        s = "-" + s[1:-1]

    s = s.replace(" ", "")

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 4:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    out = pd.to_numeric(s, errors="coerce")
    if pd.notna(out):
        return float(out)

    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)
    if match:
        out = pd.to_numeric(match.group(0), errors="coerce")
        if pd.notna(out):
            return float(out)

    return np.nan


def convert_dataframe_values_to_numeric(df):
    return df.apply(lambda col: col.map(parse_maybe_number))


# =========================================================
# PREDICTOR STANDARDIZATION
# =========================================================
def fit_predictor_standardizer(X, reference_rows=None):
    """Return column means/stds for predictor standardization.

    Parameters
    ----------
    X : array-like
        Predictor matrix.
    reference_rows : slice, array-like, or None
        Rows used to estimate the standardization parameters. For model
        selection, this should be the inner training period only, not the
        validation period.
    """
    X = np.asarray(X, dtype=float)

    if reference_rows is None:
        X_ref = X
    else:
        X_ref = X[reference_rows]

    means = np.nanmean(X_ref, axis=0)
    stds = np.nanstd(X_ref, axis=0, ddof=0)

    # Constant or near-constant predictors cannot be scaled by their std.
    # Use 1.0 so they are centered but not amplified.
    bad = (~np.isfinite(stds)) | (stds < STANDARDIZATION_EPS)
    stds = stds.copy()
    stds[bad] = 1.0

    means = np.where(np.isfinite(means), means, 0.0)

    return {
        "enabled": bool(STANDARDIZE_PREDICTORS),
        "means": means,
        "stds": stds,
    }


def apply_predictor_standardizer(X, standardizer):
    """Apply a fitted standardizer to a predictor matrix."""
    X = np.asarray(X, dtype=float)

    if not standardizer or not standardizer.get("enabled", False):
        return X

    means = np.asarray(standardizer["means"], dtype=float)
    stds = np.asarray(standardizer["stds"], dtype=float)

    return (X - means) / stds


# =========================================================
# DATE PARSING
# =========================================================
def parse_single_date(text):
    s = str(text).strip()
    if s == "":
        return pd.NaT

    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return pd.Timestamp(dt.year, dt.month, 1)

    s2 = s.replace("_", "-").replace("/", "-").replace(".", "-").strip()
    patterns = ["%Y-%m", "%m-%Y", "%b-%Y", "%B-%Y", "%Y-%b", "%Y-%B", "%Y%m", "%m%Y"]

    for fmt in patterns:
        dt = pd.to_datetime(s2, format=fmt, errors="coerce")
        if pd.notna(dt):
            return pd.Timestamp(dt.year, dt.month, 1)

    return pd.NaT


def parse_date_columns(columns):
    return [parse_single_date(col) for col in columns]


# =========================================================
# FILE VALIDATION AND LOADING
# =========================================================
def is_valid_parameter_file(file_path: Path):
    try:
        df = pd.read_excel(file_path, nrows=5)

        if df.shape[1] < FIRST_MONTHLY_COLUMN_INDEX + 1:
            return False, "not enough columns"

        monthly_cols = df.columns[FIRST_MONTHLY_COLUMN_INDEX:]
        parsed_dates = parse_date_columns(monthly_cols)
        n_valid_dates = sum(pd.notna(x) for x in parsed_dates)

        if n_valid_dates < MIN_VALID_DATE_COLUMNS:
            return False, f"only {n_valid_dates} date-like monthly columns detected"

        return True, f"{n_valid_dates} monthly columns detected"

    except Exception as e:
        return False, f"read error: {e}"


def load_monthly_mean_from_excel(file_path: Path, variable_name: str, point_ids_filter=None):
    df = pd.read_excel(file_path)

    if df.shape[1] < FIRST_MONTHLY_COLUMN_INDEX + 1:
        raise ValueError(f"{file_path.name} has too few columns.")

    if point_ids_filter is not None:
        point_col = df.columns[DATA_POINT_COLUMN_INDEX]

        wanted_ids = set(normalize_point_id(x) for x in point_ids_filter)
        df["_point_id_normalized_"] = df[point_col].map(normalize_point_id)

        before_rows = len(df)
        df = df[df["_point_id_normalized_"].isin(wanted_ids)].copy()
        after_rows = len(df)

        df = df.drop(columns=["_point_id_normalized_"], errors="ignore")

        if after_rows == 0:
            raise ValueError(
                f"{file_path.name}: no matching points found for this cluster. "
                f"Original rows: {before_rows}"
            )

    monthly_cols = df.columns[FIRST_MONTHLY_COLUMN_INDEX:]
    parsed_dates = parse_date_columns(monthly_cols)

    valid_pairs = [(col, dt) for col, dt in zip(monthly_cols, parsed_dates) if pd.notna(dt)]
    if len(valid_pairs) == 0:
        raise ValueError(f"{file_path.name}: no valid monthly columns could be parsed.")

    valid_cols = [col for col, _ in valid_pairs]
    valid_dates = [dt for _, dt in valid_pairs]

    raw_monthly_values = df.loc[:, valid_cols]
    monthly_values = convert_dataframe_values_to_numeric(raw_monthly_values)

    if int(monthly_values.notna().sum().sum()) == 0:
        raise ValueError(f"{file_path.name}: monthly values found, but none converted to numeric.")

    monthly_mean = monthly_values.mean(axis=0, skipna=True)
    valid_month_mask = monthly_mean.notna().values

    valid_dates = [d for d, keep in zip(valid_dates, valid_month_mask) if keep]
    monthly_mean = monthly_mean[valid_month_mask]

    if len(monthly_mean) == 0:
        raise ValueError(f"{file_path.name}: all converted monthly values are NaN.")

    out = pd.DataFrame({"date": valid_dates, variable_name: monthly_mean.values})
    out = out.dropna(subset=["date"])
    out = out.groupby("date", as_index=False)[variable_name].mean()
    out = out.sort_values("date").reset_index(drop=True)

    return out, df


# =========================================================
# METRICS
# =========================================================
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    residual = y_true - y_pred
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = np.nan if np.isclose(ss_tot, 0.0) else 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))

    mask_nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
        if np.any(mask_nonzero)
        else np.nan
    )

    denom = np.sum(y_true)
    pbias = np.nan if np.isclose(denom, 0.0) else 100 * np.sum(y_pred - y_true) / denom

    return r2, rmse, mae, mape, pbias


def format_metric_fixed(value, decimals=2):
    if not np.isfinite(value):
        return r"\mathrm{NaN}"
    return f"{value:.{decimals}f}"


def format_pbias_for_plot(value, decimals=2):
    if not np.isfinite(value):
        return r"\mathrm{NaN}"

    if value == 0:
        return "0.00"

    if abs(value) < 10 ** (-decimals):
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / (10 ** exponent)
        return rf"{coefficient:.2f} \times 10^{{{exponent}}}"

    return f"{value:.{decimals}f}"


# =========================================================
# GMDH MATRICES
# =========================================================
def build_single_input_design(x):
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([np.ones(len(x)), x, x ** 2])


def build_pair_quadratic_design(x1, x2):
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    return np.column_stack([np.ones(len(x1)), x1, x2, x1 ** 2, x2 ** 2, x1 * x2])


def fit_single_input_neuron(x_train, y_train, x_eval):
    X_train = build_single_input_design(x_train)
    coeffs, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_train_pred = X_train @ coeffs
    X_eval = build_single_input_design(x_eval)
    y_eval_pred = X_eval @ coeffs
    return coeffs, y_train_pred, y_eval_pred


def fit_pair_quadratic_neuron(x1_train, x2_train, y_train, x1_eval, x2_eval):
    X_train = build_pair_quadratic_design(x1_train, x2_train)
    coeffs, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_train_pred = X_train @ coeffs
    X_eval = build_pair_quadratic_design(x1_eval, x2_eval)
    y_eval_pred = X_eval @ coeffs
    return coeffs, y_train_pred, y_eval_pred


# =========================================================
# GMDH SOURCE TREE HELPERS
# =========================================================
def make_input_source(index, name):
    return {
        "kind": "input",
        "index": int(index),
        "name": str(name),
    }


def collect_base_input_names(source):
    if source["kind"] == "input":
        return [source["name"]]
    if source["kind"] == "single_neuron":
        return collect_base_input_names(source["child"])
    if source["kind"] == "pair_neuron":
        return collect_base_input_names(source["left"]) + collect_base_input_names(source["right"])
    return []


def evaluate_gmdh_source_on_row(source, row):
    kind = source["kind"]

    if kind == "input":
        return float(row[source["index"]])

    if kind == "single_neuron":
        x = evaluate_gmdh_source_on_row(source["child"], row)
        coeffs = np.asarray(source.get("coeffs_full", source["coeffs"]), dtype=float)
        X = np.array([1.0, x, x ** 2], dtype=float)
        return float(X @ coeffs)

    if kind == "pair_neuron":
        x1 = evaluate_gmdh_source_on_row(source["left"], row)
        x2 = evaluate_gmdh_source_on_row(source["right"], row)
        coeffs = np.asarray(source.get("coeffs_full", source["coeffs"]), dtype=float)
        X = np.array([1.0, x1, x2, x1 ** 2, x2 ** 2, x1 * x2], dtype=float)
        return float(X @ coeffs)

    raise ValueError(f"Unknown source kind: {kind}")


def refit_gmdh_source_on_full_data(source, X_full, y_full, cache=None):
    if cache is None:
        cache = {}

    sid = id(source)
    if sid in cache:
        return cache[sid]

    kind = source["kind"]

    if kind == "input":
        vals = X_full[:, source["index"]]
        cache[sid] = vals
        return vals

    if kind == "single_neuron":
        x = refit_gmdh_source_on_full_data(source["child"], X_full, y_full, cache)
        D = build_single_input_design(x)
        coeffs_full, _, _, _ = np.linalg.lstsq(D, y_full, rcond=None)
        source["coeffs_full"] = coeffs_full
        pred = D @ coeffs_full
        cache[sid] = pred
        return pred

    if kind == "pair_neuron":
        x1 = refit_gmdh_source_on_full_data(source["left"], X_full, y_full, cache)
        x2 = refit_gmdh_source_on_full_data(source["right"], X_full, y_full, cache)
        D = build_pair_quadratic_design(x1, x2)
        coeffs_full, _, _, _ = np.linalg.lstsq(D, y_full, rcond=None)
        source["coeffs_full"] = coeffs_full
        pred = D @ coeffs_full
        cache[sid] = pred
        return pred

    raise ValueError(f"Unknown source kind: {kind}")


def source_to_expression(source, symbol_map):
    kind = source["kind"]

    if kind == "input":
        return symbol_map.get(source["name"], source["name"])

    if kind == "single_neuron":
        child_expr = source_to_expression(source["child"], symbol_map)
        label = source.get("label", f"L{source.get('layer', '?')}N?")
        return f"{label}[{child_expr}]"

    if kind == "pair_neuron":
        left_expr = source_to_expression(source["left"], symbol_map)
        right_expr = source_to_expression(source["right"], symbol_map)
        label = source.get("label", f"L{source.get('layer', '?')}N?")
        return f"{label}[{left_expr}, {right_expr}]"

    return "UNKNOWN"


def build_model_structure_signature(model_dict, symbol_map):
    if not model_dict or "best_source" not in model_dict:
        return ""
    return source_to_expression(model_dict["best_source"], symbol_map)


# =========================================================
# GMDH REGRESSION
# =========================================================
def fit_gmdh_regression(df, predictors, target, min_rows_required=MIN_ROWS_REQUIRED):
    cols = (["date"] if "date" in df.columns else []) + list(predictors) + [target]
    sub = df[cols].dropna().copy()

    if len(sub) < min_rows_required:
        return None

    X_raw_full = sub[list(predictors)].values.astype(float)
    y_full = sub[target].values.astype(float)

    n = len(sub)
    n_val = max(GMDH_MIN_VALIDATION_ROWS, int(np.floor(n * GMDH_VALIDATION_FRACTION)))
    if n_val >= n:
        n_val = max(1, n // 3)

    n_train = n - n_val
    if n_val < 1:
        return None

    if STANDARDIZE_PREDICTORS:
        predictor_standardizer = fit_predictor_standardizer(
            X_raw_full,
            reference_rows=slice(0, n_train),
        )
        X_full = apply_predictor_standardizer(X_raw_full, predictor_standardizer)
    else:
        predictor_standardizer = {"enabled": False, "means": None, "stds": None}
        X_full = X_raw_full

    if len(predictors) == 1:
        if n_train < 3:
            return None

        x_train = X_full[:n_train, 0]
        y_train = y_full[:n_train]
        x_val = X_full[n_train:, 0]
        y_val = y_full[n_train:]

        coeffs, pred_train, pred_val = fit_single_input_neuron(x_train, y_train, x_val)
        val_r2, val_rmse, _, _, _ = calculate_metrics(y_val, pred_val)

        source = {
            "kind": "single_neuron",
            "layer": 1,
            "label": "L1N1",
            "child": make_input_source(0, predictors[0]),
            "coeffs": coeffs,
            "validation_r2": val_r2,
            "validation_rmse": val_rmse,
        }

        cache = {}
        y_pred_full = refit_gmdh_source_on_full_data(source, X_full, y_full, cache)
        final_coeffs = source.get("coeffs_full", source["coeffs"])
        r2, rmse, mae, mape, pbias = calculate_metrics(y_full, y_pred_full)
        used_base_inputs = unique_preserve_order(collect_base_input_names(source))

        return {
            "model_type": "gmdh",
            "predictors": predictors,
            "n_rows": len(sub),
            "dates": sub["date"].values if "date" in sub.columns else None,
            "y_true": y_full,
            "y_pred": y_pred_full,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "pbias": pbias,
            "best_source": source,
            "used_base_inputs": used_base_inputs,
            "predictor_standardizer": predictor_standardizer,
            "final_coefficients": final_coeffs,
            "gmdh_best_layer": 1,
            "gmdh_best_neuron_label": "L1N1",
            "validation_r2": val_r2,
            "validation_rmse": val_rmse,
            "history": [{
                "layer": 1,
                "n_candidates": 1,
                "best_validation_r2": val_r2,
                "best_validation_rmse": val_rmse,
                "best_label": "L1N1",
            }],
        }

    minimum_pair_train_rows = max(GMDH_MIN_PAIR_TRAIN_ROWS, len(predictors) + 1)
    if n_train < minimum_pair_train_rows:
        return None

    X_train = X_full[:n_train]
    y_train = y_full[:n_train]
    X_val = X_full[n_train:]
    y_val = y_full[n_train:]

    current_train = X_train.copy()
    current_val = X_val.copy()
    current_sources = [make_input_source(i, name) for i, name in enumerate(predictors)]

    best_source = None
    best_val_rmse = np.inf
    history = []

    for layer in range(1, GMDH_MAX_LAYERS + 1):
        if current_train.shape[1] < 2:
            break

        candidates = []

        for i, j in itertools.combinations(range(current_train.shape[1]), 2):
            coeffs, pred_train, pred_val = fit_pair_quadratic_neuron(
                current_train[:, i], current_train[:, j], y_train,
                current_val[:, i], current_val[:, j]
            )

            if not np.all(np.isfinite(pred_val)):
                continue

            val_r2, val_rmse, _, _, _ = calculate_metrics(y_val, pred_val)

            neuron = {
                "kind": "pair_neuron",
                "layer": layer,
                "left": current_sources[i],
                "right": current_sources[j],
                "coeffs": coeffs,
                "validation_r2": val_r2,
                "validation_rmse": val_rmse,
            }

            score_rmse = np.inf if not np.isfinite(val_rmse) else val_rmse
            score_r2 = -np.inf if not np.isfinite(val_r2) else val_r2
            candidates.append((score_rmse, -score_r2, neuron, pred_train, pred_val))

        if not candidates:
            break

        candidates.sort(key=lambda x: (x[0], x[1]))
        keep_n = min(GMDH_TOP_NEURONS_PER_LAYER, len(candidates))
        kept = candidates[:keep_n]

        next_sources = []
        next_train = []
        next_val = []

        for neuron_idx, (_, _, neuron, pred_train, pred_val) in enumerate(kept, start=1):
            neuron["label"] = f"L{layer}N{neuron_idx}"
            next_sources.append(neuron)
            next_train.append(pred_train)
            next_val.append(pred_val)

        layer_best = next_sources[0]
        history.append({
            "layer": layer,
            "n_candidates": len(candidates),
            "best_validation_r2": layer_best["validation_r2"],
            "best_validation_rmse": layer_best["validation_rmse"],
            "best_label": layer_best["label"],
        })

        if layer_best["validation_rmse"] + GMDH_MIN_IMPROVEMENT < best_val_rmse:
            best_val_rmse = layer_best["validation_rmse"]
            best_source = layer_best
        else:
            break

        current_sources = next_sources
        current_train = np.column_stack(next_train)
        current_val = np.column_stack(next_val)

    if best_source is None:
        return None

    cache = {}
    y_pred_full = refit_gmdh_source_on_full_data(best_source, X_full, y_full, cache)
    final_coeffs = best_source.get("coeffs_full", best_source["coeffs"])
    r2, rmse, mae, mape, pbias = calculate_metrics(y_full, y_pred_full)
    used_base_inputs = unique_preserve_order(collect_base_input_names(best_source))

    return {
        "model_type": "gmdh",
        "predictors": predictors,
        "n_rows": len(sub),
        "dates": sub["date"].values if "date" in sub.columns else None,
        "y_true": y_full,
        "y_pred": y_pred_full,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "pbias": pbias,
        "best_source": best_source,
        "used_base_inputs": used_base_inputs,
        "final_coefficients": final_coeffs,
        "gmdh_best_layer": best_source["layer"],
        "gmdh_best_neuron_label": best_source.get("label", ""),
        "validation_r2": best_source.get("validation_r2", np.nan),
        "validation_rmse": best_source.get("validation_rmse", np.nan),
        "history": history,
    }


# =========================================================
# PLOTTING
# =========================================================
def apply_journal_plot_style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": FIGURE_DPI,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5.5,
        "ytick.major.size": 5.5,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "axes.grid": False,
        "axes.formatter.useoffset": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def apply_plain_axis_format(ax):
    formatter_x = ScalarFormatter(useMathText=False)
    formatter_x.set_scientific(False)
    formatter_x.set_useOffset(False)

    formatter_y = ScalarFormatter(useMathText=False)
    formatter_y.set_scientific(False)
    formatter_y.set_useOffset(False)

    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)


def make_observed_vs_predicted_plot(y_true, y_pred, out_png, box_lines=None, point_color="black"):
    apply_journal_plot_style()

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    scale_factor = 10 ** AXIS_SCALE_POWER
    y_true_scaled = y_true / scale_factor
    y_pred_scaled = y_pred / scale_factor

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    ax.scatter(
        y_true_scaled, y_pred_scaled,
        s=42, alpha=0.90, color=point_color,
        edgecolors="black", linewidths=0.45, zorder=3,
    )

    all_vals = np.concatenate([y_true_scaled, y_pred_scaled])
    min_val = np.nanmin(all_vals)
    max_val = np.nanmax(all_vals)
    pad = 1.0 if np.isclose(min_val, max_val) else 0.05 * (max_val - min_val)
    lo = min_val - pad
    hi = max_val + pad

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="black", zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    if AXIS_SCALE_POWER == 0:
        scaled_unit_label = RUNOFF_UNIT_LABEL
    else:
        scaled_unit_label = rf"$10^{{{AXIS_SCALE_POWER}}}$ {RUNOFF_UNIT_LABEL}"

    ax.set_xlabel(rf"$Q_{{obs}}$ ({scaled_unit_label})", fontweight="bold")
    ax.set_ylabel(rf"$Q_{{pred}}$ ({scaled_unit_label})", fontweight="bold")

    apply_plain_axis_format(ax)
    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    if box_lines:
        ax.text(
            0.03, 0.97, "\n".join(box_lines),
            transform=ax.transAxes,
            ha="left", va="top", fontsize=10.0,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", linewidth=0.9, alpha=1.0),
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def build_metrics_box_lines(y_true, y_pred):
    # MAPE is intentionally removed from the plot box.
    # It is still calculated and saved in the Excel output tables.
    r2, rmse, _, _, pbias = calculate_metrics(y_true, y_pred)

    return [
        rf"$R^2 = {format_metric_fixed(r2)}$",
        rf"$RMSE = {format_metric_fixed(rmse)}\ \mathrm{{{RUNOFF_UNIT_LABEL}}}$",
        rf"$PBIAS = {format_pbias_for_plot(pbias)}\%$",
    ]


# =========================================================
# EXCEL HELPERS
# =========================================================
def autosize_excel_columns(writer, dataframe_dict):
    for sheet_name, df in dataframe_dict.items():
        worksheet = writer.sheets[sheet_name]

        for col_idx, col_name in enumerate(df.columns, start=1):
            max_len = len(str(col_name))
            for value in df[col_name]:
                text = "" if pd.isna(value) else str(value)
                if len(text) > max_len:
                    max_len = len(text)
            worksheet.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 60)

        worksheet.freeze_panes = "A2"


# =========================================================
# MODEL SETS
# =========================================================
def build_sensitivity_model_sets(
    predictor_names,
    run_full_model=True,
    run_two_parameter_combinations=True,
    run_leave_one_out=True,
):
    model_sets = []
    predictor_names = list(predictor_names)

    if run_two_parameter_combinations:
        for combo in itertools.combinations(predictor_names, 2):
            combo = list(combo)
            case_name = "COMB_" + "__".join(combo)
            model_sets.append({
                "scenario_type": "two_parameter_combination",
                "case_name": case_name,
                "predictors": combo,
                "removed_parameter": "None",
            })

    if run_full_model:
        model_sets.append({
            "scenario_type": "all_parameters",
            "case_name": "ALL_PARAMETERS",
            "predictors": predictor_names.copy(),
            "removed_parameter": "None",
        })

    if run_leave_one_out:
        for p in predictor_names:
            remaining = [x for x in predictor_names if x != p]
            if remaining:
                model_sets.append({
                    "scenario_type": "leave_one_out",
                    "case_name": f"REMOVE_{p}",
                    "predictors": remaining,
                    "removed_parameter": p,
                })

    return model_sets


def add_sensitivity_delta_columns(results_df):
    if results_df.empty:
        return results_df

    full_row = results_df[results_df["scenario_type"] == "all_parameters"]
    results_df = results_df.copy()

    if full_row.empty:
        results_df["delta_test_r2_vs_full"] = np.nan
        results_df["delta_test_rmse_vs_full"] = np.nan
        results_df["delta_test_mape_vs_full"] = np.nan
        results_df["delta_test_pbias_vs_full"] = np.nan
        results_df["delta_all_data_r2_vs_full"] = np.nan
        return results_df

    full_test_r2 = full_row.iloc[0]["test_r2"]
    full_test_rmse = full_row.iloc[0]["test_rmse"]
    full_test_mape = full_row.iloc[0]["test_mape_percent"]
    full_test_pbias = full_row.iloc[0]["test_pbias_percent"]
    full_all_r2 = full_row.iloc[0]["all_data_r2"]

    results_df["delta_test_r2_vs_full"] = results_df["test_r2"] - full_test_r2
    results_df["delta_test_rmse_vs_full"] = results_df["test_rmse"] - full_test_rmse
    results_df["delta_test_mape_vs_full"] = results_df["test_mape_percent"] - full_test_mape
    results_df["delta_test_pbias_vs_full"] = results_df["test_pbias_percent"] - full_test_pbias
    results_df["delta_all_data_r2_vs_full"] = results_df["all_data_r2"] - full_all_r2
    return results_df


# =========================================================
# READABLE EXCEL SUMMARY HELPERS
# =========================================================
def build_metrics_definitions_df():
    return pd.DataFrame([
        {
            "metric": "R2",
            "full_name": "Coefficient of determination",
            "meaning": "Shows how much of the observed runoff variability is explained by the model.",
            "best_direction": "Higher is better; values closer to 1 are better.",
            "how_to_use_in_this_file": "Use test_r2 first for model comparison. all_data_r2 can be optimistic because it uses all available data.",
        },
        {
            "metric": "RMSE",
            "full_name": "Root Mean Square Error",
            "meaning": "Measures the typical prediction error in the same unit as runoff.",
            "best_direction": "Lower is better.",
            "how_to_use_in_this_file": "Use test_rmse to compare predictive error. If RMSE increases after removing a parameter, that parameter is likely important.",
        },
        {
            "metric": "MAE",
            "full_name": "Mean Absolute Error",
            "meaning": "Average absolute difference between observed and predicted runoff.",
            "best_direction": "Lower is better.",
            "how_to_use_in_this_file": "Useful as a simple average error. Less sensitive to large errors than RMSE.",
        },
        {
            "metric": "MAPE",
            "full_name": "Mean Absolute Percentage Error",
            "meaning": "Average absolute error expressed as a percentage of observed runoff.",
            "best_direction": "Lower is better.",
            "how_to_use_in_this_file": "Saved in Excel only. It is not shown on plots. Be careful when observed runoff is very small.",
        },
        {
            "metric": "PBIAS",
            "full_name": "Percent Bias",
            "meaning": "Shows whether the model systematically overestimates or underestimates runoff.",
            "best_direction": "Closer to 0 is better. Positive means overestimation; negative means underestimation.",
            "how_to_use_in_this_file": "Use test_pbias_percent to check bias. A model with high R2 can still be biased.",
        },
        {
            "metric": "delta_test_r2_vs_full",
            "full_name": "Change in test R2 compared with the full model",
            "meaning": "For leave-one-out models, this shows what happens when one parameter is removed.",
            "best_direction": "Negative value means the model became worse after removing that parameter.",
            "how_to_use_in_this_file": "A large negative value indicates the removed parameter is important.",
        },
        {
            "metric": "delta_test_rmse_vs_full",
            "full_name": "Change in test RMSE compared with the full model",
            "meaning": "For leave-one-out models, this shows how the error changes after removing one parameter.",
            "best_direction": "Positive value means RMSE increased after removing that parameter.",
            "how_to_use_in_this_file": "A large positive value indicates the removed parameter is important.",
        },
        {
            "metric": "delta_test_pbias_vs_full",
            "full_name": "Change in test PBIAS compared with the full model",
            "meaning": "Shows whether removing a parameter makes the model more biased or less biased.",
            "best_direction": "Closer to 0 is better, but the sign also matters.",
            "how_to_use_in_this_file": "Use together with R2 and RMSE. Do not judge importance only from PBIAS.",
        },
    ])


def classify_model_quality(r2, rmse):
    if not np.isfinite(r2):
        return "Not available"
    if r2 >= 0.80:
        return "Very strong fit"
    if r2 >= 0.60:
        return "Good fit"
    if r2 >= 0.40:
        return "Moderate fit"
    if r2 >= 0.20:
        return "Weak fit"
    return "Poor fit"


def interpret_pbias(pbias):
    if not np.isfinite(pbias):
        return "PBIAS not available"
    if abs(pbias) < 5:
        return "Very low bias"
    if abs(pbias) < 10:
        return "Low bias"
    if pbias > 0:
        return "Overestimation bias"
    return "Underestimation bias"


def interpret_leave_one_out_row(row):
    d_r2 = row.get("delta_test_r2_vs_full", np.nan)
    d_rmse = row.get("delta_test_rmse_vs_full", np.nan)
    removed = row.get("removed_parameter", "Unknown")

    if not np.isfinite(d_r2) and not np.isfinite(d_rmse):
        return "Importance cannot be judged from available deltas."

    important_by_r2 = np.isfinite(d_r2) and d_r2 < 0
    important_by_rmse = np.isfinite(d_rmse) and d_rmse > 0

    if important_by_r2 and important_by_rmse:
        return f"Removing {removed} reduced test R2 and increased test RMSE; this parameter is likely important."
    if important_by_r2:
        return f"Removing {removed} reduced test R2; this parameter may be important."
    if important_by_rmse:
        return f"Removing {removed} increased test RMSE; this parameter may be important."
    return f"Removing {removed} did not clearly reduce performance compared with the full model."


def build_readable_summary_tables(results_df):
    summary = {}

    if results_df.empty:
        summary["READ_ME"] = pd.DataFrame([{
            "message": "No model results were available for this cluster."
        }])
        return summary

    df = results_df.copy()

    # Human-readable interpretation columns.
    df["test_model_quality"] = df.apply(
        lambda r: classify_model_quality(r.get("test_r2", np.nan), r.get("test_rmse", np.nan)),
        axis=1,
    )
    df["test_bias_interpretation"] = df["test_pbias_percent"].apply(interpret_pbias)
    df["absolute_test_pbias_percent"] = df["test_pbias_percent"].abs()

    readable_cols = [
        "cluster_sheet",
        "cluster_number",
        "scenario_type",
        "sensitivity_case",
        "removed_parameter",
        "tested_combination_abbreviations",
        "tested_combination",
        "test_r2",
        "test_rmse",
        "test_mae",
        "test_mape_percent",
        "test_pbias_percent",
        "absolute_test_pbias_percent",
        "test_model_quality",
        "test_bias_interpretation",
        "delta_test_r2_vs_full",
        "delta_test_rmse_vs_full",
        "delta_test_mape_vs_full",
        "delta_test_pbias_vs_full",
        "all_data_r2",
        "all_data_rmse",
        "train_r2",
        "train_rmse",
        "train_rows",
        "test_rows",
        "all_rows",
        "plot_color",
        "plot_folder",
        "train_structure_signature",
        "all_data_structure_signature",
    ]
    readable_cols = [c for c in readable_cols if c in df.columns]

    summary["READ_ME"] = pd.DataFrame([
        {
            "section": "How to read this Excel file",
            "explanation": "Start with BEST_MODELS_BY_TEST_R2 to see the best predictive models. Then check BEST_2_PARAMETER_COMBOS for the strongest two-variable combinations. Use PARAMETER_IMPORTANCE_LEAVE_ONE_OUT to see which parameter is important when removed.",
        },
        {
            "section": "Most important criterion",
            "explanation": "Use test_r2 and test_rmse as the main criteria because they evaluate the model on the testing period, not only on training or all data.",
        },
        {
            "section": "Parameter importance",
            "explanation": "In leave-one-out analysis, if removing a parameter decreases test_r2 and increases test_rmse, that removed parameter is likely important.",
        },
        {
            "section": "Best combination",
            "explanation": "In two-parameter combinations, the best pair is usually the one with highest test_r2, low test_rmse, and PBIAS close to zero.",
        },
        {
            "section": "Plots",
            "explanation": "Plots are unchanged. Cluster colors are fixed: 0 blue, 1 red, 2 green, 3 orange.",
        },
    ])

    summary["METRIC_DEFINITIONS"] = build_metrics_definitions_df()

    summary["RESULTS_READABLE"] = df[readable_cols].sort_values(
        ["scenario_type", "test_r2", "test_rmse"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    best_cols = readable_cols
    best_models = df.copy()
    best_models = best_models[np.isfinite(best_models["test_r2"])]
    if not best_models.empty:
        best_models = best_models.sort_values(
            ["test_r2", "test_rmse", "absolute_test_pbias_percent"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        best_models.insert(0, "rank_by_test_r2", np.arange(1, len(best_models) + 1))
        summary["BEST_MODELS_BY_TEST_R2"] = best_models[["rank_by_test_r2"] + best_cols].head(30)

    combos = df[df["scenario_type"] == "two_parameter_combination"].copy()
    combos = combos[np.isfinite(combos["test_r2"])]
    if not combos.empty:
        combos = combos.sort_values(
            ["test_r2", "test_rmse", "absolute_test_pbias_percent"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        combos.insert(0, "rank_best_two_parameter_combination", np.arange(1, len(combos) + 1))
        combos["why_this_is_good"] = combos.apply(
            lambda r: f"Test R2={r['test_r2']}, RMSE={r['test_rmse']}, PBIAS={r['test_pbias_percent']}%. Higher R2 and lower RMSE are preferred.",
            axis=1,
        )
        combo_cols = [
            "rank_best_two_parameter_combination",
            "tested_combination_abbreviations",
            "tested_combination",
            "test_r2",
            "test_rmse",
            "test_pbias_percent",
            "absolute_test_pbias_percent",
            "test_model_quality",
            "test_bias_interpretation",
            "why_this_is_good",
            "train_r2",
            "all_data_r2",
            "plot_folder",
        ]
        combo_cols = [c for c in combo_cols if c in combos.columns]
        summary["BEST_2_PARAMETER_COMBOS"] = combos[combo_cols].head(30)

    loo = df[df["scenario_type"] == "leave_one_out"].copy()
    if not loo.empty:
        loo["r2_loss_when_removed"] = -loo["delta_test_r2_vs_full"]
        loo["rmse_increase_when_removed"] = loo["delta_test_rmse_vs_full"]
        loo["parameter_importance_score"] = loo["r2_loss_when_removed"].fillna(0) + loo["rmse_increase_when_removed"].fillna(0)
        loo["importance_interpretation"] = loo.apply(interpret_leave_one_out_row, axis=1)
        loo = loo.sort_values(
            ["r2_loss_when_removed", "rmse_increase_when_removed"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        loo.insert(0, "rank_parameter_importance", np.arange(1, len(loo) + 1))

        loo_cols = [
            "rank_parameter_importance",
            "removed_parameter",
            "removed_parameter_abbreviation",
            "r2_loss_when_removed",
            "rmse_increase_when_removed",
            "delta_test_r2_vs_full",
            "delta_test_rmse_vs_full",
            "delta_test_pbias_vs_full",
            "test_r2",
            "test_rmse",
            "test_pbias_percent",
            "importance_interpretation",
            "tested_combination_abbreviations",
            "tested_combination",
            "plot_folder",
        ]
        loo_cols = [c for c in loo_cols if c in loo.columns]
        summary["PARAMETER_IMPORTANCE_LEAVE_ONE_OUT"] = loo[loo_cols]

    full = df[df["scenario_type"] == "all_parameters"].copy()
    if not full.empty:
        full["reference_explanation"] = "This is the full model. Leave-one-out deltas are compared against this row."
        full_cols = [
            "reference_explanation",
            "tested_combination_abbreviations",
            "tested_combination",
            "test_r2",
            "test_rmse",
            "test_mae",
            "test_mape_percent",
            "test_pbias_percent",
            "test_model_quality",
            "test_bias_interpretation",
            "train_r2",
            "all_data_r2",
            "plot_folder",
        ]
        full_cols = [c for c in full_cols if c in full.columns]
        summary["FULL_MODEL_REFERENCE"] = full[full_cols]

    scenario_summary = []
    for scenario_type, sub in df.groupby("scenario_type"):
        valid = sub[np.isfinite(sub["test_r2"])].copy()
        if valid.empty:
            continue
        best = valid.sort_values(
            ["test_r2", "test_rmse", "absolute_test_pbias_percent"],
            ascending=[False, True, True],
        ).iloc[0]
        scenario_summary.append({
            "scenario_type": scenario_type,
            "best_case": best.get("sensitivity_case", ""),
            "best_combination_abbreviations": best.get("tested_combination_abbreviations", ""),
            "removed_parameter": best.get("removed_parameter", "None"),
            "best_test_r2": best.get("test_r2", np.nan),
            "best_test_rmse": best.get("test_rmse", np.nan),
            "best_test_pbias_percent": best.get("test_pbias_percent", np.nan),
            "interpretation": "Best row for this scenario based on highest test R2, then lowest RMSE, then lowest absolute PBIAS.",
        })
    if scenario_summary:
        summary["BEST_BY_SCENARIO"] = pd.DataFrame(scenario_summary)

    return summary


# =========================================================
# PATHS
# =========================================================
def build_output_paths(main_input_folder: Path, cluster_sheet=None, cluster_number=None):
    base_output_folder = main_input_folder / OUTPUT_FOLDER_NAME

    # Group all subclusters from the same sheet together.
    # Example: k_2_clusters contains cluster 0 and cluster 1 in the same folder.
    if cluster_sheet is None:
        output_folder = base_output_folder
    else:
        output_folder = base_output_folder / safe_name(cluster_sheet)

    combination_plots_folder = output_folder / "combination_plots"
    leave_plots_folder = output_folder / "leave_plots"
    full_model_plots_folder = output_folder / "all_parameters_plots"

    if cluster_number is None:
        results_file_name = "gmdh_sensitivity_results.xlsx"
    else:
        results_file_name = f"gmdh_sensitivity_results__cluster_{int(cluster_number)}.xlsx"

    return {
        "output_folder": output_folder,
        "combination_plots_folder": combination_plots_folder,
        "leave_plots_folder": leave_plots_folder,
        "full_model_plots_folder": full_model_plots_folder,
        "results_excel_file": output_folder / results_file_name,
    }


def ensure_output_folders(paths):
    paths["output_folder"].mkdir(parents=True, exist_ok=True)
    paths["combination_plots_folder"].mkdir(parents=True, exist_ok=True)
    paths["leave_plots_folder"].mkdir(parents=True, exist_ok=True)
    paths["full_model_plots_folder"].mkdir(parents=True, exist_ok=True)


def get_case_plot_paths(base_plots_folder: Path):
    # Do not create one folder per case and do not create one folder per subcluster.
    # For example, for sheet k_2_clusters, cluster 0 and cluster 1 plots are saved together in:
    # k_2_clusters/combination_plots/training/
    # k_2_clusters/combination_plots/testing/
    # k_2_clusters/combination_plots/all_data/
    training_folder = base_plots_folder / "training"
    testing_folder = base_plots_folder / "testing"
    all_data_folder = base_plots_folder / "all_data"

    training_folder.mkdir(parents=True, exist_ok=True)
    testing_folder.mkdir(parents=True, exist_ok=True)
    all_data_folder.mkdir(parents=True, exist_ok=True)

    return {
        "case_folder": base_plots_folder,
        "training_folder": training_folder,
        "testing_folder": testing_folder,
        "all_data_folder": all_data_folder,
    }


# =========================================================
# PROCESS ONE CLUSTER
# =========================================================
def process_main_folder(
    input_folder: Path,
    cluster_sheet=None,
    cluster_number=None,
    cluster_label=None,
    point_ids_filter=None,
):
    paths = build_output_paths(
        input_folder,
        cluster_sheet=cluster_sheet,
        cluster_number=cluster_number,
    )
    ensure_output_folders(paths)

    combination_plots_folder = paths["combination_plots_folder"]
    leave_plots_folder = paths["leave_plots_folder"]
    full_model_plots_folder = paths["full_model_plots_folder"]
    results_excel_file = paths["results_excel_file"]

    excel_files = find_excel_files(input_folder)

    # Avoid reading the cluster Excel itself as input data if it is in the same folder.
    excel_files = [f for f in excel_files if f.resolve() != CLUSTER_EXCEL_FILE.resolve()]

    if not excel_files:
        raise FileNotFoundError(f"No Excel files found directly inside:\n{input_folder}")

    runoff_file = detect_runoff_file(excel_files)
    if runoff_file is None:
        raise FileNotFoundError(
            "Could not detect runoff file automatically.\n"
            f"Folder: {input_folder}\n"
            f"Rename the runoff file so its name contains one of: {RUNOFF_KEYWORDS}"
        )

    runoff_name = clean_parameter_name(runoff_file)
    runoff_df, _ = load_monthly_mean_from_excel(
        runoff_file,
        runoff_name,
        point_ids_filter=point_ids_filter,
    )
    runoff_df["runoff_mean"] = runoff_df[runoff_name]

    merged = runoff_df[["date", "runoff_mean"]].copy()

    used_files = []
    skipped_files = []
    predictor_names = []
    existing_names = set()

    for f in excel_files:
        if f == runoff_file:
            continue

        is_valid, reason = is_valid_parameter_file(f)
        if not is_valid:
            skipped_files.append((f.name, reason))
            continue

        param_name = clean_parameter_name(f, existing_names)
        existing_names.add(param_name)

        try:
            param_df, _ = load_monthly_mean_from_excel(
                f,
                param_name,
                point_ids_filter=point_ids_filter,
            )
            merged = pd.merge(merged, param_df, on="date", how="inner")
            predictor_names.append(param_name)
            used_files.append((f.name, param_name))
        except Exception as e:
            skipped_files.append((f.name, str(e)))

    if len(predictor_names) == 0:
        raise ValueError("No valid predictor files were found.")

    merged = merged.sort_values("date").reset_index(drop=True)
    parameter_symbol_map = build_parameter_symbol_map(predictor_names)

    model_sets = build_sensitivity_model_sets(
        predictor_names,
        run_full_model=RUN_FULL_MODEL,
        run_two_parameter_combinations=RUN_TWO_PARAMETER_COMBINATIONS,
        run_leave_one_out=RUN_LEAVE_ONE_OUT,
    )

    if not model_sets:
        raise ValueError("No sensitivity-analysis model sets could be generated.")

    results_rows = []
    history_rows = []
    skipped_cases = []

    for model_case in model_sets:
        combo = tuple(model_case["predictors"])
        removed_parameter = model_case["removed_parameter"]
        case_name = model_case["case_name"]
        scenario_type = model_case["scenario_type"]

        predictor_text = ", ".join(combo)
        abbr_text = ", ".join(parameter_symbol_map[p] for p in combo)

        if scenario_type == "two_parameter_combination":
            plot_root = combination_plots_folder
        elif scenario_type == "leave_one_out":
            plot_root = leave_plots_folder
        else:
            plot_root = full_model_plots_folder

        plot_paths = get_case_plot_paths(plot_root)

        sub = merged[["date", "runoff_mean"] + list(combo)].dropna().copy().sort_values("date").reset_index(drop=True)

        n_total = len(sub)
        if n_total < MIN_ROWS_REQUIRED:
            skipped_cases.append((case_name, f"only {n_total} common rows"))
            continue

        n_test = max(MIN_TEST_ROWS, int(np.floor(n_total * TEST_FRACTION)))
        if n_test >= n_total:
            n_test = max(1, n_total // 4)

        n_train_outer = n_total - n_test
        if n_train_outer < 5:
            skipped_cases.append((case_name, f"too few outer training rows: {n_train_outer}"))
            continue

        train_df = sub.iloc[:n_train_outer].copy().reset_index(drop=True)
        test_df = sub.iloc[n_train_outer:].copy().reset_index(drop=True)

        train_model = fit_gmdh_regression(
            train_df,
            predictors=combo,
            target="runoff_mean",
            min_rows_required=max(5, len(combo) + 2),
        )

        if train_model is None:
            skipped_cases.append((case_name, "GMDH training model could not be fitted"))
            continue

        X_test_raw = test_df[list(combo)].values.astype(float)
        X_test = apply_predictor_standardizer(
            X_test_raw,
            train_model.get("predictor_standardizer"),
        )
        y_test = test_df["runoff_mean"].values.astype(float)
        y_test_pred = np.array(
            [evaluate_gmdh_source_on_row(train_model["best_source"], row) for row in X_test],
            dtype=float,
        )

        test_r2, test_rmse, test_mae, test_mape, test_pbias = calculate_metrics(y_test, y_test_pred)

        all_model = fit_gmdh_regression(
            sub,
            predictors=combo,
            target="runoff_mean",
            min_rows_required=max(MIN_ROWS_REQUIRED, len(combo) + 2),
        )

        if all_model is None:
            skipped_cases.append((case_name, "GMDH all-data model could not be fitted"))
            continue

        cluster_tag = f"{safe_name(str(cluster_sheet))}__cluster_{int(cluster_number)}"
        cluster_color = get_cluster_plot_color(cluster_number)

        train_png = plot_paths["training_folder"] / f"{cluster_tag}__{case_name}__training_obs_vs_pred.png"
        test_png = plot_paths["testing_folder"] / f"{cluster_tag}__{case_name}__testing_obs_vs_pred.png"
        all_png = plot_paths["all_data_folder"] / f"{cluster_tag}__{case_name}__all_data_obs_vs_pred.png"

        if scenario_type == "leave_one_out":
            train_box = None
            test_box = None
            all_box = None
        else:
            train_box = build_metrics_box_lines(train_model["y_true"], train_model["y_pred"])
            test_box = build_metrics_box_lines(y_test, y_test_pred)
            all_box = build_metrics_box_lines(all_model["y_true"], all_model["y_pred"])

        make_observed_vs_predicted_plot(
            train_model["y_true"],
            train_model["y_pred"],
            train_png,
            train_box,
            point_color=cluster_color,
        )

        make_observed_vs_predicted_plot(
            y_test,
            y_test_pred,
            test_png,
            test_box,
            point_color=cluster_color,
        )

        make_observed_vs_predicted_plot(
            all_model["y_true"],
            all_model["y_pred"],
            all_png,
            all_box,
            point_color=cluster_color,
        )

        for h in train_model.get("history", []):
            history_rows.append({
                "cluster_sheet": cluster_sheet,
                "cluster_number": cluster_number,
                "cluster_label": cluster_label,
                "n_cluster_points": len(point_ids_filter) if point_ids_filter is not None else np.nan,
                "scenario_type": scenario_type,
                "sensitivity_case": case_name,
                "removed_parameter": removed_parameter,
                "predictors": predictor_text,
                "predictor_abbreviations": abbr_text,
                "model_scope": "training_model",
                "layer": h.get("layer"),
                "n_candidates": h.get("n_candidates"),
                "best_validation_r2": h.get("best_validation_r2"),
                "best_validation_rmse": h.get("best_validation_rmse"),
                "best_label": h.get("best_label"),
            })

        for h in all_model.get("history", []):
            history_rows.append({
                "cluster_sheet": cluster_sheet,
                "cluster_number": cluster_number,
                "cluster_label": cluster_label,
                "n_cluster_points": len(point_ids_filter) if point_ids_filter is not None else np.nan,
                "scenario_type": scenario_type,
                "sensitivity_case": case_name,
                "removed_parameter": removed_parameter,
                "predictors": predictor_text,
                "predictor_abbreviations": abbr_text,
                "model_scope": "all_data_model",
                "layer": h.get("layer"),
                "n_candidates": h.get("n_candidates"),
                "best_validation_r2": h.get("best_validation_r2"),
                "best_validation_rmse": h.get("best_validation_rmse"),
                "best_label": h.get("best_label"),
            })

        removed_abbr_text = parameter_symbol_map[removed_parameter] if removed_parameter in parameter_symbol_map else "None"
        train_structure_signature = build_model_structure_signature(train_model, parameter_symbol_map)
        all_data_structure_signature = build_model_structure_signature(all_model, parameter_symbol_map)

        results_rows.append({
            "cluster_sheet": cluster_sheet,
            "cluster_number": cluster_number,
            "cluster_label": cluster_label,
            "n_cluster_points": len(point_ids_filter) if point_ids_filter is not None else np.nan,
            "model_type": "gmdh",
            "scenario_type": scenario_type,
            "sensitivity_case": case_name,
            "removed_parameter": removed_parameter,
            "removed_parameter_abbreviation": removed_abbr_text,
            "n_input_parameters_used": len(combo),
            "tested_combination": predictor_text,
            "tested_combination_abbreviations": abbr_text,
            "train_r2": round(train_model["r2"], 4) if np.isfinite(train_model["r2"]) else np.nan,
            "train_rmse": round(train_model["rmse"], 4) if np.isfinite(train_model["rmse"]) else np.nan,
            "train_mae": round(train_model["mae"], 4) if np.isfinite(train_model["mae"]) else np.nan,
            "train_mape_percent": round(train_model["mape"], 4) if np.isfinite(train_model["mape"]) else np.nan,
            "train_pbias_percent": round(train_model["pbias"], 4) if np.isfinite(train_model["pbias"]) else np.nan,
            "test_r2": round(test_r2, 4) if np.isfinite(test_r2) else np.nan,
            "test_rmse": round(test_rmse, 4) if np.isfinite(test_rmse) else np.nan,
            "test_mae": round(test_mae, 4) if np.isfinite(test_mae) else np.nan,
            "test_mape_percent": round(test_mape, 4) if np.isfinite(test_mape) else np.nan,
            "test_pbias_percent": round(test_pbias, 4) if np.isfinite(test_pbias) else np.nan,
            "all_data_r2": round(all_model["r2"], 4) if np.isfinite(all_model["r2"]) else np.nan,
            "all_data_rmse": round(all_model["rmse"], 4) if np.isfinite(all_model["rmse"]) else np.nan,
            "all_data_mae": round(all_model["mae"], 4) if np.isfinite(all_model["mae"]) else np.nan,
            "all_data_mape_percent": round(all_model["mape"], 4) if np.isfinite(all_model["mape"]) else np.nan,
            "all_data_pbias_percent": round(all_model["pbias"], 4) if np.isfinite(all_model["pbias"]) else np.nan,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "all_rows": len(sub),
            "plot_folder": str(plot_paths["case_folder"]),
            "plot_color": cluster_color,
            "train_structure_signature": train_structure_signature,
            "all_data_structure_signature": all_data_structure_signature,
        })

    if not results_rows:
        raise ValueError("No GMDH sensitivity-analysis models could be fitted successfully.")

    results_df = pd.DataFrame(results_rows)

    sort_priority = {
        "two_parameter_combination": 1,
        "all_parameters": 2,
        "leave_one_out": 3,
    }

    results_df["scenario_order"] = results_df["scenario_type"].map(sort_priority)
    results_df = results_df.sort_values(
        ["cluster_sheet", "cluster_number", "scenario_order", "sensitivity_case"]
    ).reset_index(drop=True)

    results_df = add_sensitivity_delta_columns(results_df)

    history_df = pd.DataFrame(history_rows) if history_rows else pd.DataFrame()
    used_files_df = pd.DataFrame(used_files, columns=["file_name", "parameter_name"]) if used_files else pd.DataFrame()
    skipped_files_df = pd.DataFrame(skipped_files, columns=["file_name", "reason"]) if skipped_files else pd.DataFrame()
    skipped_cases_df = pd.DataFrame(skipped_cases, columns=["sensitivity_case", "reason"]) if skipped_cases else pd.DataFrame()

    readable_summary_sheets = build_readable_summary_tables(results_df)

    with pd.ExcelWriter(results_excel_file, engine="openpyxl") as writer:
        sheet_dict = {}

        for sheet_name, df_sheet in readable_summary_sheets.items():
            safe_sheet_name = sheet_name[:31]
            df_sheet.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            sheet_dict[safe_sheet_name] = df_sheet

        sheet_dict["gmdh_results_sorted"] = results_df
        results_df.to_excel(writer, sheet_name="gmdh_results_sorted", index=False)

        if not history_df.empty:
            history_df.to_excel(writer, sheet_name="gmdh_history", index=False)
            sheet_dict["gmdh_history"] = history_df

        if not used_files_df.empty:
            used_files_df.to_excel(writer, sheet_name="used_files", index=False)
            sheet_dict["used_files"] = used_files_df

        if not skipped_files_df.empty:
            skipped_files_df.to_excel(writer, sheet_name="skipped_files", index=False)
            sheet_dict["skipped_files"] = skipped_files_df

        if not skipped_cases_df.empty:
            skipped_cases_df.to_excel(writer, sheet_name="skipped_cases", index=False)
            sheet_dict["skipped_cases"] = skipped_cases_df

        autosize_excel_columns(writer, sheet_dict)

    print("\nDone for cluster.")
    print(f"Cluster sheet: {cluster_sheet}")
    print(f"Cluster number: {cluster_number}")
    print(f"Plot color: {get_cluster_plot_color(cluster_number)}")
    print(f"Results Excel: {results_excel_file}")

    return {
        "results_df": results_df,
        "history_df": history_df,
        "used_files_df": used_files_df,
        "skipped_files_df": skipped_files_df,
        "skipped_cases_df": skipped_cases_df,
        "results_excel_file": results_excel_file,
    }


# =========================================================
# SAVE MASTER EXCEL FOR ALL CLUSTERS
# =========================================================
def save_master_cluster_excel(
    output_file,
    all_results,
    all_history,
    all_skipped_cases,
    all_failed_clusters,
):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    history_df = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    skipped_cases_df = pd.concat(all_skipped_cases, ignore_index=True) if all_skipped_cases else pd.DataFrame()
    failed_clusters_df = pd.DataFrame(all_failed_clusters) if all_failed_clusters else pd.DataFrame()

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        sheet_dict = {}

        if not results_df.empty:
            results_df = results_df.sort_values(
                ["cluster_sheet", "cluster_number", "scenario_order", "sensitivity_case"]
            ).reset_index(drop=True)

            master_summary_sheets = build_readable_summary_tables(results_df)
            for sheet_name, df_sheet in master_summary_sheets.items():
                safe_sheet_name = ("M_" + sheet_name)[:31]
                df_sheet.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                sheet_dict[safe_sheet_name] = df_sheet

            results_df.to_excel(writer, sheet_name="ALL_CLUSTER_RESULTS", index=False)
            sheet_dict["ALL_CLUSTER_RESULTS"] = results_df

            best_test_df = results_df.copy()
            best_test_df = best_test_df[np.isfinite(best_test_df["test_r2"])]
            if not best_test_df.empty:
                best_test_df = best_test_df.sort_values(
                    ["cluster_sheet", "cluster_number", "test_r2"],
                    ascending=[True, True, False],
                )
                best_test_df = best_test_df.groupby(
                    ["cluster_sheet", "cluster_number"], as_index=False
                ).head(10)
                best_test_df.to_excel(writer, sheet_name="TOP_10_BY_TEST_R2", index=False)
                sheet_dict["TOP_10_BY_TEST_R2"] = best_test_df

            best_all_df = results_df.copy()
            best_all_df = best_all_df[np.isfinite(best_all_df["all_data_r2"])]
            if not best_all_df.empty:
                best_all_df = best_all_df.sort_values(
                    ["cluster_sheet", "cluster_number", "all_data_r2"],
                    ascending=[True, True, False],
                )
                best_all_df = best_all_df.groupby(
                    ["cluster_sheet", "cluster_number"], as_index=False
                ).head(10)
                best_all_df.to_excel(writer, sheet_name="TOP_10_BY_ALL_DATA_R2", index=False)
                sheet_dict["TOP_10_BY_ALL_DATA_R2"] = best_all_df

        if not history_df.empty:
            history_df.to_excel(writer, sheet_name="ALL_GMDH_HISTORY", index=False)
            sheet_dict["ALL_GMDH_HISTORY"] = history_df

        if not skipped_cases_df.empty:
            skipped_cases_df.to_excel(writer, sheet_name="ALL_SKIPPED_CASES", index=False)
            sheet_dict["ALL_SKIPPED_CASES"] = skipped_cases_df

        if not failed_clusters_df.empty:
            failed_clusters_df.to_excel(writer, sheet_name="FAILED_CLUSTERS", index=False)
            sheet_dict["FAILED_CLUSTERS"] = failed_clusters_df

        if sheet_dict:
            autosize_excel_columns(writer, sheet_dict)

    print("\nMaster Excel file saved:")
    print(output_file)


# =========================================================
# MAIN
# =========================================================
def main():
    if not MAIN_INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Main input folder does not exist:\n{MAIN_INPUT_FOLDER}")

    cluster_jobs = load_cluster_definitions(CLUSTER_EXCEL_FILE)

    print(f"\nFound {len(cluster_jobs)} cluster groups to process.")

    all_results = []
    all_history = []
    all_skipped_cases = []
    failed_clusters = []

    for job in cluster_jobs:
        print("\n" + "=" * 90)
        print(f"Processing sheet: {job['cluster_sheet']}")
        print(f"Processing cluster number: {job['cluster_number']}")
        print(f"Number of points: {job['n_points']}")
        print(f"Plot color: {get_cluster_plot_color(job['cluster_number'])}")
        print("=" * 90)

        try:
            out = process_main_folder(
                MAIN_INPUT_FOLDER,
                cluster_sheet=job["cluster_sheet"],
                cluster_number=job["cluster_number"],
                cluster_label=job["cluster_label"],
                point_ids_filter=job["point_ids"],
            )

            if not out["results_df"].empty:
                all_results.append(out["results_df"])

            if not out["history_df"].empty:
                all_history.append(out["history_df"])

            if not out["skipped_cases_df"].empty:
                skipped_tmp = out["skipped_cases_df"].copy()
                skipped_tmp.insert(0, "cluster_sheet", job["cluster_sheet"])
                skipped_tmp.insert(1, "cluster_number", job["cluster_number"])
                skipped_tmp.insert(2, "cluster_label", job["cluster_label"])
                skipped_tmp.insert(3, "n_cluster_points", job["n_points"])
                all_skipped_cases.append(skipped_tmp)

        except Exception as e:
            failed_clusters.append({
                "cluster_sheet": job["cluster_sheet"],
                "cluster_number": job["cluster_number"],
                "cluster_label": job["cluster_label"],
                "n_cluster_points": job["n_points"],
                "plot_color": get_cluster_plot_color(job["cluster_number"]),
                "reason": str(e),
            })
            print(f"FAILED for {job['cluster_label']}: {e}")

    master_excel_file = MAIN_INPUT_FOLDER / OUTPUT_FOLDER_NAME / "ALL_CLUSTER_GMDH_RESULTS.xlsx"

    save_master_cluster_excel(
        output_file=master_excel_file,
        all_results=all_results,
        all_history=all_history,
        all_skipped_cases=all_skipped_cases,
        all_failed_clusters=failed_clusters,
    )

    print("\nAll cluster processing finished.")


if __name__ == "__main__":
    main()