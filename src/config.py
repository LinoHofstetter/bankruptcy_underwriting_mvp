"""Project configuration and financial-ratio metadata."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "5year.arff"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "model.joblib"
CALIBRATED_MODEL_PATH = MODELS_DIR / "calibrated_model.joblib"
IMPUTER_PATH = MODELS_DIR / "imputer.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
CALIBRATION_CURVE_PATH = REPORTS_DIR / "calibration_curve.csv"

TARGET_COLUMN = "target"
ROW_ID_COLUMN = "row_id"
FEATURE_NAMES = [f"X{i}" for i in range(1, 65)]

FEATURE_DESCRIPTIONS = {
    "X1": "net profit / total assets",
    "X2": "total liabilities / total assets",
    "X3": "working capital / total assets",
    "X4": "current assets / short-term liabilities",
    "X5": "[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365",
    "X6": "retained earnings / total assets",
    "X7": "EBIT / total assets",
    "X8": "book value of equity / total liabilities",
    "X9": "sales / total assets",
    "X10": "equity / total assets",
    "X11": "(gross profit + extraordinary items + financial expenses) / total assets",
    "X12": "gross profit / short-term liabilities",
    "X13": "(gross profit + depreciation) / sales",
    "X14": "(gross profit + interest) / total assets",
    "X15": "(total liabilities * 365) / (gross profit + depreciation)",
    "X16": "(gross profit + depreciation) / total liabilities",
    "X17": "total assets / total liabilities",
    "X18": "gross profit / total assets",
    "X19": "gross profit / sales",
    "X20": "(inventory * 365) / sales",
    "X21": "sales n / sales n-1",
    "X22": "profit on operating activities / total assets",
    "X23": "net profit / sales",
    "X24": "gross profit in 3 years / total assets",
    "X25": "(equity - share capital) / total assets",
    "X26": "(net profit + depreciation) / total liabilities",
    "X27": "profit on operating activities / financial expenses",
    "X28": "working capital / fixed assets",
    "X29": "logarithm of total assets",
    "X30": "(total liabilities - cash) / sales",
    "X31": "(gross profit + interest) / sales",
    "X32": "(current liabilities * 365) / cost of products sold",
    "X33": "operating expenses / short-term liabilities",
    "X34": "operating expenses / total liabilities",
    "X35": "profit on sales / total assets",
    "X36": "total sales / total assets",
    "X37": "(current assets - inventories) / long-term liabilities",
    "X38": "constant capital / total assets",
    "X39": "profit on sales / sales",
    "X40": "(current assets - inventory - receivables) / short-term liabilities",
    "X41": "total liabilities / ((profit on operating activities + depreciation) * (12/365))",
    "X42": "profit on operating activities / sales",
    "X43": "rotation receivables + inventory turnover in days",
    "X44": "(receivables * 365) / sales",
    "X45": "net profit / inventory",
    "X46": "(current assets - inventory) / short-term liabilities",
    "X47": "(inventory * 365) / cost of products sold",
    "X48": "EBITDA / total assets",
    "X49": "EBITDA / sales",
    "X50": "current assets / total liabilities",
    "X51": "short-term liabilities / total assets",
    "X52": "(short-term liabilities * 365) / cost of products sold",
    "X53": "equity / fixed assets",
    "X54": "constant capital / fixed assets",
    "X55": "working capital",
    "X56": "(sales - cost of products sold) / sales",
    "X57": "(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)",
    "X58": "total costs / total sales",
    "X59": "long-term liabilities / equity",
    "X60": "sales / inventory",
    "X61": "sales / receivables",
    "X62": "(short-term liabilities * 365) / sales",
    "X63": "sales / short-term liabilities",
    "X64": "sales / fixed assets",
}


def ensure_output_directories() -> None:
    """Create all directories used for generated data and artifacts."""
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
