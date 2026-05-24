from pathlib import Path
CONFIG_DIR = Path(__file__).resolve().parent

JURISDICTIONS = ["US", "EU"]
MODEL_TYPES = [
    "Logistic Regression",
    "XGBoost",
    "FNN",
]
FAIRNESS_GROUPS = ["age_group", "income_group", "synthetic_protected_group"]
DATASET_NAME = "Combined benchmark governance dataset"
DEFAULT_RANDOM_SEED = 42
DEFAULT_DECISION_THRESHOLD = 0.50


def load_simple_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- ") and current_list_key:
            data[current_list_key].append(line[2:].strip())
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            data[key] = []
            current_list_key = key
            continue

        current_list_key = None
        data[key] = parse_scalar(value)

    return data


def parse_scalar(value: str) -> object:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config_file(filename: str) -> dict[str, object]:
    return load_simple_yaml(CONFIG_DIR / filename)


def load_model_profiles(filename: str = "model_profiles.yaml") -> dict[str, dict[str, str]]:
    path = CONFIG_DIR / filename
    profiles: dict[str, dict[str, str]] = {}
    current_model: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        if not raw_line.startswith(" ") and raw_line.rstrip().endswith(":"):
            current_model = raw_line.strip()[:-1]
            profiles[current_model] = {}
            continue

        if current_model and ":" in raw_line:
            key, value = raw_line.strip().split(":", 1)
            profiles[current_model][key.strip()] = value.strip()

    return profiles


THRESHOLDS = load_config_file("thresholds.yaml")
MODEL_PROFILES = load_model_profiles()


def get_threshold(name: str) -> float:
    return float(THRESHOLDS[name])


def get_config_value(name: str) -> object:
    return THRESHOLDS[name]


def get_model_profile(model_type: str) -> dict[str, str]:
    if model_type not in MODEL_PROFILES:
        raise ValueError(f"Unsupported model profile: {model_type}")
    return MODEL_PROFILES[model_type]
