import pandas as pd
def check_data_quality(df: pd.DataFrame) -> dict:
    """Проверяет пустые ячейки, некорректные данные, типы."""
    report = {
        "missing": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "non_numeric": {}
    }
    for col in df.columns:
        if df[col].dtype == object:
            # пробуем привести к числу
            converted = pd.to_numeric(df[col], errors="coerce")
            bad = df[col][converted.isna() & df[col].notna()]
            if not bad.empty:
                report["non_numeric"][col] = bad.tolist()[:10]
    return report

def show_table(df: pd.DataFrame, n: int = 10) -> None:
    print(df.head(n))
    print(f"\nShape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")

def extract_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Возвращает новый DataFrame с нужными столбцами."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")
    return df[cols].copy()   # копия — исходный df не меняется

def cast_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """type_map: {'col': 'float64', ...}"""
    df = df.copy()
    for col, dtype in type_map.items():
        df[col] = df[col].astype(dtype)
    return df
