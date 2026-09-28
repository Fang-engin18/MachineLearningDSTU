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

