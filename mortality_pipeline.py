import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


# ========== CONFIG ==========
PATH_2004_2017 = "Data/Multiple Cause of Death, 2004-2017.csv"
PATH_2018_2023 = "Data/Multiple Cause of Death, 2018-2023.csv"
OUTPUT_PATH = "Data/clean_mortality_2004_2023.csv"


# ========== HELPER FUNCTIONS ==========

def parse_age_min(group: str):
    """Extract minimum age from diverse CDC formats."""
    if pd.isna(group):
        return np.nan
    s = str(group).lower()

    # Case: "Under 1 year"
    if "under" in s:
        return 0
    
    # Case: "Not Stated", "Unknown", etc.
    if "not" in s or "unknown" in s:
        return np.nan
    
    # Case: "85+ years"
    m_plus = re.search(r"(\d+)\+", s)
    if m_plus:
        return int(m_plus.group(1))

    # Case: ranges "15-24 years", "5–14 years" (en-dash)
    m_range = re.search(r"(\d+)[\-\–](\d+)", s)
    if m_range:
        return int(m_range.group(1))

    # Last fallback: first number in string
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan



def parse_age_max(group: str):
    if pd.isna(group):
        return np.nan
    s = str(group).lower()

    if "under" in s:
        return 1
    
    if "not" in s or "unknown" in s:
        return np.nan
    
    m_range = re.search(r"(\d+)[\-\–](\d+)", s)
    if m_range:
        return int(m_range.group(2))

    m_plus = re.search(r"(\d+)\+", s)
    if m_plus:
        return int(m_plus.group(1)) + 9  # approx band

    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan



def load_raw(path_2004_2017: str, path_2018_2023: str):
    """Load the two raw CSVs and standardize race column names."""
    df1 = pd.read_csv(path_2004_2017)
    df2 = pd.read_csv(path_2018_2023)

    # Unify race column name (Option A: keep as provided)
    if "Single Race 6" in df2.columns:
        df2 = df2.rename(columns={"Single Race 6": "Race"})
        # We drop its code later as redundant

    # Add source flags (useful for debugging)
    df1["Source_File"] = "2004-2017"
    df2["Source_File"] = "2018-2023"

    return df1, df2


def clean_mortality(df: pd.DataFrame) -> pd.DataFrame:
    """Core cleaning + feature engineering pipeline."""
    df = df.copy()

    # 1. Remove redundant columns if they exist
    redundant_cols = [
        "State Code",
        "Year Code",
        "Ten-Year Age Groups Code",
        "Race Code",
        "Sex Code",
        "Single Race 6 Code",  # in the 2018–2023 file
    ]
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns], errors="ignore")

    # 2. Flag unreliable crude rate BEFORE conversion
    if "Crude Rate" in df.columns:
        df["Unreliable_Flag"] = df["Crude Rate"].astype(str).eq("Unreliable").astype(int)
    else:
        df["Unreliable_Flag"] = 0

    # 3. Convert numeric columns
    for col in ["Year", "Deaths", "Population"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Convert crude rate to numeric and store as new column
    if "Crude Rate" in df.columns:
        df["CrudeRate_Reported"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
        df = df.drop(columns=["Crude Rate"])
    else:
        df["CrudeRate_Reported"] = np.nan

    # 5. Clean Sex labels
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].replace({"M": "Male", "F": "Female"})

    # 6. Parse age bands into numeric
    if "Ten-Year Age Groups" in df.columns:
        df["Age_Min"] = df["Ten-Year Age Groups"].apply(parse_age_min)
        df["Age_Max"] = df["Ten-Year Age Groups"].apply(parse_age_max)
        df["Age_Mid"] = (df["Age_Min"] + df["Age_Max"]) / 2

    # 7. Calculate crude rate per 100k from deaths & population
    df["CrudeRate_Calculated"] = (df["Deaths"] / df["Population"]) * 100000
    df["CrudeRate_Calculated"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # 8. Basic sanity filters
    df = df[(df["Population"] > 0) & (df["Deaths"] >= 0)]

    return df



# ========== MAIN SCRIPT ==========

def main():
    # 1. Load raw CSVs
    df1_raw, df2_raw = load_raw(PATH_2004_2017, PATH_2018_2023)

    # 2. Clean each separately
    clean1 = clean_mortality(df1_raw)
    clean2 = clean_mortality(df2_raw)

    # 3. Merge into a single tidy dataset
    merged_clean = pd.concat([clean1, clean2], ignore_index=True)

    # Optional: sort for neatness
    merged_clean = merged_clean.sort_values(
        by=["State", "Year", "Age_Min", "Sex", "Race"],
        kind="mergesort"
    ).reset_index(drop=True)

    # 4. Save to CSV
    merged_clean.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cleaned data to: {OUTPUT_PATH}")
    print("Cleaned shape:", merged_clean.shape)
    print(merged_clean.head())


if __name__ == "__main__":
    main()
