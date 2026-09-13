import pandas as pd
import os

files_to_check = {
    "freight_base_data.csv": "Date",
    "usdinr.csv": "Date",
    "bdry.csv": "Date",
    "sea.csv": "Date",
    "vix.csv": "Date",
    "wti.csv": "Date",
    "sblk.csv": "Date",
    "port_master.csv": None,
    "route_data.csv": None,
    "vessel_master.csv": None,
    "tax_tariff_data.csv": None,
}

print("=" * 70)
print("FRIGHTSIGHT — FULL DATA QUALITY REPORT")
print("=" * 70)

for filename, date_col in files_to_check.items():
    path = f"data/{filename}"
    print(f"\n📄 {filename}")
    print("-" * 50)

    if not os.path.exists(path):
        print("  ❌ FILE NOT FOUND")
        continue

    try:
        if date_col:
            df = pd.read_csv(path, parse_dates=[date_col])
        else:
            df = pd.read_csv(path)

        print(f"  Shape: {df.shape}")
        missing = df.isnull().sum().sum()
        print(f"  Missing values: {missing}")
        print(f"  Duplicate rows: {df.duplicated().sum()}")

        if date_col:
            print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")

        if missing == 0 and df.duplicated().sum() == 0:
            print("  ✅ CLEAN")
        else:
            print("  ⚠️  NEEDS ATTENTION")

    except Exception as e:
        print(f"  ❌ ERROR reading file: {e}")

print("\n" + "=" * 70)
print("Report complete.")
print("=" * 70)