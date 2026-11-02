import numpy as np
import pandas as pd

# Load all saved files
df_original = pd.read_csv("log_2025-10-09_58-33-00.csv")
df_full = pd.read_csv("full_flightlog.csv")
df_clean = pd.read_csv("clean_cycles.csv")
df_clean_norm = pd.read_csv("normalized_flightlog_clean_cycles.csv")
df_full_norm = pd.read_csv("normalized_flightlog_full_operation.csv")
max_vals_clean = pd.read_csv("normalization_values_clean_cycles.csv", index_col=0)
max_vals_full = pd.read_csv("normalization_values_full_operation.csv", index_col=0)

# --- 1. Row counts ---
print("=== Row counts ===")
print(f"Original:       {len(df_original)}")
print(f"Full log:       {len(df_full)}")
print(f"Clean cycles:   {len(df_clean)}")
print(f"Clean norm:     {len(df_clean_norm)}")
print(f"Full norm:      {len(df_full_norm)}")

# --- 2. Check full log has same number of rows as original ---
assert len(df_full) == len(df_original), "Full log row count mismatch!"
print("\nFull log row count matches original: OK")

# --- 3. Check clean cycles are a subset of full log ---
assert len(df_clean) < len(df_full), "Clean cycles should have fewer rows than full log"
print(f"Clean cycles ({len(df_clean)}) is a subset of full log ({len(df_full)}): OK")

# --- 4. Check no unknown flight phases in clean cycles ---
assert df_clean["flight_phase_index"].isna().sum() == 0, "Clean cycles contain unknown flight phases!"
print("No unknown flight phases in clean cycles: OK")

# --- 5. Check tether force conversion ---
force_check = np.allclose(
    df_full["ground_tether_force_N"],
    df_original["ground_tether_force"] * 9.80665,
    equal_nan=True
)
assert force_check, "Tether force conversion mismatch!"
print("Tether force conversion (kg -> N): OK")

# --- 6. Check calculated power ---
power_check = np.allclose(
    df_full["ground_tether_power_calc"],
    df_original["ground_tether_force"] * 9.80665 * df_original["ground_tether_reelout_speed"],
    equal_nan=True
)
assert power_check, "Calculated tether power mismatch!"
print("Calculated tether power (F*v): OK")

# --- 7. Check normalized values are in [-1, 1] ---
numeric_cols = df_clean_norm.select_dtypes(include=np.number).columns
cols_not_normalized = ["time", "cycle", "flight_phase_index"]
cols_to_check = numeric_cols.difference(cols_not_normalized)

max_clean = df_clean_norm[cols_to_check].abs().max()
assert (max_clean <= 1.0 + 1e-9).all(), f"Clean normalized values exceed 1:\n{max_clean[max_clean > 1]}"
print("Clean normalized values in [-1, 1]: OK")

max_full = df_full_norm[cols_to_check].abs().max()
assert (max_full <= 1.0 + 1e-9).all(), f"Full normalized values exceed 1:\n{max_full[max_full > 1]}"
print("Full normalized values in [-1, 1]: OK")

# --- 8. Check NaNs ---
print("\n=== NaN counts ===")
for name, data in [("full log", df_full), ("clean cycles", df_clean)]:
    nans = data.isna().sum()
    nans = nans[nans > 0]
    if len(nans) == 0:
        print(f"{name}: no NaNs")
    else:
        print(f"{name}:\n{nans}")

# --- 9. Quick summary statistics for key signals ---
print("\n=== Key signal ranges ===")
for col in ["ground_tether_force_N", "ground_tether_reelout_speed", "ground_mech_power"]:
    print(f"\n{col}:")
    print(f"  Full log:     min={df_full[col].min():.2f}, max={df_full[col].max():.2f}")
    print(f"  Clean cycles: min={df_clean[col].min():.2f}, max={df_clean[col].max():.2f}")