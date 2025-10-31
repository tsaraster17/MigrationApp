import pandas as pd
import os

input_file = r"C:\Users\Tsar Aster17\PycharmProjects\MigrationPredictor\cleaned\employment_clean.csv"
clean_folder = r"C:\Users\Tsar Aster17\PycharmProjects\MigrationPredictor\cleaned"

print("Loading employment_clean.csv...")
df = pd.read_csv(input_file)

# Convert Code column → ISO3 uppercase
df['ISO3'] = df['Code'].astype(str).str.upper().str.strip()
df['year'] = pd.to_numeric(df['Year'], errors='coerce')

# Columns we will convert
mapping = {
    "emp_services_clean.csv": "Number of people employed in services (Herrendorf et al. data)",
    "emp_industry_clean.csv": "Number of people employed in industry  (Herrendorf et al. data)",
    "emp_agri_clean.csv": "number_employed_agri"
}

for out_name, col in mapping.items():
    if col not in df.columns:
        print(f"⚠ Column missing in employment CSV: {col}")
        continue

    sub = df[['ISO3','year',col]].copy()
    sub = sub.dropna(subset=['ISO3','year'])
    sub = sub.rename(columns={col: 'value'})
    # Remove commas and convert to numeric
    sub['value'] = sub['value'].astype(str).str.replace(',', '', regex=False)
    sub['value'] = pd.to_numeric(sub['value'], errors='coerce')
    sub = sub.dropna(subset=['value'])

    out_file = os.path.join(clean_folder, out_name)
    sub.to_csv(out_file, index=False)
    print(f"✅ Saved {out_file} — rows: {len(sub)}")

print("✅ Employment split complete.")
