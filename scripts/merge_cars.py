import pandas as pd
from rapidfuzz import process, fuzz

epa = pd.read_csv('./data/processed/cars_epa.csv')
cd  = pd.read_csv('./data/processed/cars_cd.csv')

# Extract just the car name from EPA's model field for matching
epa['match_key'] = (epa['make'] + ' ' + epa['model']).str.lower().str.strip()
cd['match_key']  = cd['model'].str.lower().str.strip()

cd_keys = cd['match_key'].tolist()

def fuzzy_cd(key):
    result = process.extractOne(key, cd_keys, scorer=fuzz.token_sort_ratio, score_cutoff=75)
    if result:
        return cd.loc[cd['match_key'] == result[0], 'cd'].values[0]
    return None

print("Fuzzy matching EPA models to Cd values...")
epa['cd'] = epa['match_key'].apply(fuzzy_cd)

coverage = epa['cd'].notna().mean() * 100
print(f"Cd coverage: {coverage:.1f}% of {len(epa):,} rows")
print(f"Matched rows: {epa['cd'].notna().sum():,}")
print(f"Sample matches:")
print(epa[epa['cd'].notna()][['make','model','cd']].drop_duplicates().head(10).to_string())

epa.drop(columns=['match_key'], inplace=True)
epa.to_csv('./data/processed/cars_unified.csv', index=False)
print(f"\nSaved to ./data/processed/cars_unified.csv")
