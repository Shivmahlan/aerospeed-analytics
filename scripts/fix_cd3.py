import io, re, requests, pandas as pd
from bs4 import BeautifulSoup

HEADERS = {'User-Agent': 'Mozilla/5.0'}
url = 'https://en.wikipedia.org/wiki/Automobile_drag_coefficient'
r = requests.get(url, headers=HEADERS, timeout=30)
soup = BeautifulSoup(r.text, 'html.parser')
tables = soup.find_all('table', {'class': 'wikitable'})

all_frames = []
for i in [0, 1]:
    df = pd.read_html(io.StringIO(str(tables[i])))[0]
    df = df.rename(columns={'Automobile': 'model', 'Calendar year': 'year'})
    # Strip citation brackets e.g. 0.48[7][8] -> 0.48
    df['cd'] = df['Cd'].astype(str).str.replace(r'\[.*?\]', '', regex=True)
    df['cd'] = df['cd'].str.extract(r'([\d.]+)').astype(float)
    df = df.dropna(subset=['cd'])
    df = df[(df['cd'] > 0.1) & (df['cd'] < 1.5)]
    df = df[['model', 'cd']].reset_index(drop=True)
    all_frames.append(df)
    print(f'Table {i}: {len(df)} rows OK')

df = pd.concat(all_frames, ignore_index=True)
df.to_csv('./data/processed/cars_cd.csv', index=False)
print(f'\nSaved {len(df)} rows')
print(f'Cd range: {df["cd"].min():.3f} to {df["cd"].max():.3f}')
print(df.to_string())
