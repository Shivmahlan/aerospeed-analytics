import io, re, requests, pandas as pd
from bs4 import BeautifulSoup

HEADERS = {'User-Agent': 'Mozilla/5.0'}
url = 'https://en.wikipedia.org/wiki/Automobile_drag_coefficient'
r = requests.get(url, headers=HEADERS, timeout=30)
soup = BeautifulSoup(r.text, 'html.parser')
tables = soup.find_all('table', {'class': 'wikitable'})

for i in [0, 1]:
    df = pd.read_html(io.StringIO(str(tables[i])))[0]
    print(f'Table {i}:')
    print(df[['Automobile','Cd']].to_string())
    print()
