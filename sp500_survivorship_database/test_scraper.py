import requests
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}
response = requests.get(url, headers=headers)
tables = pd.read_html(response.text)
print(f"Number of tables found: {len(tables)}")

df_current = tables[0]
print("Current constituents snippet:")
print(df_current.head())
print("Columns:", df_current.columns.tolist())

df_changes = tables[1]
print("\nHistorical changes snippet:")
print(df_changes.head())
print("Columns:", df_changes.columns.tolist())
