import sqlite3

conn_yf = sqlite3.connect('dataset/sp500_yF.db')
conn_tg = sqlite3.connect('dataset/sp500_Tiingo.db')

yf_rows = conn_yf.execute('SELECT Symbol, Security, GICS_Sector FROM Constituents').fetchall()

cursor = conn_tg.cursor()
cursor.executemany('''
    INSERT OR IGNORE INTO Constituents (Symbol, Security, GICS_Sector)
    VALUES (?, ?, ?)
''', yf_rows)
conn_tg.commit()

manual_sectors = {
    'FB': ('Meta Platforms (FB)', 'Communication Services'),
    'COHR': ('Coherent Corp', 'Information Technology'),
    'LITE': ('Lumentum Holdings', 'Information Technology'),
    'VEEV': ('Veeva Systems', 'Health Care'),
    'MRVL': ('Marvell Technology', 'Information Technology'),
    'ACT': ('Actavis', 'Health Care'),
    'VRT': ('Vertiv Holdings', 'Industrials'),
    'DLPH': ('Delphi Automotive', 'Consumer Discretionary'),
    'BUD': ('Anheuser-Busch InBev', 'Consumer Staples'),
    'FLEX': ('Flex Ltd.', 'Information Technology'),
    'HRS': ('Harris Corp', 'Industrials'),
    'SATS': ('EchoStar', 'Information Technology'),
    'CASY': ('Casey\'s General Stores', 'Consumer Staples'),
    'KORS': ('Michael Kors', 'Consumer Discretionary'),
    'LUK': ('Leucadia National', 'Financials'),
    'CCR': ('Consol Coal Resources', 'Energy'),
    'JEC': ('Jacobs Engineering', 'Industrials'),
    'FSR': ('Fisker', 'Consumer Discretionary')
}

for sym, (sec_name, gics) in manual_sectors.items():
    cursor.execute('''
        INSERT OR REPLACE INTO Constituents (Symbol, Security, GICS_Sector)
        VALUES (?, ?, ?)
    ''', (sym, sec_name, gics))
conn_tg.commit()

conn_yf.close()
conn_tg.close()
print("Success!")
