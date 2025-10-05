import investpy

df = investpy.indices.get_index_historical_data(
    index="S&P 500",
    country="United States",
    from_date="01/01/2023",
    to_date="31/12/2023"
)
df.to_csv('sp500_2023.csv')
print(df.head())