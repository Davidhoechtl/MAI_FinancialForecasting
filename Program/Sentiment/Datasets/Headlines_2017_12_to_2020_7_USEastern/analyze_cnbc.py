import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv("cnbc_headlines.csv")
df['Time'] = pd.to_datetime(df['Time'].str.replace('ET','', regex=False).str.strip())
print(df.head())

# Clean and parse the 'Time' column into datetime
# Example format: "7:51  PM ET Fri, 17 July 2020"
df['Datetime'] = pd.to_datetime(df['Time'], errors='coerce')

# Drop invalid rows (failed parsing)
df = df.dropna(subset=['Datetime'])

# Extract just the date part
df['Date_Only'] = df['Datetime'].dt.date

# Count number of headlines per day
counts = df.groupby('Date_Only').size().reset_index(name='Headline_Count')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(counts['Date_Only'], counts['Headline_Count'], marker='o')
plt.title("Number of Headlines per Day")
plt.xlabel("Date")
plt.ylabel("Headline Count")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Choose your intraday granularity ---
# Group by HOUR of day
df['Hour'] = df['Datetime'].dt.hour

# Count how many headlines per hour
intra_counts = df.groupby('Hour').size().reset_index(name='Headline_Count')

# Plot
plt.figure(figsize=(9, 4))
plt.bar(intra_counts['Hour'], intra_counts['Headline_Count'], color='steelblue')
plt.xticks(range(0, 24))
plt.xlabel("Hour of Day (Local Time)")
plt.ylabel("Number of Headlines")
plt.title("Intra-Day Headline Frequency")
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()