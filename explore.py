import pandas as pd

# Load the dataset
# The file uses tab (\t) as separator, and has no header row
# so we name the columns ourselves
df = pd.read_csv('data/spam.csv', sep='\t', header=None, names=['label', 'message'])

print(df.head(10))          # see first 10 rows
print(df.shape)             # how many rows and columns?
print(df['label'].value_counts())  # how many spam vs ham?
