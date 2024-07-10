import pandas as pd

# Replace 'your_file.csv' with the path to your actual CSV file
file_path = 'train.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df)

# If you want to display it in a more readable format in a Jupyter Notebook
# you can use the display function from IPython
from IPython.display import display

display(df)
