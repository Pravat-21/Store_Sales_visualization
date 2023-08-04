import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data as a dictionary (Replace this with your own data)
data = {
    'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Create a countplot using Seaborn
plt.figure(figsize=(8, 6))
sns.countplot(x='Category', data=df)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Countplot for Categorical Column')
plt.xticks(rotation=45)

# Display the plot in Streamlit
st.pyplot()
