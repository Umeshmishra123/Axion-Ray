Python script that you can use for the analysis, including data cleaning, text preprocessing, keyword extraction, and generating visualizations:
import pandas as pd
import numpy as np
import re
import nltk
from fuzzywuzzy import process, fuzz
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load your dataset
data = pd.read_excel('Task 2.xlsx')

# Data Cleaning
# 1. Fill missing values in categorical columns with mode
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    
    data[col] = data[col].fillna(data[col].mode()[0])

# 2. Handle inconsistencies in categorical columns (e.g., typos, inconsistent capitalization)
def preprocess_text(text):
    
    if pd.isnull(text):
        
        return ""
    
    text = str(text)  # Convert to string if not already
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    
    text = text.lower()  # Convert to lowercase
    
    words = [word for word in text.split() if word not in stopwords.words('english')]  # Remove stopwords
    
    return " ".join(words)

# Apply text preprocessing to all categorical columns

for col in categorical_cols:
    
    data[col] = data[col].apply(preprocess_text)

# 3. Handle numerical columns

numerical_cols = data.select_dtypes(include=['number']).columns

for col in numerical_cols:
    
    # Handle missing numerical values (if any)
    
    data[col] = data[col].fillna(data[col].median())

# Visualizing Data
# Trend of repairs over time (if applicable)
if 'REPAIR_MONTH' in data.columns and 'Repair Count' in data.columns:
    
    repair_trend = data.groupby('REPAIR_MONTH').agg({'Repair Count': 'sum'}).reset_index()
    
    plt.figure(figsize=(14, 7))
    
    sns.lineplot(data=repair_trend, x='REPAIR_MONTH', y='Repair Count', marker='o')
    
    plt.title('Trend of Repairs Over Time')
    
    plt.xlabel('Repair Month')
    
    plt.ylabel('Repair Count')
    
    plt.show()

# Word Cloud for Text Fields

def generate_wordcloud(text_column):
    
    text = " ".join(text_column.dropna())
    
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    plt.figure(figsize=(12, 8))
    
    plt.imshow(wordcloud, interpolation="bilinear")
    
    plt.axis("off")
    
    plt.show()

# Generate word cloud for all categorical columns

for col in categorical_cols:
    
    generate_wordcloud(data[col])

# Keyword extraction (Example with the most frequent words)

def extract_keywords(text_column, top_n=10):
    
    words = " ".join(text_column.dropna()).split()
    
    word_counts = Counter(words)
    
    return word_counts.most_common(top_n)

# Extract top 10 keywords for a specific column (e.g., "Issue Description")

if 'Issue Description' in data.columns:
    
    keywords = extract_keywords(data['Issue Description'], top_n=10)
    
    print("Top 10 Keywords in 'Issue Description':")
    
    print(keywords)

# Tag Generation Example (fuzzy matching)

def generate_tags(text_column):
    
    unique_values = text_column.dropna().unique()
    
    standardized_values = {}
    
    for val in unique_values:
        
        standardized_values[val] = process.extractOne(val, unique_values, scorer=fuzz.token_sort_ratio)[0]
    
    return standardized_values

# Apply fuzzy matching to a categorical column

if 'Issue Description' in data.columns:
    
    tags = generate_tags(data['Issue Description'])
    
    data['Issue Tags'] = data['Issue Description'].map(tags)

# Save the cleaned data with tags to CSV

data.to_csv('cleaned_data_with_tags.csv', index=False)

# Print a sample of the cleaned data with tags

print(data.head())

# Save the Python script itself

script_code = '''

import pandas as pd

import numpy as np

import re

import nltk

from fuzzywuzzy import process, fuzz

from nltk.corpus import stopwords

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

# Load your dataset

data = pd.read_excel('Task 2.xlsx')

# Data Cleaning
# 1. Fill missing values in categorical columns with mode

categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:

    data[col] = data[col].fillna(data[col].mode()[0])

# 2. Handle inconsistencies in categorical columns (e.g., typos, inconsistent capitalization)

def preprocess_text(text):
    
    if pd.isnull(text):
        
        return ""
    
    text = str(text)  # Convert to string if not already
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    
    text = text.lower()  # Convert to lowercase
    
    words = [word for word in text.split() if word not in stopwords.words('english')]  # Remove stopwords
    
    return " ".join(words)

# Apply text preprocessing to all categorical columns

for col in categorical_cols:
    
    data[col] = data[col].apply(preprocess_text)

# 3. Handle numerical columns

numerical_cols = data.select_dtypes(include=['number']).columns

for col in numerical_cols:
    
    # Handle missing numerical values (if any)
    
    data[col] = data[col].fillna(data[col].median())

# Visualizing Data
# Trend of repairs over time (if applicable)

if 'REPAIR_MONTH' in data.columns and 'Repair Count' in data.columns:
    
    repair_trend = data.groupby('REPAIR_MONTH').agg({'Repair Count': 'sum'}).reset_index()
    
    plt.figure(figsize=(14, 7))
    
    sns.lineplot(data=repair_trend, x='REPAIR_MONTH', y='Repair Count', marker='o')
    
    plt.title('Trend of Repairs Over Time')
    
    plt.xlabel('Repair Month')
    
    plt.ylabel('Repair Count')
    
    plt.show()


# Word Cloud for Text Fields

def generate_wordcloud(text_column):
    
    text = " ".join(text_column.dropna())
    
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    plt.figure(figsize=(12, 8))
    
    plt.imshow(wordcloud, interpolation="bilinear")
    
    plt.axis("off")
    
    plt.show()

# Generate word cloud for all categorical columns

for col in categorical_cols:
    
    generate_wordcloud(data[col])

# Keyword extraction (Example with the most frequent words)

def extract_keywords(text_column, top_n=10):
    
    words = " ".join(text_column.dropna()).split()
    
    word_counts = Counter(words)
    
    return word_counts.most_common(top_n)

# Extract top 10 keywords for a specific column (e.g., "Issue Description")

if 'Issue Description' in data.columns:
    
    keywords = extract_keywords(data['Issue Description'], top_n=10)
    
    print("Top 10 Keywords in 'Issue Description':")
    
    print(keywords)

# Tag Generation Example (fuzzy matching)

def generate_tags(text_column):
    
    unique_values = text_column.dropna().unique()
    
    standardized_values = {}
    
    for val in unique_values:
    
        standardized_values[val] = process.extractOne(val, unique_values, scorer=fuzz.token_sort_ratio)[0]
    
    return standardized_values

# Apply fuzzy matching to a categorical column

if 'Issue Description' in data.columns:
    
    tags = generate_tags(data['Issue Description'])
    
    data['Issue Tags'] = data['Issue Description'].map(tags)

# Save the cleaned data with tags to CSV

data.to_csv('cleaned_data_with_tags.csv', index=False)

# Print a sample of the cleaned data with tags

print(data.head())
'''

# Save the script to a file

with open('analysis_script.py', 'w') as file:

    file.write(script_code)


print("Script and CSV file have been generated.")


What this script does:
Data Cleaning:

Fills missing values in categorical columns with the mode (most frequent value).
Handles text preprocessing (removes non-alphabetic characters, converts to lowercase, and removes stopwords).
Fills missing values in numerical columns with the median.
Data Visualizations:

Creates a trend plot of repairs over time (if applicable).
Generates word clouds for text columns like "Issue Description".
Tag Generation:

Applies fuzzy matching to clean up and standardize text values (e.g., "Issue Description").
Creates a new "Issue Tags" column to store standardized tags.
