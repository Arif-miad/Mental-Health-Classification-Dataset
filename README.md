
---

# Mental Health Classification Dataset

## Overview

This dataset is a collection of textual statements tagged with different mental health statuses. The dataset is designed for building models that can classify mental health conditions based on the content of statements, making it ideal for training machine learning models for sentiment analysis, chatbot development, or mental health trend analysis.

The dataset consists of statements collected from various sources, with each statement labeled according to its corresponding mental health status.

## Dataset Details

### **Columns:**
- **`statement`**: The textual statement or post shared by individuals.
- **`status`**: The mental health status tag assigned to the statement. The statuses include:
  - `Normal`
  - `Anxiety`
  - `Depression`
  - `Stress`
  - `Suicidal`
  - `Bi-Polar`
  - `Personality Disorder`
- **`statement_length`**: The length of the statement in terms of the number of characters.
- **`cumulative_length`**: The cumulative length of the statements as the dataset progresses.

### **Data Sources:**
The dataset has been aggregated from various publicly available mental health-related datasets, including:
- Social media posts
- Reddit posts
- Twitter posts
- Various mental health discussion forums

### **Mental Health Status Labels:**
The statements are tagged with one of the following mental health statuses:
- **Normal**: No apparent signs of mental health issues.
- **Anxiety**: Statements related to feelings of anxiety or unease.
- **Depression**: Statements indicating depression or depressive symptoms.
- **Stress**: Statements related to stress or being overwhelmed.
- **Suicidal**: Statements suggesting suicidal thoughts or tendencies.
- **Bi-Polar**: Statements related to bi-polar disorder or mood swings.
- **Personality Disorder**: Statements related to various personality disorders.

## Use Cases

This dataset can be used for:
- **Mental Health Chatbot Development**: Train chatbots to recognize and respond to different mental health conditions.
- **Sentiment Analysis**: Classify and analyze the sentiment behind each statement to gauge an individual’s mental health.
- **Research**: Academic studies focused on mental health patterns, trends, and triggers.
- **Predictive Models**: Build models to predict mental health statuses from textual data.

## How to Use

1. **Clone the Repository**:  
   To use the dataset and code, start by cloning this repository:
   ```bash
   git clone https://github.com/your-username/mental-health-classification.git
   ```

2. **Install Dependencies**:  
   You will need to install the required libraries listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preprocessing**:  
   Use the `data_preprocessing.py` script to clean and preprocess the dataset. This step includes handling missing values, tokenizing the text, and feature engineering.

4. **Exploratory Data Analysis (EDA)**:  
   Explore the dataset and visualize data distributions using the `eda.py` script. Visualizations may include:
   - Histograms of statement lengths
   - Bar plots of mental health status distribution
   - Wordclouds or other textual visualizations

5. **Classification Models**:  
   Build machine learning models to classify the mental health status based on the statements. Example models include:
   - Logistic Regression
   - Decision Trees
   - Naive Bayes
   - Random Forest
   - Support Vector Machines (SVM)

   The results can be evaluated using metrics like accuracy, F1-score, and confusion matrices.

## Example Usage
### **Import Libraries**:
```python
import numpy as np 
import seaborn as sns
import matplotlib.pyplot  as plt
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
```
### **Load Dataset**:
```python
df = pd.read_csv('/kaggle/input/sentiment-analysis-for-mental-health/Combined Data.csv')
df.head()
df.isnull().sum()
df.shape
df.info()
df.describe()
df = df.drop(columns=['Unnamed: 0'], axis=1)

```

### **Exploratory data analysis**:
```pthon
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='status', palette='viridis')
plt.title('Count of Mental Health Statuses', fontsize=14)
plt.xlabel('Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.show()

```

```python
from wordcloud import WordCloud
text = " ".join(df['statement'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Statements', fontsize=16)
plt.show()
```

```python
import plotly.express as px
fig = px.sunburst(df, path=['status'], values='statement_length',
                  title='Sunburst Chart of Mental Health Status Distribution')
fig.show()
```
### **Data Preprocessing**:
```python
most_frequent_statement = df['statement'].mode()[0]
df['statement'] = df['statement'].fillna(most_frequent_statement)

le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])  # Encode target column

```

### **Train a Classification Model**:
```python
X = df['statement']  # Feature: Text data
y = df['status']  # Target: Encoded labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}



results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)


```

## Acknowledgments

- This dataset was curated using various publicly available datasets and sources related to mental health.
- A special thanks to the original authors of the datasets, whose work contributed to this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

- **Name**: Your Name

- **LinkedIn**: [my-linkedin-profile](www.linkedin.com/in/arif-miah-8751bb217)
- **Kaggle**: [my-kaggle-profile](https://www.kaggle.com/code/arifmia/notebook624eef4578)

---


