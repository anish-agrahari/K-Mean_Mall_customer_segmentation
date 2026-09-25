# Customer Segmentation using K-Means Clustering

## Overview

This project applies **K-Means clustering**, an unsupervised machine learning algorithm, to the **Mall Customers** dataset.

The project segments customers based on two selected features:

- **Annual Income (k$)**
- **Spending Score (1-100)**

The goal is to identify groups of customers with similar income and spending behavior.

## Objective

The main objective is to use unsupervised machine learning to discover meaningful customer groups without predefined class labels.

The project performs:

1. Dataset loading and inspection
2. Data structure and missing-value analysis
3. Feature selection
4. WCSS calculation for different numbers of clusters
5. Elbow-method analysis
6. K-Means model training
7. Customer-cluster visualization
8. Visualization of cluster centroids

## Dataset

The project uses a CSV file named:

```text
Mall_Customers.csv
```

The dataset contains **200 records and 5 columns**:

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Gender of the customer |
| `Age` | Age of the customer |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Customer spending score |

For clustering, the project uses:

```text
Annual Income (k$)
Spending Score (1-100)
```

## Machine Learning Approach

### K-Means Clustering

K-Means is an unsupervised learning algorithm that divides observations into a specified number of clusters.

The project uses:

```python
KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)
```

### WCSS

**WCSS (Within-Cluster Sum of Squares)** is calculated for cluster counts from 1 to 10.

The WCSS values are used with the **Elbow Method** to determine a suitable number of clusters.

The supplied notebook selects **5 clusters**.

## Methodology

### 1. Import Dependencies

The project uses:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```

### 2. Load Dataset

The dataset is loaded using Pandas:

```python
customer_data = pd.read_csv('Mall_Customers.csv')
```

### 3. Data Analysis

The notebook checks:

- First few records
- Number of rows and columns
- Data types
- Non-null values
- Missing values

The supplied notebook reports:

```text
Shape: 200 rows × 5 columns
Missing values: 0
```

### 4. Feature Selection

The model uses Annual Income and Spending Score:

```python
X = customer_data.iloc[:, [3, 4]].values
```

### 5. Find Optimal Number of Clusters

WCSS is calculated for values of `K` from 1 to 10.

```python
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

### 6. Train K-Means

The final clustering uses 5 clusters.

The resulting clusters and their centroids are visualized using Matplotlib.

## Results

The supplied notebook selects **5 customer clusters** using the elbow-method approach.

The final visualization shows the customer groups according to:

- Annual Income
- Spending Score

The cluster centroids are also displayed on the graph.

> Note: Cluster numbers such as Cluster 1, Cluster 2, etc. are identifiers assigned by the algorithm. They do not inherently represent business categories.

## Project Structure

```text
customer-segmentation-kmeans/
│
├── customer_segmentation.py
├── First_K_mean.ipynb
├── Mall_Customers.csv
├── requirements.txt
├── README.md
└── Project_Report.docx
```

## Requirements

The project requires Python 3.x and the following libraries:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Python Script

Make sure `Mall_Customers.csv` is in the project root directory.

Then run:

```bash
python customer_segmentation.py
```

### Option 2: Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
First_K_mean.ipynb
```

Run the notebook cells sequentially.

## Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**
- **K-Means Clustering**

## Applications

Customer segmentation can be used for:

- Targeted marketing
- Personalized offers
- Customer behavior analysis
- Retail customer profiling
- Identifying customer groups
- Marketing campaign planning

## Limitations

- Only Annual Income and Spending Score are used for clustering.
- The K-Means algorithm requires selecting the number of clusters.
- Cluster labels are algorithm-generated identifiers.
- The analysis is based on the supplied Mall Customers dataset.

## Future Scope

Possible improvements include:

- Applying additional customer features
- Comparing K-Means with DBSCAN or hierarchical clustering
- Evaluating clusters using silhouette score
- Creating an interactive customer-segmentation dashboard
- Performing deeper analysis of each customer segment

## Project Report

A detailed project report is included separately:

```text
Project_Report.docx
```

It contains the project introduction, objectives, dataset description, methodology, results, findings, applications, limitations, future scope, and conclusion.

## Author

**Anish**

Machine Learning / Data Science Project

## License

This project is intended for educational and portfolio purposes.
