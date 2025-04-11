
#  Project Overview

This project builds a machine learning model to predict the presence of heart disease using the k-Nearest Neighbors (k-NN) algorithm, along with exploratory data analysis and preprocessing steps. The dataset is based on structured medical information from patients.

---

##  Dataset

The dataset is loaded from a CSV file:  
<a href = https://github.com/EmmanuelKiriinya/Heart-Disease-ML/blob/main/Heart%20Disease%20file/heart.csv> FILE </a>

It includes features such as:
- `Age`, `Sex`
- `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`
- `RestingECG`, `MaxHR`, `ExerciseAngina`
- `Oldpeak`, `ST_Slope`
- Target variable: `HeartDisease` (0 = No, 1 = Yes)

---

##  Step-by-step Breakdown

### 1. **Library Imports**
Essential Python libraries are imported:
- `pandas`, `numpy` for data handling
- `matplotlib`, `seaborn` for visualization
- `sklearn` modules for preprocessing, modeling, evaluation

### 2. **Loading and Exploring the Data**
```python
heart_df = pd.read_csv('path/to/heart.csv')
heart_df.info()
heart_df.describe()
```
- `.info()` gives an overview of columns and data types.
- `.describe()` summarizes statistics for numeric features.

### 3. **Exploratory Data Analysis (EDA)**
- Countplots of categorical features are created to visualize distributions.
- A correlation heatmap is plotted to understand relationships between numeric variables.

### 4. **Data Preprocessing**
- One-hot encoding is used to convert categorical variables to numeric form.
- Features and labels are split:
  ```python
  X = df_clean.drop("HeartDisease", axis=1)
  y = df_clean["HeartDisease"]
  ```

- The dataset is then split into training and validation sets:
  ```python
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=417)
  ```

- Features are scaled using `MinMaxScaler`.

### 5. **Model Training & Feature Selection**
The model evaluates the importance of individual features using k-NN:
```python
for feature in features:
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train[[feature]], y_train)
    ...
```

### 6. **Hyperparameter Tuning with GridSearchCV**
A hyperparameter grid is defined and passed into `GridSearchCV` to optimize the model:
```python
param_grid = {
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid = GridSearchCV(knn, param_grid, cv=5)
```



### 7. **Model Evaluation**
The best model is selected using grid search. It is then evaluated using:
- Accuracy score
- Confusion matrix (displayed using `ConfusionMatrixDisplay`)

---

##  Results

The model achieves varying accuracy across features, and final tuning pushes accuracy further with proper metric selection (`manhattan`, `euclidean`, etc.).

---

