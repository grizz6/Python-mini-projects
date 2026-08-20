# Python Mini Projects

Two exploratory data analysis / machine learning projects built in Jupyter notebooks, using the standard pandas + scikit-learn stack.

---

## 1. 911 Call Data (`911 Call data/911 Call data.ipynb`)

**Data:** `911.csv` — raw 911 dispatch records (title, timestamp, township, zip code).

### How it's done
1. Load the CSV with `pandas.read_csv`.
2. Engineer a `Reason` column by splitting the raw `title` field on `:` (titles look like `EMS: CARDIAC EMERGENCY`) using `Series.apply(lambda x: x.split(':')[0])`.
3. Parse `timeStamp` with `pd.to_datetime`, then derive `Hour`, `Month`, and `day_name` columns from it.
4. Aggregate with `groupby` on `Month`, `Date`, and `(Hour, day_name, Reason)` to get call volume over time.
5. Visualize with `seaborn`/`matplotlib`: count plots of calls by reason/day/month, a day-of-week × hour heatmap (`sns.heatmap` on a pivoted table), and a correlation matrix of `Hour`, `Reason`, and `day_name` (label-encoded via `.astype('category').cat.codes`).
6. Discusses (in markdown, not executed) how `sklearn.model_selection.train_test_split` and `sklearn.linear_model.LinearRegression` could predict call volume from time-based features, and what R² would indicate about fit quality.

### Code used
`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` (`train_test_split`, `LinearRegression`, `mean_squared_error` imported for the regression discussion).

### The algorithm
The core technique actually executed is **groupby aggregation + pivoting** to turn a flat event log into a time-series count table (date/hour/day-of-week × call reason), which is what powers every chart. The **linear regression** step is scoped and explained (features → train/test split → fit → R² interpretation) but left as a discussion rather than a run cell.

---

## 2. Brain Tumor Project (`brain tumor project/Grishma_Gajurel_project.ipynb`)

**Data:** `brain_tumor_dataset.csv` — patient age, tumor size, tumor type, growth rate, treatment history, survival rate.

### How it's done
1. Load and inspect the dataset; handle outliers in numeric columns via **IQR capping** (clip values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`).
2. Label-encode categorical features (tumor type, treatment) and apply `StandardScaler` to normalize numeric features.
3. Explore relationships with a correlation heatmap, tumor-size distribution plots, and survival-rate-by-tumor-type boxplots.
4. Build the modeling table: `X = df[['Age', 'Tumor_Size', 'Tumor_Growth_Rate']]`, `y = df['Survival_Rate']`.
5. Split 60/40 train/test with `train_test_split(..., test_size=0.4, random_state=42)`.
6. Fit `sklearn.linear_model.LinearRegression()` on the training set, predict on the test set.
7. Evaluate with **Mean Squared Error** and **R²** (`mean_squared_error`, `r2_score`), then plot actual vs. predicted survival rate for a random sample of 100 test points as a grouped bar chart.

### Code used
`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` (`LinearRegression`, `train_test_split`, `mean_squared_error`, `r2_score`).

### The algorithm
**Ordinary least squares linear regression** — fits a line (in 3 dimensions: age, tumor size, growth rate) that minimizes squared error against survival rate, trained on 60% of the data and validated on the held-out 40%. MSE quantifies average prediction error magnitude; R² quantifies how much of the variance in survival rate the three features explain.
