# DL RNN Project

# behavior & characteristics of time series data

## What is the timeseries data

Time series data is a sequence of data points collected or recorded at specific time intervals over a period of time. This type of data is fundamental to many fields, including economics, finance, weather forecasting, and IoT applications. A time series is a chronological sequence of observations on a particular variable. The observations are typically taken at equally spaced time intervals such as hourly, daily, weekly, monthly, quarterly, or annually. Each data point is associated with a specific timestamp, making the time dimension an essential component of the data structure.

## Key characteristics of time series data

### Temporal Ordering

- Data points have a natural chronological order
- The sequence matters - shuffling the data would lose critical information
- Each observation is dependent on its position in time

### Time Based Structure

- Regularly spaced intervals (e.g., hourly, daily, monthly)
- Sometimes irregularly spaced (event-based recordings)
- Contains timestamps or datetime indices

### Autocorrelation

- Observations are often dependent on previous values
- Current values may be correlated with past values (lag relationships)
- Creates patterns that can be modeled and predicted

### Non-stationary

- Statistical properties like mean and variance often change over time
- May exhibit trends, seasonality, or cyclical patterns
- Can be affected by structural breaks or regime changes

### Common Components of Time Series Data

### Trend Component

- Long-term movement in the data (upward or downward)
- Reflects the general direction in which the graph is moving
- Examples: population growth, economic expansion, technological improvement

### Seasonal Component

- Regular, predictable patterns that repeat at fixed intervals
- Often related to calendar or business cycles
- Examples: higher retail sales during holidays, increased energy consumption in winter/summer

### Cyclical component

- Fluctuations around the trend that don't follow a fixed period
- Usually related to business or economic cycles
- Typically longer than seasonal variations (2+ years)

### Irregular Component

- Random, unpredictable fluctuations
- Remains after other components have been removed
- Represents unexplained variation or measurement errors

# **Daily Climate time series** Dataset

The [Dataset](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data?resource=download) is fully dedicated for the developers who want to train the model on Weather Forecasting for Indian climate. This dataset provides data from **1st January 2013** to **24th April 2017** in the city of Delhi, India. The 4 parameters here are. This dataset has been collected from Weather Underground API. Dataset ownership and credit goes to them.

- `date` : Date of format YYYY-MM-DD
- `Meantemp` : Mean temperature averaged out from multiple 3 hour intervals in a day.
- `humidity` : Humidity value for the day
- `wind_speed` : Wind speed measured in kmph.
- `meanpressure` : Pressure reading of weather (measure in atm)

# Used Libraries

| **Import** | **Purpose /Description** |
| --- | --- |
| `pandas as pd` | Data manipulation and analysis using DataFrames |
| `numpy as np` | Numerical operations and array handling |
| `matplotlib.pyplot as plt` | Creating static plots and visualizations |
| `seaborn as sns` | Statistical data visualization (built on matplotlib) |
| `from datetime import timedelta` | Time delta manipulation (e.g., shifting dates) |
| `plot_acf`, `plot_pacf` | Plot autocorrelation and partial autocorrelation functions (used in time series analysis) |
| `adfuller` | Perform Augmented Dickey-Fuller test to check for stationarity |
| `seasonal_decompose` | Decompose time series into trend, seasonality, and residuals |
| `MinMaxScaler` | Normalize data to a specific range, usually [0, 1] |
| `train_test_split` | Split data into training and test sets |
| `lag_plot` | Create lag plots to identify autocorrelation in time series data |

# Importing and Exploring the dataset:

### 1. **Loading the dataset:**

The dataset is read from a CSV file named **"DailyDelhiClimate.csv"** into a Pandas DataFrame called `df`. Displays the first 5 rows of the dataset, giving a quick look at the structure and values of the columns.

```python
df = pd.read_csv("DailyDelhiClimate.csv")
df.head()
```

### 2. Checking for missing values

All values are present (no missing data), which means we can proceed without needing to handle NaNs.

```python
df.isnull().sum()

```

### 2. Describing numerical columns

```python
df.describe()
```

| Metric | Interpretation |
| --- | --- |
| **count** | Total number of records (1575 for each column) |
| **mean** | Average value (e.g., average temperature is 25.2°C) |
| **std** | Standard deviation, shows variability (e.g., `wind_speed` has relatively high variability) |
| **min / max** | Minimum and maximum values (e.g., `meanpressure` ranges widely, possibly with outliers) |
| **25% / 50% / 75%** | Percentiles — indicate data spread and skewness |

### 3. Data Handling and Indexing

→ Convert data column to datetime: Converts the `date` column from string format to actual `datetime` objects using a month/day/year format. This is essential for time series analysis.

```python
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
```

 → Sorting the dataset by date:  Ensures the dataset is in chronological order. This is crucial for time-based operations such as rolling averages or forecasting.

```python
df = df.sort_values('date')
```

→ setting date as index:  Makes the `date` column the index of the DataFrame. This allows for easier time-based slicing and plotting.

```python
df.set_index('date', inplace=True)
```

→ getting data range and record counts: this shows us the dataset from which date to which date it have data 

- **Start date:** January 1, 2013
- **End date:** April 24, 2017
- **Total records:** 1575 days (i.e., daily data over ~4.3 years)

```python
df.index.min(), df.index.max(), len(df)
```

### Conclusion:

It's clean, has no missing values, and is now time-series ready with the `date` as the index.

## Data Visualization

### 1- Subplots of Climate Variables

This creates individual time series plots for each climate variable (`meantemp`, `humidity`, `wind_speed`, and `meanpressure`) over the entire dataset period. It gives a visual sense of the data distribution and fluctuations for each variable.

```python
df.plot(subplots=True, figsize=(15, 10), title='Climate Variables Over Time')
```

![image.png](image.png)

### 2- Outlier detection

**Z-score** measures how many standard deviations a value is from the mean. 
Any data point with a Z-score > 3 (or < -3) is considered an **outlier**. **Result:** A total of **18 outliers** were detected across the four columns. These may be measurement errors, extreme weather events, or unusual fluctuations.

```python
from scipy.stats import zscore
z_scores = zscore(df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']])
outliers = (abs(z_scores) > 3).sum()
```

### 3- Monthly average temperature

 **Resampling** by month calculates the average temperature for each month. Helps identify long-term trends and reduce daily noise. The resulting plot likely shows **seasonal waves**, typical of climate data.

```python
monthly_df = df['meantemp'].resample('M').mean()
monthly_df.plot(title='Monthly Mean Temperature')
```

![image.png](image%201.png)

### 4- Seasonal Decomposition of Temperature

This decomposition separates the `meantemp` time series into 3 main components:

1.  **Trend Component shows**  The **underlying direction** or movement in the data over time, ignoring seasonal and short-term fluctuations.
2. **Seasonal Component shows** The **repeating pattern** in the data that recurs at regular intervals—in this case, annually.
3.  **Residual (Noise) Component shows** The **remaining variation** in the data after removing the trend and seasonal components.

```python
result = seasonal_decompose(df['meantemp'], model='additive', period=365)
result.plot()
```

→ The trend is **gradually increasing**, indicating that average daily temperatures in Delhi have **risen** between 2013 and 2017. This could suggest a warming trend in the region over these years, possibly linked to climate change or urban heat effects.

→ There is a **clear yearly cycle** in the temperature, with peaks and dips representing **summer and winter seasons** respectively. This confirms the **seasonal nature** of Delhi's climate, where temperature oscillates in a regular yearly rhythm.

→ Residual values fluctuate roughly between **5 and +5**, suggesting a relatively small amount of noise or random variation. No significant anomalies in residuals, indicating that the decomposition model fits the data well.

![image.png](image%202.png)

### 5- Feature Correlation Matrix

heatmap shows **pairwise Pearson correlation coefficients** between the climate variables.

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

![image.png](image%203.png)

| Variable Pair | Correlation | Interpretation |
| --- | --- | --- |
| meantemp & humidity | **-0.57** | Strong **negative** correlation: higher temperatures tend to occur with lower humidity. |
| meantemp & wind_speed | **0.29** | Weak **positive** correlation. |
| humidity & wind_speed | **-0.37** | Moderate **negative** correlation. |
| meanpressure & others | ~0 | Almost **no correlation** with other variables. |

### 6- Rolling Mean and Standard Deviation

```python
df['meantemp'].rolling(window=30).mean().plot(label='30-day Mean')
df['meantemp'].rolling(window=30).std().plot(label='30-day Std')
```

![image.png](image%204.png)

→ The **rolling mean** clearly shows **annual seasonality**.  The **rolling std** is relatively low and stable, indicating consistent temperature variability year to year.

### 7- Augmented Dickey-Fuller (ADF) Test

The ADF test is used to check for **stationarity** in a time series.

- **Null Hypothesis (H₀)**: The series is **non-stationary**.
- **Alternative Hypothesis (H₁)**: The series is **stationary**.

```python
adf_result = adfuller(df['meantemp'])
```

**ADF Statistic**: -2.377 &  **p-value**: 0.148
 ⇒ Since the **p-value > 0.05**, we **fail to reject the null hypothesis**.  This suggests that the temperature series is **non-stationary**—it has trends or seasonality that need to be removed before modeling (e.g., using differencing or decomposition).

### 8- ACF & PACF Plots

The autocorrelation plot shows how the temperature values are correlated with their own past values at various lags, This gradual, linear decay pattern is typical of non-stationary time series, specifically those with a trend component. The blue shaded area represents the confidence interval (typically 95%); values outside this range are statistically significant

The partial autocorrelation plot shows the direct correlation between an observation and its lag after removing the effects of intermediate observations, This pattern suggests that once you account for the immediate previous 1-2 days' temperatures, further lags don't add significant explanatory power

![image.png](image%205.png)

### 9- **ADF Test**

The **Augmented Dickey-Fuller (ADF)** test checks whether a **time series is stationary**, meaning its statistical properties (mean, variance, autocorrelation) are **constant over time**. the data is not stationary 

```python
from statsmodels.tsa.stattools import adfuller

# ADF test on the 'meantemp' series
result = adfuller(df['meantemp'].dropna(), regression='c')  # 'c' for constant only

stat, p_value, lags, nobs, crit_vals, icbest = result

print("ADF Statistic:", stat)
print("p-value:", p_value)
print("Number of Lags Used:", lags)
print("Number of Observations Used:", nobs)
print("Critical Values:")
for key, value in crit_vals.items():
    print(f"   {key}: {value}")

# Interpretation
if p_value < 0.05:
    print("\n✅ The series is likely stationary (reject the null hypothesis).")
else:
    print("\n⚠️ The series is likely non-stationary (fail to reject the null hypothesis).")

```

```
ADF Statistic: -2.377592279031286
p-value: 0.14816094866398516
Number of Lags Used: 10
Number of Observations Used: 1564
Critical Values:
   1%: -3.4345380212339838
   5%: -2.8633897592903237
   10%: -2.5677547800740443

⚠️ The series is likely non-stationary (fail to reject the null hypothesis).
```

⇒ The **temperature** shows **seasonality** (clear yearly cycles). It also has an **increasing trend** over years (as seen in earlier decomposition).

A **non-stationary** time series has at least one of these characteristics:

- **Trend**: A long-term increase or decrease in the mean.
- **Seasonality**: Repeating patterns at regular intervals.
- **Changing variance** over time.

⇒  For models like **ARIMA** or **traditional statistical models**, stationarity is a strict requirement. For **Recurrent Neural Networks (RNNs)**  They **can handle non-stationary data** better than classical models **because they learn temporal dependencies**.

# Model Option

**LSTM (Long Short-Term Memory)** networks is a **good option** for predicting **temperature**, especially when dealing with **time series data** that is **non-stationary**

- LSTMs are a type of recurrent neural network designed to **learn patterns over time**, including **trends and seasonality**, even in **non-stationary data**.
- LSTMs can remember information over long sequences, which is useful when temperature is affected by longer-term trends

## Data Splitting

- keeping only the `meantemp` column from the dataset, which is the **target variable** for prediction.
- Scaling the data to the range [0, 1] using **Min-Max Scaling**. LSTMs train faster and perform better when input features are on the same scale, especially for time series.
- Convert a flat time series into **sliding windows for example 30 day window**
    - `X`: input sequences of `window_size` timesteps.
    - `y`: the next value (what you want the model to predict).
- Each input sample will consist of the **previous 30 days' temperatures** to predict the **next day's temperature**.
- You define two different test set sizes:
    - **90 days (~3 months)**
    - **365 days (~1 year)**

```python
# Use only 'meantemp' for univariate time series prediction
temperature_data = df[['meantemp']].copy()

# Normalize the temperature data
scaler = MinMaxScaler()
temperature_scaled = scaler.fit_transform(temperature_data)

# Convert to sequences suitable for RNN
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Define window size (e.g., past 30 days to predict next day)
window_size = 30
X, y = create_sequences(temperature_scaled, window_size)

# Indices for 3 months (90 days) and 1 year (365 days)
test_days_3mo = 90
test_days_1yr = 365

# Create train-test splits
X_train_3mo, y_train_3mo = X[:-test_days_3mo], y[:-test_days_3mo]
X_test_3mo, y_test_3mo = X[-test_days_3mo:], y[-test_days_3mo:]

X_train_1yr, y_train_1yr = X[:-test_days_1yr], y[:-test_days_1yr]
X_test_1yr, y_test_1yr = X[-test_days_1yr:], y[-test_days_1yr:]

X.shape, X_train_3mo.shape, X_test_3mo.shape, X_train_1yr.shape, X_test_1yr.shape
```

# Simple Model

### Model Building Function

**Creates a Sequential model** — layers are stacked one after another. 

An **LSTM layer** with 50 units and ReLU activation. This is the core of your RNN model.  number of LSTM cells or neurons — each one helps the model learn temporal patterns. `relu` is sometimes faster and avoids vanishing gradient issues, but `'tanh'` is more commonly used in LSTMs.

why choose 50 units ? 

- Too few units → **Underfitting** (model can’t capture the complexity of the data).
- Too many units → **Overfitting** (model memorizes training data, poor on test data).

⇒ With only ~1,500 total observations, 50 units is **reasonable** — enough to learn trends without overwhelming the data.

A **Dense layer** to produce the final output (1 temperature value).

- **Adam optimizer** — efficient and widely used for deep learning.
- **Mean Squared Error (MSE)** — suitable for regression tasks like predicting temperature.

```python
def build_rnn_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

```

### Model Training

→ `epochs=20`: 20 full passes over the training data.

→ `batch_size=32`: Processes data in batches of 32 samples before updating weights.

→ `validation_data`: Evaluates model performance on the 3-month test set after each epoch.

→ `verbose=1`: Shows training progress in the console.

On 3 months 

```python
model = build_rnn_model((X_train_3mo.shape[1], X_train_3mo.shape[2]))
history = model.fit(
    X_train_3mo,
    y_train_3mo,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_3mo, y_test_3mo),
    verbose=1
)

```

On 1 Year 

```python
model = build_rnn_model((X_train_1yr.shape[1], X_train_1yr.shape[2]))
history = model.fit(
    X_train_1yr,
    y_train_1yr,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_1yr, y_test_1yr),
    verbose=1 
)
```

## 3 months prediction

![image.png](image%206.png)

![image.png](image%207.png)

### Model Validation

```
3 Months Evaluation Metrics:
MAE: 1.844
RMSE: 2.191
MAPE: 8.11%
```

## 1 year prediction

![image.png](image%208.png)

![image.png](image%209.png)

### Model Validation

```
1 Year Evaluation Metrics:
MAE: 1.673
RMSE: 2.167
MAPE: 6.97%
```

# Complex Model

### Model Building Function

1. LSTM Layer 
    1. **100 units** in the first LSTM layer → learns more complex temporal features.
    2. `return_sequences=True` → returns the **full sequence of hidden states**, not just the final one. This is needed when you're stacking LSTM layers — the next LSTM expects a sequence input, not a single vector.
    3. Second LSTM layer with **50 units**, now consuming the sequence output from the first layer. Learns **higher-level temporal patterns** based on the encoded sequence.
2. Dropout Layer  
    1. **Dropout** randomly sets 20% of outputs to zero during training.
    2. Prevents **overfitting** by forcing the network to not rely too heavily on any one neuron.
3. Dense Layer 
    1. Final output: a single predicted temperature value.

```python
def build_advanced_rnn_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

```

### Training Setup

**EarlyStopping** stops training if validation loss doesn’t improve for 10 consecutive epochs.  Helps **avoid overfitting** and unnecessary computation. and `restore_best_weights=True`: Ensures you get the **best model**, not just the one from the final epoch.

```python
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

### Model Training

Trains for up to 50 epochs and Uses **early stopping** based on validation loss.

```python
history = model.fit(
    X_train_3mo, y_train_3mo,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_3mo, y_test_3mo),
    callbacks=[early_stop],
    verbose=1
)
```

## Model Comparison

| Feature | Simple Model | Advanced Model |
| --- | --- | --- |
| **Architecture** | 1 LSTM (50 units) → Dense(1) | LSTM(100) → Dropout → LSTM(50) → Dropout → Dense(1) |
| **Depth** | Shallow | Deep (Stacked LSTMs) |
| **Regularization** | ❌ None | ✅ Dropout (2 layers) |
| **Sequence Handling** | Only last LSTM output used | Stacked LSTMs with `return_sequences=True` |
| **Overfitting Control** | None | ✅ EarlyStopping and Dropout |
| **Training Time** | Fast | Slower (more layers and units) |
| **Model Capacity** | Lower | Higher (can learn more complex patterns) |

## 3 months prediction

![image.png](image%2010.png)

![image.png](image%2011.png)

### Validation

```
3 Months Evaluation Metrics:
MAE: 1.537
RMSE: 1.900
MAPE: 6.72%
```

## 1 year prediction

![image.png](image%2012.png)

![image.png](image%2013.png)

### Validation

```
1 Year Evaluation Metrics:
MAE: 1.449
RMSE: 1.836
MAPE: 6.02%
```

# Models Comparison

## 3 Months Comparison

| Metric | Simple Model | Complex Model | Which Performs Better? |
| --- | --- | --- | --- |
| **MAE** (Mean Absolute Error) | 1.844 | **1.537** | ✅ Complex |
| **RMSE** (Root Mean Squared Error) | 2.191 | **1.900** | ✅ Complex |
| **MAPE** (Mean Absolute Percentage Error) | 8.11% | **6.72%** | ✅ Complex |

## 1 Year Comparison

| Metric | Simple LSTM | Complex LSTM | Which Is Better? |
| --- | --- | --- | --- |
| **MAE** (Mean Absolute Error) | 1.673 | **1.449** | ✅ Complex |
| **RMSE** (Root Mean Squared Error) | 2.167 | **1.836** | ✅ Complex |
| **MAPE** (Mean Absolute Percentage Error) | 6.97% | **6.02%** | ✅ Complex |

# Conclusion

In this study, we compared two LSTM-based models — a **simple LSTM** and a **complex stacked LSTM** — for univariate temperature forecasting using past 30 days of data.

### ✅ Key Findings:

- On both **3-month** and **1-year** test periods, the **complex model consistently outperformed** the simple model across all evaluation metrics:
    - **MAE**, **RMSE**, and **MAPE** were significantly lower, indicating **better accuracy** and **fewer large prediction errors**.
- The **complex model**, which includes two stacked LSTM layers with dropout and early stopping, was able to better capture **temporal dependencies and seasonal patterns** in the data.
- Although it requires **longer training time**, the complex model demonstrated **better generalization** and **lower risk of overfitting**.