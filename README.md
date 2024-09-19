# Market-Basket-Analysis-using-Association-Mining-Rule
### Documentation for Apriori Algorithm on Grocery Sales Dataset

---

#### 1. **Importing Essential Libraries and Suppressing Warnings**
To start, we import the necessary libraries for data analysis, visualization, and applying the Apriori algorithm for association rule mining:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

- **Libraries used:**
  - **NumPy**: Used for numerical operations and array manipulations.
  - **Pandas**: Essential for data manipulation and analysis.
  - **Seaborn & Matplotlib**: Visualization libraries for plotting graphs.
  - **Warnings**: Suppressing unnecessary warnings that can clutter the output.

---

#### 2. **Loading the Data and Creating a Backup**
Next, we load the dataset and create a backup to ensure we can revert to the original data if needed.

```python
df = pd.read_csv('Groceries_dataset.csv')
df_backup = df
df.sample(6)
```

- **df**: The main dataset loaded from the `Groceries_dataset.csv` file.
- **df_backup**: A backup copy of the dataset.
- **df.sample(6)**: Displays 6 random rows from the dataset.

---

#### 3. **Exploring the Dataset**
We inspect the structure and metadata of the dataset.

```python
df.info()
```
- **df.info()**: Provides a summary of the dataset including the data types and the presence of null values.

---

#### 4. **Data Preprocessing**
To ensure accurate analysis, the 'Date' column is converted to a proper datetime format.

```python
df['Date'] = df['Date'].astype('datetime64[ns]')
```
- **df['Date']**: Converts the 'Date' column to the appropriate datetime data type.

---

#### 5. **Checking for Null Values**
We check if there are any missing values in the dataset.

```python
df.isnull().sum()
```
- **df.isnull().sum()**: Provides a count of null values in each column of the dataset.

---

#### 6. **Analyzing Item Distribution**
We analyze the distribution of different items sold based on frequency.

```python
Item_Distribution = df.groupby(by='itemDescription').size().reset_index(name='Frequency').sort_values(by='Frequency', ascending=False)
Item_Distribution.head()
```
- **groupby('itemDescription')**: Groups the data by the 'itemDescription' field to count how often each item was sold.
- **sort_values**: Sorts the results by frequency in descending order to see the most sold items.

---

#### 7. **Visualizing Top 10 Items Sold**
A bar graph is plotted to visualize the top 10 items based on the frequency sold.

```python
Bars = Item_Distribution['itemDescription'].head(10)
Height = Item_Distribution['Frequency'].head(10)
x_pos = np.arange(len(Bars))

plt.figure(figsize=(16,9))
plt.bar(x_pos, Height, color=(0.2, 0.3, 0.5, 0.5))
plt.title('Top 10 items')
plt.xlabel('Item names')
plt.ylabel('Number of Quant sold')
plt.xticks(x_pos, Bars)
plt.show()
```

- **plt.bar()**: Creates a bar chart to visualize the top 10 most frequently sold items.
- **plt.xticks()**: Labels the x-axis with the item names.

---

#### 8. **Resampling Data by Date**
We set the 'Date' column as the index for time-series analysis and plot the number of items sold per month.

```python
df_date = df.set_index(['Date'])
df_date.head()

df_date.resample('M')['itemDescription'].count().plot(figsize=(20,8), grid=True, title='No of items sold by month').set(xlabel='date', ylabel='number of items sold')
```

- **resample('M')**: Resamples the data by month.
- **plot()**: Plots a time-series graph showing the number of items sold per month.

---

#### 9. **Customer-Level Data Preparation**
For the association rule mining, we prepare customer-level data by grouping each customer's transactions.

```python
cust_level = df[["Member_number", "itemDescription"]].sort_values(by='Member_number', ascending=False)
cust_level['itemDescription'] = cust_level['itemDescription'].str.strip()
```

- **cust_level**: Data is sorted by `Member_number` and all trailing spaces are removed from the item descriptions.

---

#### 10. **Combining Transactions for Each Customer**
We combine the item transactions for each customer into lists.

```python
Transactions = [a[1]['itemDescription'].tolist() for a in list(cust_level.groupby(['Member_number']))]
```

- **Transactions**: A list of lists where each inner list contains the items bought by a specific customer.

---

#### 11. **Applying the Apriori Algorithm**
We apply the Apriori algorithm to identify frequent item sets and association rules.

```python
from apyori import apriori
rules = apriori(transactions=Transactions, min_support=0.002, min_confidence=0.05, min_lift=3, min_length=2)
results = list(rules)
```

- **apriori()**: Applies the Apriori algorithm with minimum support of 0.002, confidence of 0.05, and lift of 3.
- **results**: The association rules generated from the Apriori algorithm.

---

#### 12. **Extracting and Displaying Results**
We extract the left-hand side, right-hand side, support, confidence, and lift from the results and display the top 10 rules based on lift.

```python
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

results_in_dataframe = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidences', 'Lift'])
results_in_dataframe.nlargest(n=10, columns='Lift').reset_index()
```

- **inspect()**: Extracts relevant metrics from the rules.
- **results_in_dataframe**: Stores the association rules in a DataFrame.
- **nlargest()**: Selects the top 10 rules based on the lift.

---

#### Conclusion
This script provides an end-to-end analysis of the grocery sales dataset using the Apriori algorithm. It includes data preprocessing, visualization, and association rule mining to identify item combinations frequently bought together by customers.
