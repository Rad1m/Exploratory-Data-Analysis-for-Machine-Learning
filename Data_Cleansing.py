import skillsnetwork
import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as pltconda
import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats

# read the data into pandas data frame and look at the first 5 rows using the head() method
housing = pd.read_csv('data\Ames_Housing_Data1.tsv', sep='\t')
housing.head(10)

# find more information about the features and types using the info() method
housing.info()

# let's use the describe() function to show the count, mean, min, max of the sale price attribute.
housing["SalePrice"].describe()

housing["Sale Condition"].value_counts()

# Before proceeding with the data cleaning,
# it is useful to establish a correlation between the response variable
# (in our case the sale price) and other predictor variables,
# as some of them might not have any major impact in determining the price of the house
# and will not be used in the analysis. There are many ways to discover correlation between the target variable
# and the rest of the features.
# Building pair plots, scatter plots, heat maps, and a correlation matrixes are the most common ones.
# Below, we will use the corr() function to list the top features based
# on the pearson correlation coefficient (measures how closely two sequences of numbers are correlated).
# Correlation coefficient can only be calculated on the numerical attributes (floats and integers),
# therefore, only the numeric attributes will be selected.

hous_num = housing.select_dtypes(include = ['float64', 'int64'])
hous_num_corr = hous_num.corr()['SalePrice'][:-1] # -1 means that the latest row is SalePrice
top_features = hous_num_corr[abs(hous_num_corr) > 0.5].sort_values(ascending=False) #displays pearsons correlation coefficient greater than 0.5
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))


# let's generate some par plots to visually inspect the correlation
# between some of these features and the target variable.
# We will use seaborns sns.pairplot() function for this analysis.
# Also, building pair plots is one of the possible ways to spot the outliers that might be present in the data.

for i in range(0, len(hous_num.columns), 5):
    sns.pairplot(data=hous_num,
                x_vars=hous_num.columns[i:i+5],
                y_vars=['SalePrice'])
    
# inspect whether our 'SalePrice' data are normally distributed.
# The assumption of the normal distribution must be met in order to perform any type of regression analysis.
# There are several ways to check for this assumption, however here, we will use the visual method,
# by plotting the 'SalePrice' distribution using the distplot() function from the seaborn library

sp_untransformed = sns.displot(housing['SalePrice'])

# use the skew() function to calculate our skewness level of the SalePrice
print("Skewness: %f" % housing['SalePrice'].skew())

# we can try to transform our data, so it looks more normally distributed.
# We can use the np.log() function from the numpy library to perform log transform.
# This documentation contains more information about the numpy log transform.
log_transformed = np.log(housing['SalePrice'])
sp_transformed = sns.displot(log_transformed)

# the log method transformed the 'SalePrice' distribution into a more symmetrical bell curve and the skewness level now is -0.01, well within the range.
print("Skewness: %f" % (log_transformed).skew())

# visually inspect the 'Lot Area' feature. If there is any skewness present, apply log transform to make it more normally distributed.
la_plot = sns.displot(housing['Lot Area'])
print("Skewness: %f" % housing['Lot Area'].skew())
la_log = np.log(housing['Lot Area'])
print("Skewness: %f" % la_log.skew())

# use pandas duplicated() function and search by the 'PID' column, which contains a unique index number for each entry.
duplicate = housing[housing.duplicated(['PID'])]
duplicate

# there is one duplicate row in this dataset.
# To remove it, we can use pandas drop_duplicates() function.
# By default, it removes all duplicate rows based on all the columns.
dup_removed = housing.drop_duplicates()
dup_removed

# An alternative way to check if there are any duplicated Indexes in our dataset is using index.is_unique function
housing.index.is_unique

# remove duplicates on a specific column by setting the subset equal to the column that contains the duplicate, such as 'Order'.
removed_sub = housing.drop_duplicates(subset=['Order'])

# To summarize all the missing values in our dataset, we will use isnull() function.
# Then, we will add them all up, by using sum() function, sort them with sort_values() function,
# and plot the first 20 columns (as the majority of our missing values fall within first 20 columns),
# using the bar plot function from the matplotlib library.

total = housing.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)

# let's look at 'Mas Vnr Area' feature and replace the missing values with the mean value of that column.
mean = housing["Mas Vnr Area"].mean()
housing["Mas Vnr Area"].fillna(mean, inplace = True)

# One of the most important transformations we need to apply to our data is feature scaling.
# There are two common ways to get all attributes to have the same scale: min-max scaling and standardization.
# Min-max scaling (or normalization) is the simplest:
# values are shifted and rescaled so they end up ranging from 0 to 1.
# This is done by subtracting the min value and dividing by the max minus min.
# Standardization is different: first it subtracts the mean value (so standardized values always have a zero mean),
# and then it divides by the standard deviation, so that the resulting distribution has unit variance.

# First, we will normalize our data.
norm_data = MinMaxScaler().fit_transform(hous_num)
norm_data

# we can also standardize our data
scaled_data = StandardScaler().fit_transform(hous_num)
scaled_data

# use StandardScaler() and fit_transform() functions to standardize the 'SalePrice' feature only.
# scaled_sprice = StandardScaler().fit_transform(housing['SalePrice'][:,np.newaxis])
scaled_sprice = StandardScaler().fit_transform(np.array(housing['SalePrice'])[:, None])
scaled_sprice

# Uni-variate Analysis
# A box plot is a method for graphically depicting groups of numerical data through their quartiles
# Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles.
# Outliers may be plotted as individual points. 
sns.boxplot(x=housing['Lot Area'])
sns.boxplot(x=housing['SalePrice'])

# Bi-variate Analysis
# Next, we will look at the bi-variate analysis of the two features, the sale price, 'SalePrice',
# and the ground living area, 'GrLivArea', and plot the scatter plot of the relationship between these two parameters
price_area = housing.plot.scatter(x='Gr Liv Area',
                      y='SalePrice')

# Deleting the Outliers
# First, we will sort all of our 'Gr Liv Area' values and select only the last two.
housing.sort_values(by = 'Gr Liv Area', ascending = False)[:2]

# Now we will use the pandas drop() function to remove these two rows.
outliers_dropped = housing.drop(housing.index[[1499,2181]])
new_plot = outliers_dropped.plot.scatter(x='Gr Liv Area',
                                         y='SalePrice')

# determine whether there are any outliers in the 'Lot Area' feature
sns.boxplot(x=housing['Lot Area'])
price_lot = housing.plot.scatter(x='Lot Area', y='SalePrice')   
housing['Lot_Area_Stats'] = stats.zscore(housing['Lot Area'])
housing[['Lot Area','Lot_Area_Stats']].describe().round(3)
housing.sort_values(by = 'Lot Area', ascending = False)[:1]
lot_area_rem = housing.drop(housing.index[[957]])

# Z-score is another way to identify outliers mathematically.
housing['LQFSF_Stats'] = stats.zscore(housing['Low Qual Fin SF'])
housing[['Low Qual Fin SF','LQFSF_Stats']].describe().round(3)



