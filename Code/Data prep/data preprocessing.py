
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
#%%
global_path = '/Users/asmabaccouche/Heart disease/Heart-disease'
data_path = global_path+'/Data/Raw data/heart_disease.csv'
# Read data
df = pd.read_csv(data_path, index_col=0)
df['age_years'] = df['age']//365
df = df.drop(['age'], axis=1)
# check for null data
df.isnull().any()
#%%
train, test= train_test_split(df, test_size=0.2, random_state=42)
train.to_csv(global_path+'/Data/Modelling data/Train/train0.csv', index=False)
test.to_csv(global_path+'/Data/Modelling data/Test/test0.csv', index=False)
#%%
# check for NaN values
missing_values = ["n/a", "na", "--"]
df = pd.read_csv(data_path, na_values = missing_values, index_col=0)
df['age_years'] = df['age']//365
df = df.drop(['age'], axis=1)
df.isnull().any()
#%%
# adding a new column
df['bmi']=(df['weight']/(df['height'] **2) * 10000)
df['bmi']=df['bmi'].round(decimals = 1)
df = df.reset_index(drop=True)
#%%
# data descriotion
df.describe()
#%%
# Visualize raw data
def histogram_column(df, column_name):
    fig, ax = plt.subplots() 
    # count the occurrence of each class 
    data = df[column_name].value_counts() 
    # get x and y data 
    points = data.index 
    frequency = data.values 
    # create bar chart 
    ax.bar(points, frequency) 
    # set title and labels 
    ax.set_title(column_name + ' histogram') 
    ax.set_xlabel(column_name) 
    ax.set_ylabel('Frequency')

for i in range(len(df.columns)):
    histogram_column(df, df.columns[i])
    #df[df.columns[i]].value_counts().sort_index().plot.bar()
#%%
# Correlation matrix - all columns
sns.heatmap(df.corr())
#%%
# Correlation matrix - only numerical columns
#sns.pairplot(df)
#%%
# scatter plot of BMI
plt.figure(figsize=[8,5])
plt.scatter(data = df, x ='weight', y = 'bmi',alpha = 1/2)
#setting axis limits to view the plot distribution better
plt.ylim((10,50))
plt.xlim((40,140))
#setting title and axis labels
plt.title('Relationship between BMI and Weight')
plt.ylabel('BMI(kg/m2)')
plt.xlabel('Weight(kg)')
#%%
# label distribution
x= df['cardio'].unique()
y = df['cardio'].value_counts(sort = False)
plt.bar(x,y)
plt.title('Cardiovascular disease')
plt.xlabel('CVD')
plt.ylabel('Count')
plt.show()
#%%
plt.figure(figsize=[8,5])
cvd_markers =[[1,'o'],[0,'^']]
#looping through the markers to create a plot
for cvd, marker in cvd_markers:
    plot_data= df.loc[df['cardio']== cvd]
    sns.regplot(data = plot_data,x='bmi',y='weight',x_jitter = 0.05,fit_reg=False,marker=marker)
#setting title and axis labels
plt.title('Relationship between CVD, Weight and BMI')
plt.xlabel('BMI(kg/m2)')
plt.ylabel('Weight(kg)')
plt.legend(['CVD Yes','CVD No'])

#%%
# check for outliers
# 1) Visually
def boxplot_column(df, column_name):
    fig, ax = plt.subplots() 
    sns.boxplot(x=df[column_name], ax=ax)
    # set title and labels 
    ax.set_title(column_name + ' boxplot') 
    ax.set_xlabel(column_name) 

for i in range(len(df.columns)):
    boxplot_column(df, df.columns[i])

# 1) Height:
# one outlier at 250
# 2) weight:
# one outlier at 200, other less than 25
# 3) ap_hi:
# outliers when more than 10000
# 4) ap_lo:
# outliers when more than 6000
# 5) age_years:
# outlier when less or equal to 30
#%%
def scatterplot_column(df, column_name1, column_name2):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(df[column_name1], df[column_name2])
    ax.set_title(column_name1 + ' and ' + column_name2 + ' scatter plot') 
    ax.set_xlabel(column_name1)
    ax.set_ylabel(column_name2)
    plt.show()

for i in range(1, len(df.columns)-1):
    for j in range(i+1, len(df.columns)):
        scatterplot_column(df, df.columns[i], df.columns[j])

# 1) Height:
# one outlier at 250
# 2) ap_hi:
# outliers when more than 10000
# 3) ap_lo:
# outliers when more than 6000
#%%
outliers = (df['height']>=250)|(df['weight']>=200)|(df['weight']<=25)|(df['ap_hi']>=10000)|(df['ap_lo']>=5000)|(df['age_years']<=30)
# removing outliers
df_out1 = df.drop(df.index[[outliers]])
df_out1.to_csv(global_path+'/Data/Preprocessed data/data_clean1.csv', index=False)
#%%
# check for outliers
# 2) IQR Score (interquartile range)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
print(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any())

# removing outliers
df_out2 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)] 
df_out2.to_csv(global_path+'/Data/Preprocessed data/data_clean2.csv', index=False)
#%%
# check for outliers
# 3) z Score + threshold

z = np.abs(stats.zscore(df))
print(z)
threshold = 3
a = np.where(z > threshold)
# removing outliers
df_out3= df[(z < threshold).all(axis=1)]
df_out3.to_csv(global_path+'/Data/Preprocessed data/data_clean3.csv', index=False)
# %%
