import pandas as pd
import numpy as np

from datetime import datetime,timedelta, date
import requests
import io

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns 

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

import warnings
warnings.filterwarnings("ignore")

### 0. Dataset preparation

#dataset loading
bond_url = 'https://iss.moex.com/iss/apps/infogrid/emission/rates.csv?iss.dp=comma&iss.df=%25d.%25m.%25Y&iss.tf=%25H:%25M:%25S&iss.dtf=%25d.%25m.%25Y%25H:%25M:%25S&iss.only=rates&limit=unlimited&lang=ru'
s=requests.get(bond_url).content
df = pd.read_csv(io.StringIO(s.decode('windows-1251')), sep=';', header=1)

### 1. Data cleaning

# 1. Choosing bonds in RUB

#available currencies
print('Available currencies:\n', df['FACEUNIT'].value_counts())
df_2 = df[df['FACEUNIT'] == 'RUB'].reset_index(drop = True)

# 2. Selecting relevant columns

#available columns
print('Available columns:',[col for col in df.columns])

#renaming for convenience
df_2.rename(columns = {'SECID':'ID', 
                       'FACEVALUE':'NOMINAL_VALUE',
                       'INITIALFACEVALUE':'INITIAL_NOMINAL_VALUE',
                       'FACEUNIT':'NOMINAL_CURRENCY',
                       'COUPONDATE':'COUPONDATE_NEXT',
                       'INN':'EMITENT_INN',
                       'MATDATE':'REDEMPTIONDATE'}, inplace = True)

#selecting relevant columns
df_2 = df_2[['EMITENTNAME', 'NAME','INITIAL_NOMINAL_VALUE',
             'COUPONPERCENT','COUPONVALUE','COUPONFREQUENCY',
             'COUPONDATE_NEXT','COUPONDAYSPASSED','COUPONDAYSREMAIN',
             'COUPONLENGTH','PRICE_RUB','ISSUEDATE','REDEMPTIONDATE',
             'DAYSTOREDEMPTION','HIGH_RISK']]

# 3. Translation to the correct data format

#data types before correction
print('Data types before correction:\n', df_2.dtypes, '\n')

#a function that corrects data types
def type_changer(x):
    if isinstance(x,str):
        x_new_type = float(x.replace(',','.'))
    else:
        x_new_type = x
    return x_new_type
    
#converting string data to numeric data
for col in ['INITIAL_NOMINAL_VALUE', 'COUPONPERCENT', 'COUPONVALUE',
            'COUPONDAYSPASSED', 'COUPONDAYSREMAIN','COUPONLENGTH', 'PRICE_RUB']:
    df_2[col] = df_2[col].apply(type_changer)
    
#converting string data to datetime
for col in ['COUPONDATE_NEXT','ISSUEDATE', 'REDEMPTIONDATE']:
    df_2[col] = pd.to_datetime(df_2[col])

#data types after correction
print('Data types after correction:\n',df_2.dtypes)

# 4. Missings processing

#n_samples before missings processing
samples_before = df_2.shape[0]
print('Missings before processing:\n',df_2.isnull().sum(), '\n')

#replacement missings of coupon value in relative 
df_2['COUPONPERCENT'].replace(np.nan,0,inplace = True) # missings may mean a coupon-free bond

#replacement missings of coupon value in absolute 
condition = (df_2['COUPONPERCENT'].notna()) & (df_2['COUPONVALUE'].isna()) & (df_2['COUPONFREQUENCY'].notna())
df_2['COUPONVALUE'].loc[condition] = ((df_2['COUPONPERCENT']*df_2['INITIAL_NOMINAL_VALUE'])/df_2['COUPONFREQUENCY'])/100 # COUPONVALUE - вычисляем по формуле

#deleting missings in other columns
for col in ['COUPONFREQUENCY','COUPONDATE_NEXT','COUPONDAYSPASSED',
            'COUPONDAYSREMAIN','COUPONLENGTH','PRICE_RUB',
            'ISSUEDATE','REDEMPTIONDATE','DAYSTOREDEMPTION']:
    
    df_2 = df_2[df_2[col].isnull() == False].reset_index(drop = True)
    
#n_samples after missing processing
print('Missing after processing:\n',df_2.isnull().sum())

# 5. Adding necessary fields

#bond maturity
df_2['DAYSTOREDEMPTION_SINCE_START'] = df_2['REDEMPTIONDATE'] - df_2['ISSUEDATE'] 
df_2['DAYSTOREDEMPTION_SINCE_START'] = df_2['DAYSTOREDEMPTION_SINCE_START'].apply(lambda x: x.days)

#today date
df_2['TODAY'] = date.today() 
df_2['TODAY'] = pd.to_datetime(df_2['TODAY'])

#date of coupon payment
df_2['COUPONDATE_LAST'] = df_2.apply(lambda x: x['TODAY'] - timedelta(days = x['COUPONDAYSPASSED']), axis = 1) 

# 6. Removing errors in calculating coupon payment date

#deleting rows where date of next coupon payment is incorrectly calculated
df_3 = df_2.copy()
df_3['is_mistake_in_coupon_date'] = df_3.apply(lambda x: 0 if x['COUPONDATE_NEXT'] - timedelta(days = x['COUPONDAYSREMAIN']) == x['TODAY'] else 1, axis = 1)
df_3 = df_3[df_3['is_mistake_in_coupon_date'] == 0].reset_index(drop = True)

#deleted rows
samples_after = df_3.shape[0]
print('Deleted samples:', samples_before - samples_after, '\n')

# 7. Exclude bonds with a difference in price and face value >= 20%

df_3 = df_3[df_3['PRICE_RUB']>= (df_3['INITIAL_NOMINAL_VALUE']*0.8)].reset_index(drop = True)

# 8. Forming the final processed dataset with convenient arrangement of columns

df_3 = df_3[['EMITENTNAME','NAME','INITIAL_NOMINAL_VALUE','COUPONPERCENT','COUPONVALUE',
             'COUPONFREQUENCY','COUPONDATE_LAST','COUPONDAYSPASSED','TODAY','COUPONDAYSREMAIN',
             'COUPONDATE_NEXT','COUPONLENGTH','PRICE_RUB','ISSUEDATE','REDEMPTIONDATE',
             'DAYSTOREDEMPTION_SINCE_START','DAYSTOREDEMPTION','HIGH_RISK']]
df_3.head()             

### 2. Calculation of full coupon yield

#accumulated coupon income(ACI)
df_3['ACI_period'] = df_3['COUPONDAYSPASSED']/df_3['COUPONLENGTH']
df_3['ACI'] = df_3['ACI_period'] * (((df_3['COUPONPERCENT']/100)/df_3['COUPONFREQUENCY']) * df_3['INITIAL_NOMINAL_VALUE'])

#future coupon income(FCI)
df_3['FCI_period'] = 365/df_3['COUPONLENGTH'] #the calculation period is 365 days from the current date
df_3['FCI'] = df_3['FCI_period'] * (((df_3['COUPONPERCENT']/100)/df_3['COUPONFREQUENCY']) * df_3['INITIAL_NOMINAL_VALUE']) * 0.87

#tax
df_3['TAX'] = 0.87
df_3['TAX'].loc[df_3['INITIAL_NOMINAL_VALUE'] <= df_3['PRICE_RUB']] = 1

#input
N = df_3['INITIAL_NOMINAL_VALUE']
P = df_3['PRICE_RUB']
tax = df_3['TAX']
FCI = df_3['FCI']
ACI = df_3['ACI']

#broker and exchange commissions
broker_com = 0.06/100
exchange_com = 0.0125/100

#calculation
df_3['PROFIT'] = ((N-P)*tax - ACI + 0.87*FCI)/(1+broker_com+exchange_com)
df_3['COSTS'] = P+ACI
df_3['YIELD_FULL'] = (df_3['PROFIT']/df_3['COSTS']) * 100

##### Emissions analysis

#boxplot of full coupon yield
plt.style.use('fivethirtyeight')
plt.figure(figsize = (15,4))
sns.boxplot(df_3['YIELD_FULL'])
plt.title('Emissions analysis, step one')
plt.xticks(np.linspace(-30,30,13))
plt.show()

# Removing emissions with a yield greater lower 1% quantile and higher 99% quantile (emissions) 
q_1 = np.quantile(df_3['YIELD_FULL'], 0.01)
q_99 = np.quantile(df_3['YIELD_FULL'], 0.99)
df_4 = df_3[(df_3['YIELD_FULL'] >= q_1) & (df_3['YIELD_FULL'] <= q_99)].reset_index(drop = True)

# boxplot of full coupon yield after removing emissions
plt.figure(figsize = (15,4))
sns.boxplot(df_4['YIELD_FULL'])
plt.title("Emissions analysis, step two")
plt.show()

# Coclusion: Only three emissions left, other bons are within 1- 99% interval

### 3. Ruble bond market

# issuers with the most bonds
df_4_emitents = df_4.groupby('EMITENTNAME').agg({'NAME':'count'}).reset_index().sort_values('NAME', ascending = False)

# average nominal value, coupon, purchase price, maturity and full yield to maturity by market
df_4.describe().loc['mean'][['INITIAL_NOMINAL_VALUE',
                             'COUPONPERCENT',
                             'PRICE_RUB',
                             'DAYSTOREDEMPTION_SINCE_START',
                             'YIELD_FULL']]

#### 3.1 Distributions

#distribution of COUPONPERCENT
plt.style.use('fivethirtyeight')
df_4['COUPONPERCENT'].plot(kind = 'hist',
                           figsize = (15,4),
                           bins = 200,
                           title = 'Distribution of COUPONPERCENT',
                           xlabel = 'COUPONPERCENT')
plt.show()

#distribution of full coupon yield (hist)
df_4['YIELD_FULL'].plot(kind = 'hist',
                   figsize = (15,4),
                   bins = 200,
                   title = 'Distribution of full coupon yield',
                   xlabel = 'YIELD_FULL')
plt.show()

### 3. Full coupon yield clustering

#input
X = df_4['YIELD_FULL'].values.reshape(-1, 1)

#### 3.1 distribution of full coupon yield (density)

#plot params
plt.figure(figsize = (15,4))
true_dens = sns.distplot(df_4['YIELD_FULL'],bins = 200)
plt.title(f'Distplot of YIELD')
plt.show()

# Coclusion: Distribution is close to normal

#### 3.2 Available kernel estimators(K)

#input
X_plot = np.linspace(np.min(X), np.max(X), len(X)).reshape(-1,1)
X_src = np.zeros((1, 1))
kernels = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]

#plot params
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)

def format_func(x, loc):
    if x == 0:
        return "0"
    elif x == 1:
        return "h"
    elif x == -1:
        return "-h"
    else:
        return "%ih" % x

#plotting available kernels
for i, kernel in enumerate(kernels):
    
    axi = ax.ravel()[i]
    kde = KernelDensity(kernel=kernel).fit(X_src)
    log_dens = kde.score_samples(X_plot)
    
    axi.fill(X_plot, np.exp(log_dens), "-k", fc="#AAAAFF")
    axi.text(-2.6, 0.95, kernel)

    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
    axi.yaxis.set_major_locator(plt.NullLocator())

    axi.set_ylim(0, 1.05)
    axi.set_xlim(-2.9, 2.9)

fig.set_size_inches([12,4])
ax[0, 1].set_title("Available Kernels")
plt.show()

#### 3.3 Kernel estimators of YIELD_FULL (1/nh * sum(K((x-x_i/h))))

#input
X_plot = np.linspace(np.min(X), np.max(X), len(X)).reshape(-1,1)
h = 1.06 * X.std() * (len(X)**(-1/5)) #bandwidth

colors = ["navy", "cornflowerblue", "darkorange",'red','green','blue']
kernels = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]

#plot true density
fig, ax = plt.subplots()
sns.distplot(X, bins = 20, norm_hist = True, color = 'black', label = 'true_dens' )

#plot kernel density
for color, kernel in zip(colors, kernels):
    
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(X)
    log_dens = kde.score_samples(X_plot)
    
    ax.plot(X_plot,np.exp(log_dens),color=color,linestyle="-",label=f'kernel = {kernel}')
    
fig.set_size_inches([12,4])
ax.set_xlabel('YIELD')
ax.set_ylabel('density')
ax.set_title('Distributions by different kernels')
ax.legend(loc='best', fontsize = 'xx-small')

plt.show()

# Coclusion: Judging by the plot, cosine best estimates the distribution

#### 3.4 Clustering with different kernel estimators. Chossing the best

#input
rows = len(kernels)
cols = 1
axs = plt.figure(figsize=(15,30),
                 constrained_layout=True).subplots(rows,cols,sharex=True,sharey=True)

#plot clustering with each kernel estimator
for ax, kernel in zip(axs,kernels):
    
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(X)
    log_dens = kde.score_samples(X_plot)
    min_, max_ = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens, np.greater)[0]
    
    print(f'Grouping kernel {kernel}:')
    
    if len(min_) > 1:
        print(X_plot[min_])
        ax.set_title('kernel=%s' % str(kernel))
        ax.plot(X_plot, log_dens, 'black', X_plot[min_], log_dens[min_], 'ro')
        
    else:
        print('error')

#### 3.5 Clustering with the best kernel estimator

#Calculation of boundaries of clustering intervals
kde = KernelDensity(kernel='cosine', bandwidth=h).fit(X)
log_dens = kde.score_samples(X_plot)
min_, max_ = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens, np.greater)[0]

#Plot a graph visualizing clustering
fig, ax = plt.subplots()

ax.plot(X_plot[:min_[0]+1], log_dens[:min_[0]+1])
for i in range(len(min_)-1):
    ax.plot(X_plot[min_[i]:min_[i+1]+1], log_dens[min_[i]:min_[i+1]+1])

ax.plot(X_plot[min_[len(min_)-1]:], log_dens[min_[len(min_)-1]:])
ax.plot(X_plot[min_], log_dens[min_], 'ro')

fig.set_size_inches([15,4])
ax.set_xlabel('YIELD')
ax.set_title('Clustering YIELD_FULL')
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

plt.show()

#output of boundaries
intervals = [round(i,2) for i in X_plot[min_].reshape(1,-1)[0]]
print('Clustering boundaries:\n',intervals)