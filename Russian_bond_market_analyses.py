#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from datetime import datetime,timedelta, date
import requests
import io

import matplotlib.pyplot as plt
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

import warnings
warnings.filterwarnings("ignore")


# ### 0. Подготовка датасета

# In[2]:


#загрузка данных
bond_url = 'https://iss.moex.com/iss/apps/infogrid/emission/rates.csv?iss.dp=comma&iss.df=%25d.%25m.%25Y&iss.tf=%25H:%25M:%25S&iss.dtf=%25d.%25m.%25Y%25H:%25M:%25S&iss.only=rates&limit=unlimited&lang=ru'
s=requests.get(bond_url).content
df = pd.read_csv(io.StringIO(s.decode('windows-1251')), sep=';', header=1)


# ### 1. Очистка данных

# In[3]:


# 1. Выбор рублевых облигаций
print('Доступные валюты:', '\n', df['FACEUNIT'].value_counts(), '\n') #смотрим облигации в каких валютах у нас есть
df_2 = df[df['FACEUNIT'] == 'RUB'].reset_index(drop = True) #выбираем рублевые облигации

# 2. Выбор релевантных колонок
print('Доступные колонки:','\n', df.columns, '\n')
df_2.rename(columns = {'SECID':'ID', #переименовывание для удобства
                       'FACEVALUE':'NOMINAL_VALUE',
                       'INITIALFACEVALUE':'INITIAL_NOMINAL_VALUE',
                       'FACEUNIT':'NOMINAL_CURRENCY',
                       'COUPONDATE':'COUPONDATE_NEXT',
                       'INN':'EMITENT_INN',
                       'MATDATE':'REDEMPTIONDATE'}, inplace = True)

df_2 = df_2[['EMITENTNAME', #выбор нужных полей
             'NAME',
             'INITIAL_NOMINAL_VALUE',
             'COUPONPERCENT',
             'COUPONVALUE',
             'COUPONFREQUENCY',
             'COUPONDATE_NEXT',
             'COUPONDAYSPASSED',
             'COUPONDAYSREMAIN',
             'COUPONLENGTH',
             'PRICE_RUB',
             'ISSUEDATE',
             'REDEMPTIONDATE',
             'DAYSTOREDEMPTION',
             'HIGH_RISK']]

# 3. Перевод в корректный формат данных
print('Типы данных:\n',df_2.dtypes, '\n') # смотрим какие форматы были изначально
df_2['EMITENTNAME'] = df_2['EMITENTNAME'].astype('object') # перевод в форматы python
df_2['NAME'] = df_2['NAME'].astype('object')
df_2['INITIAL_NOMINAL_VALUE'] = df_2['INITIAL_NOMINAL_VALUE'].apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
df_2['COUPONPERCENT'] = df_2['COUPONPERCENT'].apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
df_2['COUPONVALUE'] = df_2['COUPONVALUE'].apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
df_2['COUPONFREQUENCY'] = df_2['COUPONFREQUENCY']
df_2['COUPONDATE_NEXT'] = pd.to_datetime(df_2['COUPONDATE_NEXT'])
df_2['COUPONDAYSPASSED'] = df_2['COUPONDAYSPASSED'].apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
df_2['COUPONDAYSREMAIN'] = df_2['COUPONDAYSREMAIN'].apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
df_2['COUPONLENGTH'] = df_2['COUPONLENGTH'].apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
df_2['PRICE_RUB'] = df_2['PRICE_RUB'].apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
df_2['ISSUEDATE'] = pd.to_datetime(df_2['ISSUEDATE'])
df_2['REDEMPTIONDATE'] = pd.to_datetime(df_2['REDEMPTIONDATE'])
df_2['DAYSTOREDEMPTION'] = df_2['DAYSTOREDEMPTION'].astype('float')
df_2['HIGH_RISK'] = df_2['HIGH_RISK']


# 4. Обработка пропусков
samples_before = df_2.shape[0]
print('Просмотр пропусков до обработки:\n',df_2.isnull().sum(), '\n') #просмотр пропусков до обработки

# EMITENTNAME - пропуски необходимо оставить
df_2['EMITENTNAME'] = df_2['EMITENTNAME']
# COUPONPERCENT - пропуск означает бескупонную облигацию
df_2['COUPONPERCENT'].replace(np.nan,0,inplace = True)
# COUPONFREQUENCY - удалим такие наблюдения
df_2 = df_2[df_2['COUPONFREQUENCY'].notna()].reset_index(drop = True)
# COUPONVALUE - вычисляем по формуле
condition = (df_2['COUPONPERCENT'].notna()) & (df_2['COUPONVALUE'].isna()) & (df_2['COUPONFREQUENCY'].notna())
df_2['COUPONVALUE'].loc[condition] = (df_2['COUPONPERCENT']*df_2['INITIAL_NOMINAL_VALUE'])/df_2['COUPONFREQUENCY']
#COUPONDATE_NEXT - удалим такие наблюдения
df_2 = df_2[df_2['COUPONDATE_NEXT'].notna()].reset_index(drop = True)
#COUPONDAYSPASSED - удалим такие наблюдения
df_2 = df_2[df_2['COUPONDAYSPASSED'].notna()].reset_index(drop = True)
#COUPONDAYSREMAIN - удалим такие наблюдения
df_2 = df_2[df_2['COUPONDAYSREMAIN'].notna()].reset_index(drop = True)
#COUPONLENGTH - удалим такие наблюдения
df_2 = df_2[df_2['COUPONLENGTH'].notna()].reset_index(drop = True)
# PRICE_RUB - удалим такие наблюдения
df_2 = df_2[df_2['PRICE_RUB'].notna()].reset_index(drop = True)
# ISSUEDATE - удалим такие наблюдения
df_2 = df_2[df_2['ISSUEDATE'].notna()].reset_index(drop = True)
# REDEMPTIONDATE - бессрочные облигации. Удалим такие наблюдения
df_2 = df_2[df_2['REDEMPTIONDATE'].notna()].reset_index(drop = True)
# DAYSTOREDEMPTION - бессрочные облигации. Удалим такие наблюдения
df_2 = df_2[df_2['DAYSTOREDEMPTION'].notna()].reset_index(drop = True)

print('Просмотр пропусков после обработки:\n',df_2.isnull().sum(), '\n') #просмотр пропусков после обработки

# 5. Добавление нужных полей
df_2['DAYSTOREDEMPTION_SINCE_START'] = df_2['REDEMPTIONDATE'] - df_2['ISSUEDATE']
df_2['DAYSTOREDEMPTION_SINCE_START'] = df_2['DAYSTOREDEMPTION_SINCE_START'].apply(lambda x: x.days)

df_2['TODAY'] = date.today()
df_2['TODAY'] = pd.to_datetime(df_2['TODAY'])

df_2['COUPONDATE_LAST'] = df_2.apply(lambda x: x['TODAY'] - timedelta(days = x['COUPONDAYSPASSED']), axis = 1)

# 6. Уберем ошибки в расчете даты выплаты купона
df_3 = df_2.copy()
df_3['is_mistake_in_coupon_date'] = df_3.apply(lambda x: 0 if x['COUPONDATE_NEXT'] - timedelta(days = x['COUPONDAYSREMAIN']) == x['TODAY'] else 1, axis = 1)
df_3 = df_3[df_3['is_mistake_in_coupon_date'] == 0].reset_index(drop = True)

samples_after = df_3.shape[0]
print('Удалено строк:', samples_before - samples_after, '\n')

# 7. Сформируем итогвоый датасет с удобным расположением полей
df_3 = df_3[['EMITENTNAME',
             'NAME',
             'INITIAL_NOMINAL_VALUE',
             'COUPONPERCENT',
             'COUPONVALUE',
             'COUPONFREQUENCY',
             'COUPONDATE_LAST',
             'COUPONDAYSPASSED',
             'TODAY',
             'COUPONDAYSREMAIN',
             'COUPONDATE_NEXT',
             'COUPONLENGTH',
             'PRICE_RUB',
             'ISSUEDATE',
             'REDEMPTIONDATE',
             'DAYSTOREDEMPTION_SINCE_START',
             'DAYSTOREDEMPTION',
             'HIGH_RISK']]


# ### 2. Расчет полной купонной доходности к погашению

# In[4]:


# accumulated coupon income(ACI)
df_3['ACI_period'] = df_3['COUPONDAYSPASSED']/df_3['COUPONLENGTH'] # доля периода выплаты купона (накопелнная)
#df_3 = df_3[df_3['ACI_period'] <= 0.2].reset_index(drop = True) # выбор значений <= 0.2
df_3['ACI'] = df_3['ACI_period'] * (((df_3['COUPONPERCENT']/100)/df_3['COUPONFREQUENCY']) * df_3['INITIAL_NOMINAL_VALUE'])

# future coupon income(FCI)
df_3['FCI_period'] = 365/df_3['COUPONLENGTH'] #период расчета 365 дней с текущей даты
df_3['FCI'] = df_3['FCI_period'] * (((df_3['COUPONPERCENT']/100)/df_3['COUPONFREQUENCY']) * df_3['INITIAL_NOMINAL_VALUE']) * 0.87

# Налог
df_3['TAX'] = 0.87
df_3['TAX'].loc[df_3['INITIAL_NOMINAL_VALUE'] <= df_3['PRICE_RUB']] = 1


# In[5]:


# входные параметры
N = df_3['INITIAL_NOMINAL_VALUE']
P = df_3['PRICE_RUB']
tax = df_3['TAX']
FCI = df_3['FCI']
ACI = df_3['ACI']

broker_com = 0.06/100
exchange_com = 0.0125/100

# расчет
df_3['PROFIT'] = ((N-P)*tax + (FCI-ACI))/(1+broker_com+exchange_com)
df_3['COSTS'] = P+ACI
df_3['YIELD_FULL'] = (df_3['PROFIT']/df_3['COSTS']) * 100


# In[6]:


# boxplot полной доходности к погашению
plt.style.use('fivethirtyeight')
plt.figure(figsize = (15,4))
sns.boxplot(df_3['YIELD_FULL'])
plt.title('Анализ выбросов, шаг первый')
plt.show()

#удалим выбросы с доходностью больше 100
df_4 = df_3[(df_3['YIELD_FULL'] <= 100)].reset_index(drop = True)


# In[7]:


# boxplot полной доходности к погашению после удаления выбросов
plt.figure(figsize = (15,4))
sns.boxplot(df_4['YIELD_FULL'])
plt.title("Анализ выбросов, шаг второй")
plt.show()


# ### 3. Рынок рублевых облигаций

# In[8]:


# кто выпускал больше всех облигаций
df_4_emitents = df_4.groupby('EMITENTNAME').agg({'NAME':'count'}).reset_index().sort_values('NAME', ascending = False)


# In[9]:


# средняя номинальная стоимость, купон, цена выкупа, срок погашения, полная и текущая доходность к погашению
df_4.describe().loc['mean'][['INITIAL_NOMINAL_VALUE',
                             'COUPONPERCENT',
                             'PRICE_RUB',
                             'DAYSTOREDEMPTION_SINCE_START',
                             'YIELD_FULL']]


# #### 3.1 Распредления

# In[10]:


# 1. распределение COUPONPERCENT
plt.style.use('fivethirtyeight')
df_4['COUPONPERCENT'].plot(kind = 'hist',
                           figsize = (15,4),
                           bins = 200,
                           title = 'Распределение COUPONPERCENT',
                           xlabel = 'COUPONPERCENT')


# In[11]:


# 2. Доходности к погашению (гистограмма)
df_4['YIELD_FULL'].plot(kind = 'hist',
                   figsize = (15,4),
                   bins = 200,
                   title = 'Распределение доходности к пошагению',
                   xlabel = 'YIELD_FULL')


# ### 3. Кластеризация по полной доходности к погашению

# In[12]:


#параметры купонной доходности
x_init = df_4['YIELD_FULL']
X = x_init.values.reshape(-1, 1)


# #### 3.1 Распределение доходности к погашению (плотность)

# In[13]:


#параметры графика
plt.figure(figsize = (15,4))
true_dens = sns.distplot(x_init,
                         bins = 20,
                         norm_hist = True).get_lines()[0].get_data()[1] #получение y точек плотности распредления 
plt.title(f'Distplot of {x_init.name}')
plt.show()


# #### 3.2 Возможные ядерные функции (K)

# In[14]:


# входные параметры
X_plot = np.linspace(-6, 6, len(X))[:, None]
X_src = np.zeros((1, 1))
kernels = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]

# параметры гарфика
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

# отрисовка плотностей доступных ядерных функций
for i, kernel in enumerate(kernels):
    
    axi = ax.ravel()[i]
    log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
    axi.fill(X_plot[:, 0], np.exp(log_dens), "-k", fc="#AAAAFF")
    axi.text(-2.6, 0.95, kernel)

    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
    axi.yaxis.set_major_locator(plt.NullLocator())

    axi.set_ylim(0, 1.05)
    axi.set_xlim(-2.9, 2.9)

ax[0, 1].set_title("Available Kernels")


# #### 3.3 Ядерный оценщик YIELD_FULL (1/nh * sum(K((x-x_i/h))))

# In[15]:


# входные параметры
X_plot = np.linspace(np.min(X), np.max(X), len(true_dens))
h = 1.06 * x_init.std() * (len(x_init)**(-1/5)) #ширина полосы
colors = ["navy", "cornflowerblue", "darkorange",'red','green','blue']
kernels = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]

# параметры графика
plt.figure()
fig, ax = plt.subplots()
ax.fill(X_plot.reshape(-1, 1)[:, 0],
        true_dens,
        fc="black", 
        alpha=0.2,
        label="input distribution")

lw = 2

#отрисовка плотности каждого ядерного оценщика
for color, kernel in zip(colors, kernels):
    
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(X)
    log_dens = kde.score_samples(X_plot.reshape(-1, 1))
    
    ax.plot(X_plot.reshape(-1, 1)[:, 0],
            np.exp(log_dens),
            color=color,
            lw=lw,
            linestyle="-",
            label=f'kernel = {kernel}')
    
ax.legend(loc='best', fontsize = 'xx-small')
plt.show()


# #### 3.4 Кластеризация при помощи разных kernels. выбор наилучшего.

# In[16]:


#общие параметры графика
rows = len(kernels)
cols = 1
axs = plt.figure(figsize=(15,30),
                 constrained_layout=True).subplots(rows,
                                                   cols,
                                                   sharex=True,
                                                   sharey=True)

#отрисовка кластеризации при помощи каждого kernel
for ax, kernel in zip(axs,kernels):
    
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(X)
    log_dens = kde.score_samples(X_plot.reshape(-1, 1))
    min_, max_ = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens, np.greater)[0]
    
    print(f'Grouping kernel {kernel}:')
    if len(min_) > 1:
        print(X_plot[min_])
        ax.set_title('kernel=%s' % str(kernel))
        ax.plot(X_plot, log_dens, 'black',
                 X_plot[min_], log_dens[min_], 'ro')
    else:
        print('error')


# #### 3.5 Кластеризация при помощи лучшего kernel: epanechnikov

# In[17]:


#расчет границ интервалов кластеризации
kde = KernelDensity(kernel='epanechnikov', bandwidth=h).fit(X)
s = X_plot  #для удобства задания параметров гарфика
e = kde.score_samples(s.reshape(-1, 1)) #для удобства задания параметров гарфика
mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

#отрисовка гарфика
plt.figure(figsize=(15,5))
plt.plot(s[:mi[0]+1], e[:mi[0]+1], 'red',
         s[mi[0]:mi[1]+1], e[mi[0]:mi[1]+1], 'green',
         s[mi[1]:mi[2]+1], e[mi[1]:mi[2]+1], 'blue',
         s[mi[2]:mi[3]+1], e[mi[2]:mi[3]+1], 'yellow',
         s[mi[3]:mi[4]+1], e[mi[3]:mi[4]+1], 'orange',
         s[mi[4]:mi[5]+1], e[mi[4]:mi[5]+1], 'purple',
         s[mi[5]:mi[6]+1], e[mi[5]:mi[6]+1], 'white',
         s[mi[6]:mi[7]+1], e[mi[6]:mi[7]+1], 'lightgreen',
         s[mi[7]:mi[8]+1], e[mi[7]:mi[8]+1], 'black',
         s[mi[8]:mi[9]+1], e[mi[8]:mi[9]+1], 'violet',
         s[mi[9]:mi[10]+1], e[mi[9]:mi[10]+1], 'purple',
         s[mi[10]:mi[11]+1], e[mi[10]:mi[11]+1], 'brown',
         s[mi[11]:], e[mi[11]:], 'gray',
         s[mi], e[mi], 'ro')
plt.title('Кластеризация YIELD_FULL')
plt.show()

