# Analysis-of-the-Russian-bond-market

**Implemented functionality**

1. Determination of bond yields according to MICEX data
2. An assessment of ruble bond market by distribution of coupon yield
3. Division into clusters by coupon yield
4. Graphical representation of the clusters obtained 

**Project logic:**

1.Uploading dataset 
2.Data preprocessing 
3.Calculation of coupon yield including tax, accumulated coupon income, future coupon income, broker and exchange commission
4.Assessment of ruble bond market 
5.1D Clustering using defferent distribution functions from KDE algorithm

**Clustering algorithm:**

Kernel Density Estimation (KDE)
https://ru.wikipedia.org/wiki/Ядерная_оценка_плотности

**Assumptions**:

1.Analysis of ruble bonds only
2.Bonds withput information on price and length of coupon period are not taken into account
3.Bond with yield >= 100 are not taken into account

**Results:**

MEAN_INITIAL_NOMINAL_VALUE     63667.232598
MEAN_COUPONPERCENT             7.110484
MEAN_PRICE_RUB                 60438.205637
MEAN_YIELD                     10.916718

![clusters](https://user-images.githubusercontent.com/89735790/152028097-871aa8d8-f669-4f54-85a5-908966120f03.jpg)


