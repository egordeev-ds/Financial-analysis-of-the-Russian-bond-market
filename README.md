# Analysis-of-the-Russian-bond-market

## **Implemented functionality**

1. Determination of bond yields according to MICEX data
2. An assessment of ruble bond market by distribution of coupon yield
3. Division into clusters by coupon yield
4. Graphical representation of the clusters obtained 

## **Project logic:**

1. Uploading dataset 
2. Data preprocessing 
3. Calculation of coupon yield including tax, accumulated coupon income, future coupon income, broker and exchange commission
4. Assessment of ruble bond market 
5. 1D Clustering using defferent distribution functions from KDE algorithm

## **Clustering algorithm:**

Kernel Density Estimation (KDE)
https://ru.wikipedia.org/wiki/Ядерная_оценка_плотности

## **Assumptions**:

1. Analysis of ruble bonds only
2. Bonds withput information on price and length of coupon period are not taken into account
3. Bonds with a difference in price and face value >= 20% are not taken into account

## **Results:**

### Market Assessment:
<table>
  <tr>
    <th>Indicator</th>
    <th>Value</th> 
  </tr>
  <tr>
    <td>MEAN_INITIAL_NOMINAL_VALUE
    <td>61686.350436</td> 
  </tr>
  <tr>
    <td>MEAN_COUPONPERCENT</td>
    <td>7.223393</td> 
  </tr>
    <tr>
    <td>MEAN_PRICE_RUB</td>
    <td>58943.164298</td> 
  </tr>
    </tr>
    <tr>
    <td>MEAN_YIELD</td>
    <td>8.268599</td> 
  </tr>
</table>

### Clustering:
Yield turned out to be divided into 21 intervals, their borders are the red dots on the x axis
![clusters](https://github.com/egordeev-ds/Analysis-of-the-Russian-bond-market/blob/master/YIELD%20Clusters.jpg)



