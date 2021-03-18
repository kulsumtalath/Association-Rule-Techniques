```python
import pandas as pd
```


```python
dataset = [['Milk','Bread','Wheat Bread'],['Jam' , 'Bread','Butter'],['Milk','Jam','Bread','Butter'],['Jam','Butter']]
```


```python
dataset
```




    [['Milk', 'Bread', 'Wheat Bread'],
     ['Jam', 'Bread', 'Butter'],
     ['Milk', 'Jam', 'Bread', 'Butter'],
     ['Jam', 'Butter']]




```python
from mlxtend.preprocessing import TransactionEncoder  #converting text into boolean format
```


```python
te = TransactionEncoder()
```


```python
te_ary=te.fit(dataset).transform(dataset)
```


```python
df=pd.DataFrame(te_ary,columns=te.columns_)
print(df)
```

       Bread  Butter    Jam   Milk  Wheat Bread
    0   True   False  False   True         True
    1   True    True   True  False        False
    2   True    True   True   True        False
    3  False    True   True  False        False
    


```python
from mlxtend.frequent_patterns import apriori
```


```python
apriori(df,min_support=0.1) #0.1 is min support
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.75</td>
      <td>(0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>(1)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.75</td>
      <td>(2)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.50</td>
      <td>(3)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.25</td>
      <td>(4)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.50</td>
      <td>(0, 1)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.50</td>
      <td>(0, 2)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.50</td>
      <td>(0, 3)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.25</td>
      <td>(0, 4)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.75</td>
      <td>(1, 2)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.25</td>
      <td>(1, 3)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.25</td>
      <td>(2, 3)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.25</td>
      <td>(3, 4)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.50</td>
      <td>(0, 1, 2)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.25</td>
      <td>(0, 1, 3)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.25</td>
      <td>(0, 2, 3)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.25</td>
      <td>(0, 3, 4)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.25</td>
      <td>(1, 2, 3)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.25</td>
      <td>(0, 1, 2, 3)</td>
    </tr>
  </tbody>
</table>
</div>




```python
apriori(df,min_support=0.1,use_colnames=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.75</td>
      <td>(Bread)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>(Butter)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.75</td>
      <td>(Jam)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.50</td>
      <td>(Milk)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.25</td>
      <td>(Wheat Bread)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.50</td>
      <td>(Butter, Bread)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.50</td>
      <td>(Jam, Bread)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.50</td>
      <td>(Milk, Bread)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.25</td>
      <td>(Bread, Wheat Bread)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.75</td>
      <td>(Butter, Jam)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.25</td>
      <td>(Butter, Milk)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.25</td>
      <td>(Jam, Milk)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.25</td>
      <td>(Milk, Wheat Bread)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.50</td>
      <td>(Butter, Bread, Jam)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.25</td>
      <td>(Butter, Milk, Bread)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.25</td>
      <td>(Jam, Milk, Bread)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.25</td>
      <td>(Milk, Bread, Wheat Bread)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.25</td>
      <td>(Butter, Milk, Jam)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.25</td>
      <td>(Butter, Milk, Bread, Jam)</td>
    </tr>
  </tbody>
</table>
</div>




```python
frequent_itemsets = apriori(df,min_support=0.1,use_colnames=True)
```


```python
frequent_itemsets["length"]=frequent_itemsets["itemsets"].apply(lambda x:len(x)) 
```


```python
frequent_itemsets
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.75</td>
      <td>(Bread)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>(Butter)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.75</td>
      <td>(Jam)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.50</td>
      <td>(Milk)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.25</td>
      <td>(Wheat Bread)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.50</td>
      <td>(Butter, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.50</td>
      <td>(Jam, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.50</td>
      <td>(Milk, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.25</td>
      <td>(Bread, Wheat Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.75</td>
      <td>(Butter, Jam)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.25</td>
      <td>(Butter, Milk)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.25</td>
      <td>(Jam, Milk)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.25</td>
      <td>(Milk, Wheat Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.50</td>
      <td>(Butter, Bread, Jam)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.25</td>
      <td>(Butter, Milk, Bread)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.25</td>
      <td>(Jam, Milk, Bread)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.25</td>
      <td>(Milk, Bread, Wheat Bread)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.25</td>
      <td>(Butter, Milk, Jam)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.25</td>
      <td>(Butter, Milk, Bread, Jam)</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
frequent_itemsets[(frequent_itemsets['length']==3)&(frequent_itemsets["support"]>=0.5)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>0.5</td>
      <td>(Butter, Bread, Jam)</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
frequent_itemsets[(frequent_itemsets['length']==2)&(frequent_itemsets["support"]>=0.5)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.50</td>
      <td>(Butter, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.50</td>
      <td>(Jam, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.50</td>
      <td>(Milk, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.75</td>
      <td>(Butter, Jam)</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
frequent_itemsets[(frequent_itemsets["support"]>=0.5)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.75</td>
      <td>(Bread)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>(Butter)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.75</td>
      <td>(Jam)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.50</td>
      <td>(Milk)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.50</td>
      <td>(Butter, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.50</td>
      <td>(Jam, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.50</td>
      <td>(Milk, Bread)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.75</td>
      <td>(Butter, Jam)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.50</td>
      <td>(Butter, Bread, Jam)</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
frequent_itemsets[(frequent_itemsets["support"]>0.5)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.75</td>
      <td>(Bread)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>(Butter)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.75</td>
      <td>(Jam)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.75</td>
      <td>(Butter, Jam)</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
