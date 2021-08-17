# -*- coding: utf-8 -*-
"""PV.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q9CmwRrftAWDLDG6TLISDTyI4i1de6VL
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import numpy as np
# %matplotlib inline  

#timestamp
from datetime import datetime
import time

#import libaray
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive/')

import os
import os.path
file_list = os.listdir("/content/drive/MyDrive/PV")

date_list = []
for filename in file_list:  
  split = filename.split('.', 1) #split by '.'; one time  
  date = split[1].split('_')[1]
  date_list.append(date)

print(len(date_list))
print(date_list)

df=pd.read_excel("/content/drive/MyDrive/PV/"+file_list[0], header=2)

exam=pd.read_csv('/content/drive/MyDrive/energy/Load_raw_data_20210807.csv')
exam

import datetime

total_sum_1_y = pd.DataFrame([], dtype='float')
for date in file_list:
  #print(date)                                            
  #load data
  ##load peak1
  t1,month, t2,t3,t4=date.split('-')
  s1,year=t1.split('_')
  day, s2=t2.split('_')
  df= pd.read_excel("/content/drive/MyDrive/PV/"+date, header=2)
  df=df.replace('-',0)
  sum1=np.array(df.iloc[3:,6].values,dtype=float)+np.array(df.iloc[3:,8].values,dtype=float)+np.array(df.iloc[3:,10].values,dtype=float)
  sum2=np.array(df.iloc[3:,12].values,dtype=float)+np.array(df.iloc[3:,14].values,dtype=float)+np.array(df.iloc[3:,16].values,dtype=float)
  sum3=np.array(df.iloc[3:,18].values,dtype=float)+np.array(df.iloc[3:,20].values,dtype=float)+np.array(df.iloc[3:,22].values,dtype=float)
  sum=sum1+sum2+sum3
  if (df.shape[0]==28):
    sum=sum[:-1]
  elif (df.shape[0]>28):
    rest=df.shape[0]-28+1
    sum=sum[:-rest]
  else:
    print('error')
    break
  sum=pd.Series(sum)    
  #convert colnames into timestamp and get total energy consumption per hour
  #print(date)
  timestamp = [0] * len(sum.index) #initialization

  #print("timestamp: ", timestamp)

  #day information
  sum = pd.DataFrame(sum, columns=['PV'])
  for j in range(24):
    sum['day'] = year+'-'+month+'-'+day+ ' ' + str(0)
  for i in range(len(sum.index)):
    sum['day'][i] = year+'-'+month+'-'+day+ ' ' + str(i)
    timestamp[i] = time.mktime(datetime.datetime.strptime(year+'-'+month+'-'+day+ ' ' + str(i), '%Y-%m-%d %H').timetuple())
  sum.index = timestamp
  
  #print(sum)

  #save on panda series (For tensorflow, it sholud be converted into np.array)
  total_sum_1_y = pd.concat([total_sum_1_y, sum], axis=0)


print(total_sum_1_y.shape)
print(total_sum_1_y)

year3 = []
for j in range(len(total_sum_1_y)):
  if (total_sum_1_y['day'].values[j].split('-')[0]=='2021'):
    year3= np.append(year3, total_sum_1_y['PV'].values[j])

year2 = []
for j in range(len(total_sum_1_y)):
  if (total_sum_1_y['day'].values[j].split('-')[0]=='2020'):
    year2= np.append(year2, total_sum_1_y['PV'].values[j])

plt.figure(figsize=(20,8))

plt.plot(total_sum['PV'])

total_sum_new=pd.concat([total_sum_1_y.iloc[:6170],total_sum_1_y.iloc[6330:6790],total_sum_1_y.iloc[7294:]])
total_sum_new.to_csv('/content/drive/MyDrive/energy/PV_total_sum_new.csv')

"""##Low pass Filter

"""

total_sum=total_sum_new.drop("day",axis=1)

total_sum

total_sum.columns=['T (degC)']
total_sum
plot_cols = ['T (degC)']
#plot_features = total_sum[plot_cols]
#_ = plot_features.plot(subplots=True)

plot_features = total_sum[plot_cols].iloc[:480]
_ = plt.plot(plot_features)



df=total_sum
fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')

# Commented out IPython magic to ensure Python compatibility.
from scipy import signal
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import numpy as np
# %matplotlib inline  

#timestamp
from datetime import datetime
import time

#import libaray
import tensorflow as tf
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

#figure size
plt.rcParams['figure.figsize'] = [20, 5]
df_data=df
xn = df_data['T (degC)']
t = df_data.index
plt.plot(t, xn)

del t

t=[]
for i in range(len(xn)):
  t.append(float(i))

plt.rcParams['figure.figsize'] = [20, 5]

#filter variable
b, a = signal.butter(1, 0.3)

y = signal.filtfilt(b, a, xn)

plt.figure
plt.subplot(1,1,1)
plt.plot(t, xn, 'b')
plt.plot(t, y, 'r') #lfilter
plt.legend(('noisy signal','filtfilt'), loc='best')
plt.grid(True)
plt.show()

df['T (degC)']=y

df=df.reset_index()
df=df.drop('index', axis=1)