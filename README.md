# Energy_Comp

# Team: EnerGIST

Dohyun Kim & Jihun Jeung

프로젝트 진행상황:

1. LOAD data를 WindowsGenerator를 이용하여, Model을 만들고 학습시키려함. 
2. 향후 Load data에 기상청 데이터 ex) weather data를 붙여 CNN_LSTM 모델을 적용해보려 함. (normalization을 적용함, )
3. Electricity data를 일별로 총합을 구해 total_sum.csv를 만듬
4. 특정 구간에서 급격히 떨어지는 부분을 빼는 등 데이터 전처리 과정이 필요함

5. PV의 경우. 제한조건 및 objective function 등 필요한 수식을 만듬.
6. PULP library를 공부 중. 이를 적용 예정.
7. Spark를 이용해 PV Data과 load data를 input으로 전기요금을 계산해볼 예정 (?)

# 1. 부하량 예측
## Input data and data source
- electricity : GIST Load data
- weather : 기상자료개방포탈 | [링크](https://data.kma.go.kr/cmmn/main.do)
- korea_electricity : 한국전력거래소_시간별 전력수요량(시간단위 전국 발전단 수요 데이터) | [링크](https://www.data.go.kr/data/15065266/fileData.do)

example data

timestamp | electricity | korea_electricity | temperature | ...
---- | ---- | ---- | ----| ---
| | | |
| | | |


## prprocessing
- timestamp : yyyy-mm-dd hh:mm 형식을 timestamp로 바꾸어줌.  
```
timestamp = [0] * length #initialization
for i in range(length):
  timestamp[i] = time.mktime(datetime.datetime.strptime(weather_df['timestamp'][i], '%Y-%m-%d %H:%M').timetuple())
weather_df['timestamp'] = timestamp
```
- low pass filter
```
#figure size
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

```



![initial](https://user-images.githubusercontent.com/48517782/129563138-da10c8de-fbe1-4f1c-8f03-619c550a1d46.png)
Figure. low pass filter
![initial](https://user-images.githubusercontent.com/48517782/129563078-0ad0ce38-fef3-47cf-908c-47793ddec420.png)
Figure. low pass filter (zoom-in)
- trim outlier (electricity < 2000)
```
#remove outlier data (the sequence of removal is important to keep the consistency of index)
df_data2 = df_data
df_data2 = df_data2.drop(list(range(3739,3742)), axis=0)
df_data2 = df_data2.drop(list(range(3595,3598)), axis=0)
df_data2 = df_data2.drop(list(range(2472,2499)), axis=0)
df_data2 = df_data2.drop(list(range(2288,2291)), axis=0)
df_data2 = df_data2.drop(list(range(1813,1820)), axis=0)
```
![data_cleaning](https://user-images.githubusercontent.com/48517782/129665352-436806b9-359e-44e2-aa25-962ffe0ab472.png)

- feature significance
```
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
# split into input and output
X = df_data.iloc[:, 1:] #timestamp
y = df_data.iloc[:, 0]
# fit random forest model
model = RandomForestRegressor(n_estimators=500, random_state=1)
model.fit(X, y)
# show importance scores
names = df_data.columns.values[1:]
for name, score in zip(names, model.feature_importances_):
  print(name, score)
print('------------------------------------------------------')
```

```
electricity 5.1648249770227e-07
year 0.1641875308457792
month 0.829884928244495
date 0.005837147814393805
day 5.034596613288041e-05
time 2.350464802388303e-06
temperature 7.149524364290612e-07
강수량(mm) 2.7390410543482482e-08
풍속(m/s) 2.794462996947997e-07
풍향(16방위) 1.4122559125107157e-07
습도(%) 5.663451765984923e-07
증기압(hPa) 3.6992711347979863e-06
이슬점온도(°C) 3.963274619044288e-06
현지기압(hPa) 1.433693757345914e-05
해면기압(hPa) 1.1197462280482841e-05
전운량(10분위) 8.481221874674216e-07
시정(10m) 1.0409165837576703e-06
지면온도(°C) 3.648376055851243e-07
```

![initial](https://user-images.githubusercontent.com/48517782/129665073-c3756e12-1067-4916-91f3-35742629bf39.png)


## Model : LSTnet
1. LSTNET은 skipGRU, CNN-GRU, attention을 이용한 postskipGRU, 그리고 autoregression을 합쳐진 모델임. (Reference[2] 참고)   
![initial](https://user-images.githubusercontent.com/48517782/129564609-c47ab851-235b-491d-a781-6c64ab6280c7.png)   
  - `loss function= MSE`, `optimizer=Adam`, `learning rate=0.001`으로 설정   
2. 7 day * 24 day/hour의 timeseries 데이터(`Input_window = 7`)로 다음날 (`sliding_width = 1`) 24 hour를 예측(`label_width = 1`)하는 모델을 만듬.   
![initial](https://www.researchgate.net/publication/350511416/figure/fig1/AS:1007244028174337@1617157098300/LSTM-sliding-window-prediction-principle.png)
3. LSTNET으로 GIST 전력 사용량, 대한민국 전력 사용량 그리고 기온의 각각 모델을 만들고, 각각 예측모델의 output을 concat시켜 GIST 전력사용량을 예측하는 ensemble 모델로 train함.   
![initial](https://static.commonlounge.com/fp/600w/E4CzBIUXn3fadwbqDNNpMghCK1520496433_kc)   
4. `keras tuner`는 TensorFlow 프로그램에 대한 최적의 하이퍼파라미터 세트를 선택하는 데 도움을 주는 라이브러리이다. `BayesianOptimizer`은 gaussian process으로 적합한 hyperparameter tuning한다. `max_trials=80`으로, `objective='val_loss'`으로 설정하여 loss value가 가장 낮은 hyperparameter를 선정하였다.

## Running model

# 2. 태양광 발전량 예측


# 3. 캠퍼스 BESS 스케쥴링 알고리즘
Library `pulp`로 cost를 최소화해주는 BESS schedule을 찾아주는 linear optimization을 구현하였음. `optimization(season, electricity_data, PV_data)`으로 사용가능하다. argument는 24시간 동안 사용한 electricity 값(e.g. electricity_t)과 PV 값(e.g. PV_t), 그리고 계절(e.g. 'summer')이다. 함수의 출력값은 `total_cost`와 시간별 `bess_value`이다.
```
>>> PV_t=[0,0,0,0,0,0,0,0.6,11.8,58.5,125.2,196.1,235.7,308.1,68.8,48,25,1.5,0,0,0,0,0,0] 
>>> electricity_t=[4972.86446897, 4827.32813707, 4714.39951693, 4640.4637654 ,
       4619.08798299, 4696.02395711, 4934.15191917, 5358.15019654,
       5870.89139808, 6269.93405485, 6444.9726441 , 6461.49278899,
       6454.26552802, 6488.01214138, 6506.49025023, 6471.5219868 ,
       6409.99699496, 6273.83475456, 6020.27016168, 5780.28216133,
       5637.47629486, 5505.49176825, 5340.39785949, 5166.06439475 ] #define electricity
>>> total_cost, bess_value = optimization('summer', electricity_t, PV_t)
>>> print(total_cost)
11904816.96244172
>>> print(bess_value)
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 193.2964, 250.0, 0.0, 250.0, 59.50512, 180.68505, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```
# Reference
- Tensorflow2 | [page](https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ko)
- Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks | [paper](https://arxiv.org/abs/1703.07015)
