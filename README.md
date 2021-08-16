# Energy_Comp

# Team: EnerGIST

Dohyun Kim & Jihoon Jung

프로젝트 진행상황:

1. LOAD data를 WindowsGenerator를 이용하여, Model을 만들고 학습시키려함. 
2. 향후 Load data에 기상청 데이터 ex) weather data를 붙여 CNN_LSTM 모델을 적용해보려 함. (normalization을 적용함, )
3. Electricity data를 일별로 총합을 구해 total_sum.csv를 만듬
4. 특정 구간에서 급격히 떨어지는 부분을 빼는 등 데이터 전처리 과정이 필요함

5. PV의 경우. 제한조건 및 objective function 등 필요한 수식을 만듬.
6. PULP library를 공부 중. 이를 적용 예정.
7. Spark를 이용해 PV Data과 load data를 input으로 전기요금을 계산해볼 예정 (?)


2021.08.15 test
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

## Model : LSTnet
그림첨부
- window = 7 (day) 
- label(shift) = 1 (day)
![initial](https://www.researchgate.net/publication/350511416/figure/fig1/AS:1007244028174337@1617157098300/LSTM-sliding-window-prediction-principle.png)
![initial](https://user-images.githubusercontent.com/48517782/129564609-c47ab851-235b-491d-a781-6c64ab6280c7.png)
1. Reference [2]에 있는 Lstnet을 참조함, model에 있는 skipGRU, CNN-GRU, attention을 이용한 postskipGRU, 그리고 autoregression을 합친 모델을 만듬
2. input data를 24시간 받아왔을 때, 다음 25시를 예측하는 모델을 만듬, 그리고 windows를 활용해 다음날 데이터를 예측하는 모델을 만듬
3. loss는 MSE, optimizer는 Adam, learning rate=0.001 정도 설정
4. Ensemble을 이용하여 temperature데이터, 한국전력공사의 데이터, 그리고 지스트 전력 데이터를 각각 모델을 만든 후, concat시켜 train시키는 방향으로 만듬
5. keras tuner를 이용하여 hyperparameter tuning을 진행함, 확률적으로 찾아주는 BayesianOptimizer 이용,
## Running model


### set-up
To get started with this model, 


# 2. 태양광 발전량 예측


# 3. 캠퍼스 BESS스 케쥴링 알고리즘


# Reference
- Tensorflow2 | [page](https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ko)
- Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks | [paper](https://arxiv.org/abs/1703.07015)
