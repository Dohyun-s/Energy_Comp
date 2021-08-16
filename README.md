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
## data source
- electricity : GIST Load data
- weather : 기상자료개방포탈 | [링크](https://data.kma.go.kr/cmmn/main.do)
- korea_electricity : 한국전력거래소_시간별 전력수요량(시간단위 전국 발전단 수요 데이터) | [링크](https://www.data.go.kr/data/15065266/fileData.do)

## prprocessing
- timestamp : yyyy-mm-dd hh:mm 형식을 timestamp로 바꾸어줌.  
- low pass filter
![initial](https://user-images.githubusercontent.com/48517782/129563138-da10c8de-fbe1-4f1c-8f03-619c550a1d46.png)
Figure. low pass filter
![initial](https://user-images.githubusercontent.com/48517782/129563078-0ad0ce38-fef3-47cf-908c-47793ddec420.png)
Figure. low pass filter (zoom-in)
- trim outlier
- 

## Background : 그림첨부
- window = 7 (day) 
- shift = 1 (day)

## Running model
### set-up
To get started with this model, 


# 2. 태양광 발전량 예측


# 3. 캠퍼스 BESS스 케쥴링 알고리즘


# Reference
- Tensorflow2 | [page](https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ko)
- Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks | [paper](https://arxiv.org/abs/1703.07015)
