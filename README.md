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
