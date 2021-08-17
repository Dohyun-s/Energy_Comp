#environmental setting
!pip install pulp
from pulp import *
import numpy as np

def optimization(season, electricity_data, PV_data) :
  """
  Input
  season : select the season among 'spring', 'summer', 'fall', or 'winter'
  electricity data : list(24)
  PV_data : list(24)
  ---------------------------------------------------------------------------
  Output
  value(prob.objective) : float; total cost
  output : list(24); 
  """
  #time
  idx=[i for i in range(0,24)]

  #price
  def step(season):
    cost = np.zeros(shape=(24,))
    if season=='summer':
      cost[23]=cost[0]=cost[1]=cost[2]=cost[3]=cost[4]=cost[5]=cost[6]=cost[7]=cost[8]=40.3
      #경부하 23,0,1,2,3,4,5,6,7,8
      cost[9]=cost[12]=cost[17]=cost[18]=cost[19]=cost[20]=cost[21]=cost[22]=85.0
      #중간부하 9,12,17,18,19,20,21,22
      cost[10]=cost[11]=cost[13]=cost[14]=cost[15]=cost[16]=150.9
      #최대부하 10,11,13,14,15,16
    elif season=='winter':
      cost[23]=cost[0]=cost[1]=cost[2]=cost[3]=cost[4]=cost[5]=cost[6]=cost[7]=cost[8]=44.3
      #경부하 23,0,1,2,3,4,5,6,7,8
      cost[9]=cost[12]=cost[13]=cost[14]=cost[15]=cost[16]=cost[20]=cost[21]=83.5
      #중간부하 9,12,13,14,15,16,20,21
      cost[10]=cost[11]=cost[17]=cost[18]=cost[19]=cost[22]=122.2
      #최대부하 10,11,17,18,19,22
    elif season=='spring' or 'fall':
      cost[23]=cost[0]=cost[1]=cost[2]=cost[3]=cost[4]=cost[5]=cost[6]=cost[7]=cost[8]=40.3
      #경부하 23,0,1,2,3,4,5,6,7,8
      cost[9]=cost[12]=cost[17]=cost[18]=cost[19]=cost[20]=cost[22]=cost[21]=54.7
      #중간부하 9,12,17,18,19,20,21,22
      cost[10]=cost[11]=cost[13]=cost[14]=cost[15]=cost[16]=75.2
      #최대부하 10,11,13,14,15,16
    else:
      print('error')
    return cost

  prob=LpProblem("lpproblem", LpMinimize)
  opt_t=LpVariable.dicts("Opt", idx , lowBound=0, upBound=250, cat='Continous' )
  
  prob += lpSum( 6980+step(season)[i]*(b_t[i]-opt_t[i]) if b_t[i]-opt_t[i]>=0 else 0 for i in idx), "Sum_of_Costs"
  #optimization function
  prob+= lpSum(a_t[i]*93/100 -opt_t[i]*100/93 for i in idx)==0

  for i in idx:
    prob+= (125-opt_t[i])<=125
    prob+= (125-opt_t[i])>=-125


  for t in range(1,24):  
      prob+= (lpSum(a_t[i]*93/100 -opt_t[i]*100/93 for i in range(0,t)))<=125 
      prob+= (lpSum(a_t[i]*93/100 -opt_t[i]*100/93 for i in range(0,t)))>=-125
  
  #solve optimization
  prob.solve()
  print( "Total Cost = ", value(prob.objective))

  #output
  output = [0] * 24
  for v in prob.variables():
    print( v.name, "=", v.varValue )
    output[int(v.name[4:])] = v.varValue

  return value(prob.objective), output  
