import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from cvxopt import blas, solvers
import cvxopt as opt
import random
import math
class stockreader():
  def __init__(self):
    self.__df=pd.DataFrame()
    
  def setStockDic(self, T, *args):
    index = pd.date_range('2010-1-1', periods=T, freq='D')
    for stock in args:
      self.__df[stock] = web.DataReader(stock, 'yahoo', index)['Adj Close']
  def getStockDic(self):
    return self.__df

  def optimal_portfolio(self,returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    return risks , returns

  def resampling(self):
    randNo = random.randint(100,500)
    e = stockreader()
    e.setStockDic(randNo, "GOOGL","BRK-B")
    df = e.getStockDic()
    print df
    r = df.pct_change(1).dropna()
    meanDailyReturn = r.mean().dropna()
    covMatrix = r.cov()
    z = e.optimal_portfolio(r.as_matrix().T)
    return z
    
if __name__ == '__main__':
  #p=0
  e = stockreader()
  e.setStockDic(500, "GOOGL","BRK-B")
  df = e.getStockDic()
  r = df.pct_change(1).dropna()
  risks, returns = e.optimal_portfolio(r.as_matrix().T)
  m1 = np.polyfit(returns, risks, 2)
  y = 0.015
  plt.plot(risks, returns, 'y-o')
  plt.show()
  delta = ((y - m1[2])/m1[0] + (m1[1]/(2*m1[0]))**2 )
  if delta<0: 
    print "no real root"
  else:
    x = math.sqrt(delta) - m1[1]/(2*m1[0])
    print "return= %s, risk = %s" %(x, y)

  #for i in range(0,5):
  #  p += np.poly1d(e.resampling())
  #  z = p(0.01)/5.
  #print z 
  


  
  
