from __future__ import division, print_function 

import numpy as np
import scipy.linalg

def lqr2(A,B,Q,R):

#ref Bertsekas, p.151
 
#first, try to solve the ricatti equation
 X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
 
#compute the LQR gain
 K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
 
 eigVals, eigVecs = scipy.linalg.eig(A-B*K)

 return K, X, eigVals
 
def dlqr(A,B,Q,R):

 #ref Bertsekas, p.151
 
 #first, try to solve the ricatti equation
 X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
 
#compute the LQR gain
 K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
 
 eigVals, eigVecs = scipy.linalg.eig(A-B*K)
 
 return K, X, eigVals