# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:58:06 2024

@author: nachi
"""

diag_Q = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])*1000
diag_R = np.array([1,1,1])
Q = np.diag(diag_Q)
R = np.diag(diag_R)
K, P, eigenvalues = control.dlqr(A_discrete, B_discrete, Q, R)
Ad = A_discrete
Bd = B_discrete

TEST= Ad.T@P@Ad-P-Ad.T@P@Bd@np.linalg.inv((Bd.T@P@Bd+R))@Bd.T@P@Ad+Q

print(TEST)
# print(K)
# np.max(abs(K))