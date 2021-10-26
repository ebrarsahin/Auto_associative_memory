# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:47:33 2021

@author: shneb
"""
import numpy as np
import matplotlib.pyplot as plt
print("\n")
pattern_E=np.array([[1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,
                     1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1]])

pattern_S=np.array([[1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,
                    1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]])

target_E=np.array([[1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,
                     1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1]])


target_S=np.array([[1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,
                    1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1]])
weight_E=(np.dot(pattern_E.T,target_E))
weight_S=(np.dot(pattern_S.T,target_S))
weight=weight_E+weight_S
print("In case TEST_E with 1 mistaken  data")
test_E=np.array([[-1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,
                     1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1]])
print("\n")
print("TEST MATRIX:")
print(test_E)
print("\n")
print("y : test_E x weight")
y=np.dot(test_E,weight)
print("\n")
print(y)
y[y<=0]=-1
y[y>0]=1
print("\n")
print("AFTER THE ACTIVATION FUNCTION:")
print(y)
plt.imshow(y)
print("\n")
comparison= y==pattern_E
equal_arrays=comparison.all()
if equal_arrays==True:
    print("KNOWN PATTERN")
else:
    print("THIS IS NOT THE SAME AS PATTERN E" )    
plt.imshow(y.reshape(10,7))

