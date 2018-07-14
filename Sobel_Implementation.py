# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:21:01 2018

@author: Falguni Das Shuvo
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import floor


sobel_v = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel_h = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

a_str = input("Enter file path with file name \n")


a = cv.imread(a_str, 0)

row,col = a.shape

#zero padding
z_a = np.zeros(shape=(row+2,col+2))
z_a[1:row+1, 1:col+1] = a

print(z_a)

#result size = floor(((n+2p-f)/s) + 1) <-- individually for row & column
#here, n = 5, p=1, s=1, f=3
# result size = floor(((5+2-3)/1)+1) = 5
n_r = a.shape[0]
n_c = a.shape[1]

p = 1

f_r = 3
f_c = 3

s = 1

res_row = floor(((n_r + 2 * p - f_r)/float(s)) + 1) 
res_col = floor(((n_c + 2 * p - f_c)/float(s)) + 1)
result = result_h = result_v = np.zeros(shape=(res_row,res_col), dtype=np.int32)

#print(res_row, res_col)

for r in range(res_row):
    for c in range(res_col):
        result_h[r,c] = (z_a[r:r+3,c:c+3] * sobel_h).sum()
        result_v[r,c] = (z_a[r:r+3,c:c+3] * sobel_v).sum()

result = np.sqrt(np.power(result_h,2) + np.power(result_v,2))

result = np.clip(result,0,255)
result = result.astype(a.dtype)

print('result_h', result_h.dtype)
print('result_v', result_v.dtype)
print('result', result.dtype)

print(result)

plt.imshow(result)

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',result)
cv.waitKey(0)
cv.destroyAllWindows()


dot_index = a_str.find('.')

if not dot_index == -1:
    name_f = a_str[:dot_index]
    name_l = a_str[dot_index:]
    
    cv.imwrite(name_f + '_sobel' + name_l, result)