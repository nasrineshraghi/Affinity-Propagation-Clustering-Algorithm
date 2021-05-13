# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:32:17 2021

@author: neshragh


Make R code -- evaluation metrics\

"""


from sklearn import metrics
from sklearn.metrics import davies_bouldin_score





my_label= [3,3,4 , 4,  4,  4,  4 , 3 , 4 , 6 , 5 , 6 , 2 , 2 , 3 , 5 , 4  ,3  ,3 , 6 ,
  4,  3 , 5 , 5 , 6 , 5 , 4 , 1 , 4 , 4 ]

my_centers =[
[ 1,     1],
 [1,     1],
  [       2 ,    1],
   [      2  ,   1],
    [     2   ,  1],
     [    2    , 1],
      [   2     ,1],
       [  1  ,   1],
        [ 2   ,  1],
[        6     ,1],
 [       4     ,1],
  [      6     ,1],
   [     5,     1],
    [    5 ,    1],
     [  1   ,  1],
      [  4   ,  1],
       [ 2    , 1],
        [1     ,1],
[        1,     1],
 [      6  ,   1],
  [      2  ,   1],
   [     1   ,  1],
    [    4    , 1],
     [   4     ,1],
[        6,     1],
 [       4 ,    1],
  [      3  ,   1],
   [     2   ,  2],
    [    2    , 1],
     [   3     ,1]
     ]
        
        
        


#During
##
#my_label=[ 6,1,4,  1,  2,  2,  4,  1 , 5,  2,  4 , 5 , 5 , 2 , 6 , 1,  2,  6 ,
#          4 , 1 , 3 , 6 , 1 , 2 , 2 , 4 , 1 , 6 , 6 , 1 ]
##
#
##
#my_centers =[
#[         1  ,   1],
# [        2   ,  1],
#  [       6    , 1],
#   [      2     ,1],
#    [     4,     2],
#     [    4 ,    2],
#        [ 5  ,   1],
#         [3   ,  1],
#[         2    , 2],
# [       4,     1],
#  [      6 ,    1],
#   [     2  ,   2],
#    [    2   ,  2],
#     [   4    , 1],
#      [  1     ,1],
#       [ 2,     1],
#        [4 ,    1],
#[        1  ,   1],
# [       6   ,  1],
#  [      2    , 1],
#   [     1     ,2],
#    [    1,     1],
#      [  2 ,    1],
#       [ 4  ,   1],
#        [4   ,  1],
#[        6    , 2],
# [       2     ,1],
#  [      1 ,    1],
#   [     1  ,   1],
#    [    2   ,  1]
#]



#my_label=[4,6, 2,  4,  4,  2,  4 , 3 , 2 , 3 , 3 , 3 , 1 , 4 , 1 , 3  ,3  ,1  ,5 , 1,
#          5 , 6 , 3 , 3 , 3 , 2,  4 , 3 , 3 , 1]
#
#my_centers =[
#[         5,     1],
# [        3 ,    1],
#  [       4  ,   2],
#   [      5   ,  1],
#    [     5    , 1],
#     [    5     ,3],
#      [   5,     1],
#       [  4 ,    1],
#        [ 4  ,   2],
#[        4    , 1],
# [       4     ,1],
#  [      4,     1],
#   [     6 ,    1],
#    [    5  ,   1],
#     [   6   ,  1],
#      [  4    , 1],
#       [ 4     ,1],
#        [6,     1],
# [       6 ,    2],
#[        6  ,   1],
# [       6   ,  2],
#[        3    , 1],
#[        4     ,1],
# [       4,     1],
#  [      4 ,    1],
#   [     5  ,   2],
#    [    5   ,  1],
#     [   4    , 1],
#      [  4     ,1],
#       [ 6     ,1]]
#
#
#


print('Calinski-Harabasz Index Macro:',metrics.calinski_harabasz_score(my_centers, my_label))  
print('Silhouette Coefficient Macro:',metrics.silhouette_score(
        my_centers, my_label, metric='euclidean'))
print('Davies-Bouldin Index Macro:',davies_bouldin_score(my_centers, my_label))

