class Solution:
    
#     def searchleftright(self, n: int):
#         res = []

#         def backtrack(curr,left_count,right_count):
#             # base case
#             if len(curr) == 2 * n:
#                 res.append(curr)
#                 return    
    
#             if left_count<2:
#                 backtrack(curr+1,left_count+1,right_count)
#             if right_count<2:
#                 backtrack(curr+1,left_count,right_count+1)
            

#         backtrack("",0,0)  
#         return res