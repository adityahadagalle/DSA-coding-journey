# nums = [1,5,4,2,9,9,9]
# k = 3

# left = 0
# freq = {}
# summ = 0
# best = 0

# for right in range(len(nums)):
#     freq[nums[right]] = freq.get(nums[right], 0) + 1
#     summ += nums[right]

#     if right - left + 1 == k:

#         if len(freq) == k:
#             best = max(best, summ)

#         summ -= nums[left]

#         freq[nums[left]] -= 1

#         if freq[nums[left]] == 0:
#             del freq[nums[left]]

#         left += 1

# print(best)




# nums = [1,1,1,0,0,0,1,1,1,1,0]
# k = 2
# zeros=0
# best=0
# left=0
# for right in range(len(nums)):
#     if nums[right]==0:
#         zeros+=1
#     while zeros>k:
#         if nums[left]==1:
#             left+=1
#         else:
#             zeros-=1
#             left+=1
#     best=max(best,right-left+1)    
# print(best)            
            
            
            
            
# s = "abcdabcdebb"
# left=0
# freq={}
# best=0
# for right in range(len(s)):
#     freq[s[right]] = freq.get(s[right], 0) + 1
#     while freq[s[right]]>1:
#         freq[s[left]]-=1
#         if freq[s[left]] == 0:
#             del freq[s[left]]
#         left+=1    
            
#     best=max(best,right-left+1)
        
# print(best)                




# nums = [23,2,4,6,7]
# k = 13
# freq={0:-1} 
# pre=0
# for i in range(len(nums)):
#     pre+=nums[i]
#     rem=pre%k
#     if rem in freq:
#         if i-freq[rem]>=2:
#             print(True)
#             break
#     else:
#         freq[rem] = i
        
        
    
    
    
    
######################recursion#################################################recursion#################################################recursion#################################################recursion###########################

# def fun(count):
#     if count==3:
#         return
#     print("Adii")
#     count+=1
#     fun(count)
# fun(count=0)  



# def fun(i,n):
#     if i==6:
#         return
#     print(i)
#     fun(i+1,n)
# fun(i=1,n=6)  


# def fun(i,n):
#     if i<n:
#         return
#     print(i)
#     fun(i-1,n)
# fun(10,1)      
    
    
# def fun(n):
#     if n==1:
#         return 1
#     return n*fun(n-1)    

# print((fun(5)))


# def fun(n):
#     if n==1:
#         return 1
#     return n+fun(n-1)
# print(fun(10))
    
# i=0
# j=0  
# def fun(i,j,arr):

#     if i>=j:
#         return True
#     if arr[i]==arr[j]:
#         i+=1
#         j-=1
#     else:
#         return False
#     return fun(i,j,arr)
# print(fun(i=0,j=4,arr="madam") )   


# def fun(x,n):
#     if n==1:
#         return x
#     return x*fun(x,n-1)
# print(fun(2,5))




# def fun(n):
#     if n==0:
#         return 1
#     if n==1:
#         return 1
#     return fun(n-1)+fun(n-2)
# print(fun(5))

# def fun(n,m):
#     if n==1 or m==1:
#         return 1
#     return fun(n,m-1)+fun(n-1,m)
# print(fun(3,3))


# def fun(i,arr):
#     if i >= len(arr):
#         return 0
#     rob=arr[i]+fun(i+2,arr)
#     skip=fun(i+1,arr)
#     return max(rob,skip)
# print(fun(0,[2,7,9,3,1]))



# def fun(i,arr):
#     if i>=len(arr):
#         return 0
#     rob=arr[i]+fun(i+2,arr)
# #     skip=arr[i+1]
# #     return max(rob,skip)
# # print(fun(i=0,arr=[3,2,7,10]))

# ###########################backtracking#####################################################################
# "How many choices do I have at each recursive step?"

# If the answer is:

# 2 choices → Think Take/Skip.
# Many choices → Think Loop.



# def backtrack(index):

#     if index == len(nums):
#         ans.append(path[:])
#         return

#     # Take
#     path.append(nums[index])
#     backtrack(index+1)

#     # Undo
#     path.pop()

#     # Skip
#     backtrack(index+1)


# nums = [1,2,3]

# ans = []
# path = []

# backtrack(0)

# print(ans)         


# def backtrack(index):

#     if index == len(nums):  
#         ans.append(path[:])
#         return

#     # Take
#     if path[-1]!=path[:]:
#         path.append(nums[index])
#     backtrack(index+1)

#     # Undo
#     path.pop()

#     # Skip
#     backtrack(index+1)


# nums = [1,2,2]

# ans = []
# path = []

# backtrack(0)

# print(ans)  







# nums=[1,2,3]
# def backtrack(index):

#     if index == len(nums):  
#         ans.append(path[:])
#         return
#     for index in range(len(nums)):
    
#         path.append(nums[index])
#     backtrack(index+1)
    
#     path.pop()
#     backtrack(index+1)
# # nums = [1,2,2]

# ans = []
# path = []

# backtrack(0)

# print(ans)  

    
ans = []
path = []
used = [False, False, False]    
nums=[1,2,3]
def backtrack(index):
    if len(path)==len(nums):
        ans.append(path[:])
        return
    for i in range(len(nums)):
        if not used[i]:
            used[i]=True
            path.append(nums[i])
            backtrack(index+1) 
            path.pop()
            used[i]=False
backtrack(0)             
print(ans)
