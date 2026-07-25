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

    
# ans = []
# path = []
# used = [False, False, False]    
# nums=[1,2,3]
# def backtrack(index):
#     if len(path)==len(nums):
#         ans.append(path[:])
#         return
#     for i in range(len(nums)):
#         if not used[i]:
#             used[i]=True
#             path.append(nums[i])
#             backtrack(index+1) 
#             path.pop()
#             used[i]=Fals
# backtrack(0)    
# print(ans)

# 1. Backtracking is about making choices

# Before today, you were trying to think:

# "How do I generate all answers?"

# Now you think:

# "What choices do I have from the current state?"

# For this problem, the choices are:

# (
# )
# 2. Not every choice is valid

# This is the first problem where you had to prune.

# Before exploring a branch, ask:

# Can I place "(" ?

# Can I place ")" ?

# If not, don't recurse.

# This is a huge backtracking concept.

# 3. State matters

# You discovered that recursion needs information about the current state.

# Here the state is:

# path
# open
# close

# Without this information, the recursive call cannot continue correctly.

# 4. Choose → Explore → Undo

# You finally wrote the classic pattern:

# path.append(...)

# backtrack(...)

# path.pop()

# This is the heart of backtracking.

# 5. The template became real

# Earlier you asked about this template:

# Base case

# Choices

# Check validity

# Choose

# Recurse

# Undo

# Today you actually filled it.

# Template	Generate Parentheses
# Base case	len(path)==2*n
# Choices	( and )
# Invalid	open==n or close>=open
# Choose	path.append()
# Recurse	backtrack(...)
# Undo	path.pop()

# This is probably the biggest thing you learned.

# 6. Not every backtracking problem needs a loop

# This is something new.

# Earlier:

# Permutations:

# for choice in choices:

# Today:

# if can_place_open:
#     ...

# if can_place_close:
#     ...

# There are still two choices, but they don't have to come from a loop.

# 7. Different problems need different state

# Compare everything you've learned so far:

# Subsets

# State:

# index
# path
# Permutations

# State:

# path
# used
# Generate Parentheses

# State:

# path
# open
# close

# This is a really important realization:

# Every backtracking problem asks, "What information do I need to describe my current state?"

# Once you know the state, the recursion becomes much easier to write.
# ans = []
# path = []
# open=0
# close=0
# def backtrack(n,open,close):
#     if len(path)==n*2:
#         ans.append(path[:])
#         return
   
#     for i in ["(",")"]:

#             if i=="(" and open<n:
#                 path.append("(")
#                 backtrack(n,open+1,close)
#                 path.pop()
         
#             if i==")":
#                 if close<n:
#                     if close<open:
#                             path.append(")") 
#                             backtrack(n,open,close+1)
#                             path.pop()
# backtrack(2,0,0)      
# print(ans)                        

                                    

        
        
# ans = []
# path = []

# def backtrack(tar, nums, summ, start):
#     if summ == tar:
#         ans.append(path[:])
#         return

#     for i in range(start, len(nums)):
#         if summ + nums[i] <= tar:

#             path.append(nums[i])

#             backtrack(tar, nums, summ + nums[i], i)

#             path.pop()

# backtrack(7, [2,3,6,7], 0, 0)
# print(ans) 
# 1. State is everything ⭐⭐⭐⭐⭐

# Before coding, ask:

# What information does the next recursive call need?

# For this problem, the state is:

# start
# remaining target (or sum)
# path

# Every backtracking problem has a different state.

# 2. The start parameter

# This is the biggest lesson.

# start means:

# "The next recursive call is only allowed to choose from this index onward."

# It serves two purposes:

# ✅ Allows reusing the current element.
# ✅ Prevents duplicate combinations like:
# 2 3
# 3 2
# 3. Think in terms of choices

# Every recursive call asks:

# "What can I choose next?"

# Not:

# "What is the whole answer?"

# That mindset shift is huge.

# 4. Choose → Recurse → Undo

# Every backtracking problem follows:

# Choose

# Recurse

# Undo

# You finally got comfortable with this pattern.

# 5. Don't plan ahead

# This was your biggest confusion.

# You kept asking:

# "How many times should I choose 2?"

# Now you know:

# The algorithm never knows.

# It simply keeps exploring until it succeeds or fails.

# 6. Parent and child calls have different state

# This is subtle but important.

# Parent:

# path = [2]

# Child:

# path = [2,2]

# Each recursive call has its own view of the problem.

# 7. Parameters describe the current world

# This is a mental model I want you to keep forever.

# Instead of thinking:

# "Parameters are inputs."

# Think:

# "Parameters describe the world that this recursive call lives in."

# For example:

# backtrack(start=1, target=4)

# means:

# "I'm in a world where I have 4 left to make, and I'm only allowed to choose from index 1 onward."

# That way of reading recursive calls makes them much easier to understand.

# 8. You don't modify everything

# Instead of doing:

# summ += nums[i]
# ...
# summ -= nums[i]

# you learned it's often cleaner to pass the updated value directly:

# backtrack(..., summ + nums[i], ...)

# This keeps the parent's state untouched.

# 9. The same template, different state

# Look at your journey:

# Problem	State
# Subsets	index, path
# Permutations	used, path
# Generate Parentheses	open, close, path
# Combination Sum	start, remaining target, path

# Notice something?

# The template never changed.

# Only the state changed.

# ⭐ The biggest lesson of all

# If there's one thing I hope you take away, it's this:

# Backtracking isn't about writing recursion. It's about identifying the right state.         

# s = "aab"
# path=[]
# ans=[]
# end=0
# def ispalin(s):
#     return s==s[::-1]
        
# def backtrack(start):
#     if start == len(s):
#         ans.append(path[:])
#         return
#     for end in range(start,len(s)):
#         piece=s[start:end+1]
#         if ispalin(piece):
#             path.append(piece)
#             backtrack(end+1)
#             path.pop()

 
# backtrack(0)
# print(ans  )   


 
# path=[]
# ans=[]
# s="25525511135"
# def backtrack(start):
#     if start == len(s):
#         if len(path) == 4:
#             ans.append(path[:])
#             return
#     for end in range(start, min(start+3,len(s))):
#         peace = s[start:end + 1]
#         if int(peace)<=255:
#             continue
        
            
#         path.append(peace)
#         backtrack(end+3)
#         path.pop()
# backtrack(0)
# print(ans)            
            
        
      


# path = []
# ans = []

# def backtrack(start, s, k):
#     if start == len(s):
#         return

#     first = s[start]

#     for i in range(len(k)):
#         path.append(first)
#         path.append(k[i])

#         ans.append("".join(path))

#         path.pop()
#         path.pop()

#     backtrack(start + 1, s, k)

# backtrack(0, "abc", "def")
# print(ans)
      
      
      
ans=[]
path=[]
k=0
def backtrack(start,target,nums,summ):
    global k
    nums.sort()
    if summ==target:
        ans.append(path[:])
        return
    for i in range(start,len(nums)):
        if summ+nums[i]>target:
            continue
        if nums[i]!=k:
            path.append(nums[i])
            backtrack(i+1,target,nums,summ+nums[i])
            k=path.pop()
backtrack(0,4,[1,1,2,2],0)            
print(ans)
