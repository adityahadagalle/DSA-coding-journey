# n=121
# count=0
# ki=[]
# while n>0:
#     k=n%10
    
#     n=n//10
# num=153
# n=num
# c=0
# while num>0:
#     last=num%10
#     c+=last**3
#     num=num//10
# print(c)

# s="nitin"

# l=0
# n=len(s)
# r=n-1
# while l<r:
#     if s[l]!=s[r]:
#         print("not palindrome")
#         break

#     l+=1
#     r-=1
# else:
    
#     print("palindrome")    

# num=[1,22,45]
# for i in range(0,len(num)-1):
#     if num[i]>num[i+1]:
#         print("not")
#         break
# else:   
#     print("true")
# num=[1,1,2,3,4,4,4]
# n=len(num)

# i=0
# j=i+1
# while j<n:
#     if num[i]!=num[j]:
#         i+=1
#     num[i],num[j]=num[j],num[i]
#     j+=1
# print(num)
# num = [1,2,6,5,0]
# n = len(num)

# i = 0

# for j in range(n):
#     if num[j] != 0:
#         num[i], num[j] = num[j], num[i]
#         i += 1

# print(num)




# n = len(num)

# i = 0

# for j in range(n):
#     if num[j]>=5:
#         num[i],num[j]=num[j],num[i]
#         i+=1
# print(num)        
        
# max=0
# smax=0   
# num = [1,2,3,1,0,4]
# for i in range(0,len(num)):
#     if num[i]>max:
#         smax=max
#         max=num[i]
# print(smax)



# num=[1,1,2,2,2,3,3,3]
# i=0
# j=1
# while j<len(num):
#     if num[i]!=num[j]:
#         num[i],num[j]=num[j],num[i]
#         i+=1
#     j+=1    
# print(num) 

# num=[1,2,3,4,3,23,34,32,4,5]
# i=0
# j=len(num)-1
# while i<j:
#     num[i],num[j]=num[j],num[i]
#     i+=1
#     j-=1
# print(num)    
# num=[1,2,3,4,5] 
# n=len(num)
# num[:]=[num[-1]]+num[0:n-1] 



# temp=num[-1]
# i=n-1
# while i>0:
#     num[i] =num[i-1] 
# num.insert(0,temp)
# print(num)


# num=[1,2,3,4,5]
# n=len(num)
# k=6%n
# for _ in range(0,k):
#     m=num.pop() 
#     num.insert(0,m)

#     print(num)    
# print(1%6)  


# num=[16,1,0,9,100]
# k=[]
# i=0
# j=len(num)-1
# while i<=j:
#     if num[j]<num[i]:
#         k.append(num[i])
#         i+=1
#     else:
#         k.append(num[j])  
#         j-=1
   
# print(k)   

# num = [0, 1, 3, 4]
# n = len(num)

# s1 = n * (n + 1) // 2
# s2 = sum(num)

# missing = s1 - s2
# print(missing)

# num=[1,1,0,1,1,1,1,1,1,0,1,1,1,1]
# max=0
# count=0
# for i in range(0,len(num)):
#     if num[i]!=0:
#         count+=1
#     if num[i]==0:
#         count=0    
#     if count>max:
#         max=count
# print(max) 
        
# num=[1,1,2,2,2,3,3,3]
# k=[]
# i=0
# j=1
# k.append(num[i])
# while j<len(num):
#     if num[i]!=num[j]:
#         k.append(num[j])
#     i+=1
#     j+=1
# print(k)        
    

# nums = [-1,  0,  1,  2, -1, -4]
# nums.sort()
# k=[]
# print(nums)  
# t=0
# for i in range(len(nums)-1):

#     left = i+1
#     right = len(nums)-1
#     while left<right:
#         if nums[i]+nums[left]+nums[right]<t:
#             left+=1
#         elif nums[i]+nums[left]+nums[right]>t:
#             right-=1
#         else:
#             k.append(nums[i])
#             k.append(nums[left])
#             k.append(nums[right]) 
#             break       
# print(k) 


# nums = [-1, 0, 1, 2, -1, -4]
# nums.sort()

# k = []
# t = 0
# n = len(nums)

# for i in range(n - 2):

#     left = i + 1
#     right = n - 1

#     while left < right:

#         s = nums[i] + nums[left] + nums[right]

#         if s == t:
#             k.append([nums[i], nums[left], nums[right]])
#             left += 1
#             right -= 1

#         elif s < t:
#             left += 1

#         else:
#             right -= 1

# print(k)
       
# num = [2, 3, 6, 15]
# target = 9
# i=0
# j=len(num)-1
# while i<j:
#     if num[i]+num[j]<target:
#         i+=1
#     elif num[i]+num[j]>target:
#         j-=1
#     else:
#         print(num[i],num[j])     
#         break
                 
 
# num = [-1, 2, 1, -4]
# target = 1

# # Step 1: Sort the array
# num.sort()  
# print("Sorted array:", num)

# # Step 2: Initialize closest sum with first 3 numbers
# closest_sum = num[0] + num[1] + num[2]

# # Step 3: Loop over the array
# for i in range(len(num) - 2):  # n-2 because we need at least two numbers after i
#     l = i + 1
#     r = len(num) - 1

#     # Step 4: Two pointers
#     while l < r:
#         total = num[i] + num[l] + num[r]

#         # Update closest sum if this is closer to target
#         if abs(total - target) < abs(closest_sum - target):
#             closest_sum = total

#         # Move pointers
#         if total < target:
#             l += 1
#         elif total > target:
#             r -= 1
#         else:
#             # Exact match
#             closest_sum = total
#             break  # Can't get better than exact match
# print("Closest sum to target:", closest_sum)
       
     
    
# def triplets_with_smaller_sum(nums, target):
#     nums.sort()
#     count = 0

#     for i in range(len(nums) - 2):
#         l = i + 1
#         r = len(nums) - 1

#         while l < r:
#             current_sum = nums[i] + nums[l] + nums[r]

#             if current_sum < target:
#                 count += (r - l)
#                 l += 1
#             else:
#                 r -= 1

#     return count

#sliding window ############################################################# 

# num = [1,2,3,4,5,3,9]
# k = 2

# i = 0
# j = 1
# summ = num[i] + num[j]   # first window sum
# max_sum = summ           # store maximum

# while j < len(num) - 1:
#     # slide window
#     summ = summ - num[i] + num[j+1]

#     if summ > max_sum:
#         max_sum = summ

#     i += 1
#     j += 1

# print(max_sum)






# nums=[2,3,1,2,4,3,32]
# target = 7

# left = 0
# curr_sum = 0
# min_len = float('inf')

# for right in range(len(nums)):
#     curr_sum += nums[right]

#     while curr_sum >= target:
#         min_len = min(min_len, right - left + 1)
#         curr_sum -= nums[left]
#         left += 1

# 
# print(min_len if min_len != float('inf') else 0)
# 
# 
# 
# 
# 
# 
# 
# num=[3,1,2,7,4,2,1,1,5]
# t=8
# summ=0
# k=[]
# i=0
# for right in range(len(num)):
#     summ+=num[right]

#     while summ>=t:
#             summ-=num[i]
#             k.append(right-i+1)
#             i+=1
# print(min(k))        



# num = [3,1,2,7]
# t = 8

# summ = 0
# left = 0
# max_len = 0

# for right in range(len(num)):
#     summ += num[right]          # expand window

#     while summ > t:             # shrink if invalid
#         summ -= num[left]
#         left += 1

#     max_len = max(max_len, right - left + 1)

# print(max_len)
# The longest substring with exactly 3 distinct characters is:
# "cbebebe"
# Length = 7  



 
# s = "aabacbebebe"
# k = 3

# left = 0
# freq = {}
# max_len = -1

# for right in range(len(s)):
#     freq[s[right]] = freq.get(s[right], 0) + 1

#     while len(freq) > k:
#         freq[s[left]] -= 1
#         if freq[s[left]] == 0:
#             del freq[s[left]]
#         left += 1

#     if len(freq) == k:
#         max_len = max(max_len, right - left + 1)

# print(max_len)

# arr = [2, 1, 5, 1, 3, 2]
# k= 7
# summ=0
# left=0
# max_len=0
# for right in range(len(arr)):
#     summ+=arr[right]
    
#     while summ>k:
#         summ-=arr[left]
#         left+=1
    
#     max_len=max(max_len,right-left+1)    
        
# print(max_len)     


# arr = [2,3,1,2,4,3]
# target = 7
# left=0
# summ=0
# max_len=float('inf')
# for right in range(len(arr)):
#     summ+=arr[right]
    
#     while summ>=target:
#         max_len=min(max_len,right-left+1)
#         summ-=arr[left]
#         left+=1
            
    
  
# print(max_len)       

# s=[1, 2, 1, 2, 3]
# left=0
# k=2
# freq={}
# max_len=-1
# for right in range(len(s)):
#     freq[s[right]] = freq.get(s[right], 0) + 1
    
#     while len(freq)>k:
#         freq[s[left]]-=1
#         if freq[s[left]]==0:
#             del freq[s[left]]
#         left += 1
#     if len(freq)==k:
#         max_len = max(max_len, right - left + 1)

# print(max_len)        
# s='aabb'
# freq={}
# left=0
# for right in range(len(s)):
#     freq[s[right]] = freq.get(s[right], 0) + 1
    
#     while right-left+1:
#         if len(freq)==right-left+1:
#             print("yes")
#             break
#         else:
#             print("no")        
#             break





# n=[1,2]
# n1=[3,4]
# i=0
# j=0
# k=[]
# while i < len(n):
#     if n[i]<=n1[j]:
#         if n[i]!=k:
#             k.append(n[i])
#         i+=1
#     if n[i]>=n1[j]:
#         if n1[j]!=k:
#             k.append(n1[j])           
#         j+=1
#     if n[i]!=k:
#         k.append(n[i])
#         i+=1
#     if n1[j]!=k:
#         k.append(n1[j])
#         j+=1        
# print(k)                


# arr = [2, 1, 5, 1, 3, 2]
# K = 7
# left=0
# summ=0
# maxl=0
# for right in range(len(arr)):
#     summ+=arr[right]
    
#     while summ>K:
#         summ-=arr[left]
#         left+=1
#     maxl=max(maxl,right-left+1)    
# print(maxl)    


# arr = [1,2,1,1]
# K = 3
# left=0
# summ=0
# maxl=0
# for right in range(len(arr)):
#     summ+=arr[right]
    
#     while summ>K:
#         summ-=arr[left]
#         left+=1
#     maxl=max(maxl,right-left+1)    
# print(maxl)    

#fixed window ###########################################
# arr = [2, 1, 5, 1, 3, 2]
# k = 2
# windsum=sum(arr[:k])
# maxx=windsum
# for i in range(k,len(arr)):
#     windsum+=arr[i]
#     windsum-=arr[i-k]
#     maxx=min(windsum,maxx) 

# print(maxx)




# arr = [2, 3, 4, 1, 5]
# k = 2
# x = 5
# count=0
# ws=sum(arr[:k])
# maxs=ws
# for i in range(k,len(arr)):
#     ws+=arr[i]
#     ws-=arr[i-k]
#     if ws>x:
#         count+=1
# print(count)



# s = "abaiiidef"
# k = 3
# count=0
# maxc=0
# vowel={'a','e','i','o','u'}
# windsum=s[:k]
# for i in windsum:
#     if i in vowel:
#         count+=1


# arr = [1,3,2,6,-1,4,1,8,2]
# k = 5
# lis=[]
# summ=sum(arr[:k])
# j=summ/k
# lis.append(j)
# print(summ)
# for i in range(k,len(arr)):
#     summ+=arr[i]
#     summ-=arr[i-k]
#     n=summ/k
#     lis.append(n)
# print(lis)


# arr = [9, 10, 1, 2, 3, 5] 
# k = 5
# summ=sum(arr[:k])
# maxx=summ
# for i in range(1,k+1):
#     summ-=arr[k-i]
#     summ+=arr[len(arr)-i]
#     maxx=max(maxx,summ)
# print(maxx)    

# s1 = "ab"
# s2 = "eidbaooo"
# k=2
# lenn=s2[:k]
# print(lenn)
# for i in range(k,len(s2)):
    
from collections import Counter
    
# s1 = "ab"
# s2 = "eidbaooo"

# k = len(s1)
# map=(Counter(s1))
# for i in range(len(s2)-1):
#     sub = s2[i:i+k]
#     map2=Counter(sub)
#     if map==map2:
#         print("T")
#         break
#     else:
#         print("f")
#         break



    
# from collections import Counter

# s1 = "ab"
# s2 = "eidbaooo"

# k = len(s1)
# map1 = Counter(s1)

# found = False

# for i in range(len(s2) - 1):
#     sub = s2[i:i+k]
#     map2 = Counter(sub)
    
#     if map1 == map2:
#         print("True")
#         found = True
#         break

# if not found:
#     print("False")
    
    
# def checkInclusion(s1, s2):
#     if len(s1) > len(s2):
#         return False

#     k = len(s1)
#     need = [0] * 26
#     window = [0] * 26

#     # Build frequency of s1 and first window
#     for i in range(k):
#         need[ord(s1[i]) - ord('a')] += 1
#         window[ord(s2[i]) - ord('a')] += 1

#     if need == window:
#         return True

#     # Slide the window
#     for i in range(k, len(s2)):
#         window[ord(s2[i]) - ord('a')] += 1        # Add new char
#         window[ord(s2[i - k]) - ord('a')] -= 1    # Remove old char

#         if need == window:
#             return True

#     return False
# s1="aab"
# s2="sdfgaab"
# checkInclusion(s1,s2)
      
# s="AABABAB"
# k=1
# freq = [0] * 26
# left = 0
# maxFreq = 0
# maxLen = 0

# for right in range(len(s)):
#     idx=ord(s[right])-ord('A')
#     freq[idx]+=1
#     maxFreq=max(maxFreq,freq[idx])
    
    
#     while (right-left+1)-maxFreq>k:
#         freq[ord(s[left])-ord('A')]-=1
#         left+=1
#     maxLen=max(maxLen,right-left+1)    
# print(maxLen)


# nums = [1,1,1,0,0,0,1,1,1,1,0]
# k = 2
# ones=0
# zeros=0
# maxlen=0
# maxfreq=0
# left=0
# for right in range(len(nums)):
#     if nums[right]==0:
#         zeros+=1
    
    
#     while zeros>k:
#         if nums[left]==0:
#             zeros-=1
#         left+=1
#     maxlen=max(maxlen,right-left+1)    
# print(maxlen)    




# nums= [1,1,0,1,1,0,1,1,1,1,1,0,1,1]

# k = 1

# ones=0
# zeros=0
# maxlen=0
# maxfreq=0
# left=0

# for right in range(len(nums)):
#     if nums[right]==0:
#         zeros+=1
#     while zeros>k:
#         if nums[left]==0:
#             zeros-=1
#         left+=1    
#     maxlen=max(maxlen,right-left)    
    
    
# print(maxlen)


# s = "ADOBECODEBANC"
# t = "ABC"
# left=0
# count={}
# tcount={}
# formed=0
# for right1 in (t):
#     tcount[right1] = count.get(right1, 0) + 1
# for right in (s):
#     count[right] = count.get(right, 0) + 1

#     if tcount[right1]==count[right]:
#         formed+=1
#     while formed>3:
        

# s= "abccde"
# count={}
# freq = [0] * 26
# left=0
# k=0
# maxLen=0
# for right in range(len(s)):
#     idx=ord(s[right])-ord('a')
#     freq[idx]+=1
#     while freq[idx]>1:
#         freq[ord(s[left])-ord('a')]-=1
#         left+=1
#     maxLen=max(maxLen,right-left+1)
# print(maxLen)




#KADANES ALGORITHM########################################################################################

# arr=[1,2,3,-4]
# best=arr[0]
# maxx=0
# for i in range(1,len(arr)):
#     v1=best+arr[i]
#     v2=arr[i]
#     best=max(v1,v2)
#     maxx=max(best,maxx)
# print(maxx)

# def maximumSum(arr):
#     keep = arr[0]      # no deletion
#     delete = 0         # one deletion used
#     result = arr[0]

#     for i in range(1, len(arr)):
#         delete = max(keep, delete + arr[i])  # delete current OR extend deleted
#         keep = max(arr[i], keep + arr[i])    # normal kadane
#         result = max(result, keep, delete)

#     return result
# print(maximumSum(arr=[1,-2,-2,3]))


# arr = [2, -1, 2]

# fmax = arr[0]   # max subarray ending here
# fmin = arr[0]   # min subarray ending here
# lmax = arr[0]   # global max subarray sum
# lmin = arr[0]   # global min subarray sum

# for i in range(1, len(arr)):
#     v1 = arr[i]
    
#     # store old fmax before updating
#     prev_fmax = fmax
#     prev_fmin = fmin

#     # Kadane for maximum sum
#     fmax = max(v1, v1 + prev_fmax)

#     # Kadane for minimum sum
#     fmin = min(v1, v1 + prev_fmin)

#     lmax = max(lmax, fmax)
#     lmin = min(lmin, fmin)

# last = max(abs(lmax), abs(lmin))
# print(last)



# arr=[5,-3,5]
# best=arr[0]
# best1=arr[0]
# maxx=0
# lmax=0
# lmax1=0
# sumofarray=sum(arr)
# for i in range(1,len(arr)):
#     v1=arr[i]
#     v2=arr[i]+best
#     best=max(v1,v2)
#     lmax=max(lmax,best)
#     best1=min(v1,v2)
#     lmax1=min(lmax1,best1)
#     lastend=(sumofarray-lmax1) 
#     k=max(lastend,lmax)
# print(k)    


#prefixsum##############33333333333333333333###########33333############3333333333333333333########333333333333333333

# def pivotIndex(nums):
#     total_sum = sum(nums)
#     left_sum = 0

#     for i in range(len(nums)):
#         right_sum = total_sum - left_sum - nums[i]

#         if left_sum == right_sum:
#             return i

#         left_sum += nums[i]

#     return -1


# print(pivotIndex([1, 7, 3, 6, 5, 6]))  # Output: 3


# nums = [10, 4, -8, 7]
# left=0
# summ=sum(nums)
# count=0
# for i in range(len(nums)):
#     right=summ-nums[i]-left
#     if right>=left:
#         count+=1
#     left+=nums[i]
# print(count)    



 
# arr = [1, 2, 3, 4, 5]
# k = 4
# pre=0
# count=0
# hash={0:1}

# for i in range(len(arr)):
#     pre+=arr[i]
#     if pre%4==1:

        

        
#         count+=hash[l]
#     hash[pre] = hash.get(pre, 0) + 1  
# print(count)      
    
    
    
# arr = [1, 2, 3, 4, 5]
# k = 4

# pre = 0
# count = 0
# rem_map = {0: 1}   # remainder 0 seen once

# for num in arr:
#     pre += num
    
#     rem = pre % k
    
#     # Handle negative remainder case (important in some languages)
#     if rem < 0:
#         rem += k
    
#     if rem in rem_map:
#         count += rem_map[rem]
    
#     rem_map[rem] = rem_map.get(rem, 0) + 1

# print(count)
    
    
    
# arr = [5,0,0,0]
# k = 3

# pre = 0
# count = 0
# rem_map = {0: 1} 

# for i in range(len(arr)):
#     pre+=arr[i]
#     rem =pre%k
    
#     if rem in rem_map:
#         count+=rem_map[rem]
#     rem_map[rem] = rem_map.get(rem, 0) + 1            
# print(count)    



# nums = [0, 1, 0, 1,0,1]
# pre=0
# count=0
# rem_map={0:1}
# for i in range(len(nums)):
#     if nums[i]==0:
#         nums[i]=-1
#     pre+=nums[i]
#     if pre in rem_map:
#         count+=rem_map[pre]
#     rem_map[pre] = rem_map.get(pre, 0) + 1
        
# print(count)     


# nums = [1, 0,1,0]
# pre=0
# count=0
# rem_map={0:1}
# store=0
# for i in range(len(nums)):
#     if nums[i]==0:
#         nums[i]=-1
#     pre+=nums[i]
#     if pre ==0:
#         store=i+1
#         count=max(count,store) 
      
# print(count)        
        
        
# nums = [0,1,0,0,1,1,0,1]
# pre=0
# count=0
# rem_map={}
# store=0
# for i in range(len(nums)): 
#     if nums[i]==0:
#         nums[i]=-1
#     pre+=nums[i]
#     if pre==0:
#         rem_map[i] = rem_map.get(i, 0) + 1
# print(max(rem_map.keys()) +1 )
  
#def longest_subarray_sum_k(nums, k):
#     prefix_sum = 0
#     max_len = 0
#     first_seen = {}  # prefix_sum -> first index

#     for i in range(len(nums)):
#         prefix_sum += nums[i]

#         # Case 1: subarray from index 0 to i
#         if prefix_sum == k:
#             max_len = i + 1

#         # Case 2: subarray between two indices
#         if (prefix_sum - k) in first_seen:
#             length = i - first_seen[prefix_sum - k]
#             max_len = max(max_len, length)

#         # Store prefix sum ONLY if not seen before (to keep longest length)
#         if prefix_sum not in first_seen:
#             first_seen[prefix_sum] = i

#     return max_len


# # Example run
# nums = [1, -1, 5, -2, 3]
# k = 3
# print(longest_subarray_sum_k(nums, k))  # Output: 4

  
# nums = [1, 2, 1, 2, 1]
# k = 2
# freq = {0: 1}
# count = 0
# summ = 0

# for i in range(len(nums)):
#     if nums[i] == 2:
#         nums[i] = 0
#     summ += nums[i]
    
#     sub = summ - k
#     if sub in freq:
#         count += freq[sub]  # first add to count
    
#     # then update freq
#     freq[summ] = freq.get(summ, 0) + 1

# print(count) 
  
#MERGE INTERVALLS################################################################################################################################################################################################### 
# arr =[[1,3],[6,9]]
# newInterval = [2,5]


# st1=arr[0][0]
# end1=arr[0][1]
# result=[]
# for i in range(len(arr)):
#     st2=arr[i][0]
#     end2=arr[i][1]
#     if st1<newInterval[0] and st2>newInterval[0]:
#         result.append([newInterval[0],newInterval[1]])
#     st1=st2
#     end1=end2    
#     result.append([st1,end1])    


# def merge_interval(arr):
    
#     st=arr[0][0]
#     end=arr[0][1]
#     result=[]
#     for i in range(len(arr)):
#         st2=arr[i][0]
#         end2=arr[i][1]
#         if end>=st2:
#             end=max(end,end2)
#         else:
#             result.append([st,end])
#             st=st2
#             end=end2  
            
            
#     result.append([st,end])
#     return result
# print(merge_interval(result))


# arr =[[1,3],[6,9]]
# newInterval = [2,5]


# st1=arr[0][0]
# end1=arr[0][1]
# result=[]
# for i in range(len(arr)):
#     st2=arr[i][0]
#     end2=arr[i][1]
#     if st1<newInterval[0] and st2>newInterval[0]:
#         result.append([newInterval[0],newInterval[1]])
#     st1=st2
#     end1=end2    
#     result.append([st1,end1])    
# print(result)

# arr1  = [[0,2],[5,10],[13,23],[24,25]]
# arr2 = [[1,5],[8,12],[15,24],[25,26]]
# st1=arr1[0][0]
# end1=arr1[0][1]
# st2=arr2[0][0]
# end2=arr2[0][1]
# result=[]
# for i in range(len(arr1)):
#     for j in range(len(arr2)):
#         if end1>st2:
#             result.append([end1,st2])
        
#         elif st1==end2:
#             result.append([st1,end2])




# arr1 = [[3,7]]
# arr2 = [[1,2]]



# result = []
# i=0
# j=0
# while i<len(arr1) and j<len(arr2):
    
#     st1=arr1[i][0]
#     end1=arr1[i][1]
#     st2=arr2[j][0]
#     end2=arr2[j][1]
#     if not (end1<st2 or end2<st1):
#         maxx=max(st1,st2)
#         minn=min(end1,end2)
#         result.append([maxx,minn])
#     if end1<end2:
#             i+=1
#     else :
#             j+=1
        
# print(result)            


# arr=[[1,3], [2,6], [8,10]]
# st1=arr[0][0]
# end1=arr[0][1]
# found=False
# for i in range(1,len(arr)):
#     st2=arr[i][0]
#     end2=arr[i][1]
#     if end1>=st2:
#         found=True
#         print("true")
#         break
    
#     st1=st2
#     end1=end2 
# else:
#     print("False") 


# def minMeetingRooms(intervals):
#     if not intervals:
#         return 0

#     # Step 1: Separate start and end times
#     starts = sorted([i[0] for i in intervals])
#     ends = sorted([i[1] for i in intervals])

#     # Step 2: Initialize pointers and counters
#     s_ptr = 0  # pointer for starts
#     e_ptr = 0  # pointer for ends
#     rooms = 0
#     max_rooms = 0

#     # Step 3: Process all meetings
#     while s_ptr < len(starts):
#         if starts[s_ptr] < ends[e_ptr]:
#             # New meeting starts before earliest ended → need a new room
#             rooms += 1
#             s_ptr += 1
#         else:
#             # Earliest meeting ended → room freed
#             rooms -= 1
#             e_ptr += 1

#         max_rooms = max(max_rooms, rooms)

#     return max_rooms

# # Example
# intervals = [[1, 5], [2, 6], [8, 9], [3, 7]]
# print(minMeetingRooms(intervals))  # Output: 3



# intervals = [[1,2], [2,3], [3,4],[1,3]]
# intervals.sort(key=lambda x: x[1])

# end1=intervals[0][1]
# count=0
# for i in range(1,len(intervals)):
#     st2=intervals[i][0]
#     end2=intervals[i][1]
#     if end1>st2:
#         count+=1
#     else:
#         end1=end2 
# print(count)      



# intervals = [[1,10],[2,3],[4,5],[6,7]]

# intervals.sort(key=lambda x:x[1])
# count=1
# end1=intervals[0][1]
# for i in range(1,len(intervals)):
#     st2=intervals[i][0]
#     end2=intervals[i][1]
#     if end1<=st2:
#         count+=1
    
#         end1=end2
# print(count)        


# intervals = [[1,2], [3,4], [0,6], [5,7], [8,9], [5,9]]

# intervals.sort(key=lambda x:x[1])
# print(intervals)
# end1=intervals[0][1]
# for i in range(1,len(intervals)):
#     st2=intervals[i][0]
#     end2=intervals[i][1]
    
# start = [1, 3, 0, 5, 8, 5]
# end   = [2, 4, 6, 7, 9, 9]
# start.sort(key=lambda x:x)
# end.sort(key=lambda x:x)




# activities = [[1,10], [2,7], [3,19], [8,12], [10,20]]
# activities.sort(key=lambda x: x[1])
# end1=activities[0][1]
# count=1
# for i in range(1,len(activities)):
#     st2=activities[i][0]
#     end2=activities[i][1]
#     if end1<=st2:
#         count+=1
#     else:
#         end1=end2
# print(count)        


# activities = [
#   [1,4,4],
#   [2,5,2],
#   [3,6,3],
#   [7,8,6]
# ]
# # Two-pointer prep (separate and sort)
# # prepare arrays
# starts = sorted([(x[0], x[2]) for x in activities])  # (start, load)
# ends = sorted([(x[1], x[2]) for x in activities])    # (end, load)

# i = 0
# j = 0
# current_load = 0
# max_load = 0

# while i < len(starts) and j < len(ends):
#     if starts[i][0] < ends[j][0]:
#         current_load += starts[i][1]
#         max_load = max(max_load, current_load)
#         i += 1
#     else:
#         current_load -= ends[j][1]
#         j += 1

# print(max_load)



#stack############################################333333333333333333##################################################################33333333333333333333333333333333333333333333333333333333333333333333


# arr=[1,2,2,1,3,1]
# stack=[]
# result=[]
# for i in arr:
#     if stack and stack[-1]==i:
#         stack.pop()
        
#     else:
#         stack.append(i)
        
# print(stack)
    
    
    
    

# s = "(}"
# result=[]
# for i in s:
#     if i =='(' or i=='[' or i== '{':
#         result.append(i)
#     elif result and i==')' and result[-1]=='(':
#         result.pop()
#     elif result and i=='}' and result[-1]=='{':
#         result.pop()  
#     elif result and i==']' and result[-1]=='[':
#         result.pop()
      
# if not result:
#     print("True")
# else:
#     print('False')              
        


# s="hello"
# stack=list(s)
# reve=''
# while stack:
#     reve+=stack.pop()
# print(reve)    
 
 
 
 
# arr=[3,1,2,6,5]
# result=[]
# sta=[]
# for i in range(len(arr)):
#     while sta and sta[-1]<=arr[i]:
#         sta.pop()
#     if not sta:
#         result.append(-1)
#     else:
#         result.append(sta[-1])
#     sta.append(arr[i])          
# print(result)               
# print(sta)

# arr = [73, 74, 75, 71, 69, 72, 76, 73]
# sta=[]
# result=[]
# for i in range(len(arr)-1,-1,-1):
#     while sta and arr[sta[-1]]<=arr[i]:
#         sta.pop()
  
    
#     if not sta:
#         result.append(0)
#     else:
#         result.append(sta[-1]-i)
        
#     sta.append(i)       
# print(result)

    
    

# nums = [1,2,3,4,5]
# n = len(nums)
# result = [-1] * n
# stack = []

# for i in range(2 * n):
#     while stack and nums[stack[-1]] <nums[i % n]:
#         index = stack.pop()
#         result[index] = nums[i % n]
    
#     if i < n:
#         stack.append(i)

# print(result)


# nums =  [100, 80, 60, 70, 60, 75, 85]
# n = len(nums)
# result = [1]*n
# stack = []
# count=0
# for i in range(len(nums)):
#     while stack and nums[stack[-1]]<=nums[i]:
#         k=stack.pop()
#     if stack:    
#         index=i-stack[-1]
#         result[i]=index  
#     else:
#         result[i]=i+1          
#     stack.append(i)   
# print(result)



# nums =[3,5,-6,2,-1,4]
# result=[]
# stack=[]
# for i in range(len(nums)):
    
#     while stack and stack[-1]<nums[i]:
#             stack.pop()
#     if abs(stack[-1])<abs(nums[i]):
#             result.append(abs(nums[i])) 
#     else:
#             result.append(abs(stack[-1]))
#     stack.append(nums[i]) 
# print(result)               


# nums =[3,5,-6,2,-1,4]
# result=[]
# stack=[]
# for i in range(len(nums)):
#     if nums[i]>0:
#         stack.append(nums[i])
#     else:
        
#         while stack and stack[-1]<abs(nums[i]):
#             stack.pop()
#             if stack and abs(nums[i])>stack[-1]:
#                 result.append(abs(nums[i]))
#             else:
#                 stack and result.append(stack[-1])        
    
# print(result)


# nums = [5,10,-5]
# stack = []

# for num in nums:
    
#     if num > 0:
#         stack.append(num)
    
#     else:
#         while stack and stack[-1] > 0 and stack[-1] < abs(num):
#             stack.pop()
        
#         if stack and stack[-1] > 0:
#             if stack[-1] == abs(num):
#                 stack.pop()
#             # else stack[-1] > abs(num)
#             # do nothing (negative gets destroyed)
#         else:
#             stack.append(abs(num))

# print(stack)

# arr=['a','a','a','b','a']
# sta=[]
# result=[]
# k=3
# for i in range(len(arr)):
#     if sta and sta[-1][0]==arr[i]:
#         sta[-1][1] += 1
#         if  sta and sta[-1][1]>=k:
#             sta.pop()
#     else:
        
#         sta.append([arr[i],1]) 
# re=''
# for i,j in sta:
#     re+=i*j
# print(re)

# arr="/home//foo/../bar/./baz/"
# sta=[]
# result=[]
# for i in range(len(arr)):
#     if arr[i]==".":
#         if sta[-1][1]==2:
#             while sta and sta[-1]=='/':
#                 sta.pop()
        
#     sta.append([arr[i],1])
# re=''
# for i,j in sta:
#     re+=i*j
# print(re)


# arr = "/home//foo/../bar/./baz/"

# parts = arr.split('/')
# sta=[]
# result=[]
# for part in parts:
    
#     if sta and part=='..':
#         sta.pop()
#     if part=='.'or part=='':
#         continue    
            
#     elif part!='..' :
#      sta.append(part)
    

# result = "/" + "/".join(sta)

# print(result)

# num = "120340"
# k = 2
# result=[]
# sta=[]
# count=0
# for i in range(len(num)):
#     if count<k:
#         while sta and sta[-1]>num[i]:
#             sta.pop()
#             count+=1
#     sta.append(num[i])   
# if count==0:
#     sta=num[:-k]
        
# result = ''.join(sta).lstrip('0') or '0'    
# print(result)

    
    

#     if sta and sta[-1][0]==arr[i]:
#         sta[-1][1] += 1
#         if  sta and sta[-1][1]>=k:
#             sta.pop()
#     else:
        
# s = "bcabc"

# last = {c:i for i,c in enumerate(s)}
# stack = []
# seen = set()
# for i ,c in enumerate(s):
#     if s[i] in seen:
#         continue
#     while stack and stack[-1]>s[i] and last[stack[-1]]>i:
#          seen.remove(stack.pop())
#     stack.append(s[i])
#     seen.add(s[i])
# print(stack)
# print(stack)


       
#HASH MAP############################################################################################################3#################################################################################################################
# s = "leetcodel"
# hash = {}

# # first loop → count frequency
# for i in range(len(s)):
#     if s[i] in hash:
#         hash[s[i]] += 1
#     else:
#         hash[s[i]] = 1

# # second loop → find first unique character
# for i in range(len(s)):
#     if hash[s[i]] == 1:
#         print(i)
#         break
# else:
#     print(-1)



# ransomNote = "aa"
# magazine = "aab"
# hash={}
# hash1={}
# for ch in ransomNote:
#     hash[ch] = hash.get(ch, 0) + 1
# for ch1 in magazine:
#     hash1[ch1] = hash1.get(ch1, 0) + 1
#     if hash[ch]<hash1[ch1]:
#         print("True")
 
#binary search#############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################333333333333333333333333333333333333333333333
# arr = [1, 3, 6, 8, 10]
# target = 5
# low=0
# result=-1
# high=len(arr)-1
# while low<=high:
#     mid=(high+low)//2

#     if arr[mid] <target:
#         low=mid+1
#     else:
#         result=mid
#         high=mid-1
# print(result)
  

#First and Last Position problem
# arr = [1,3,3,3,3,3,5]
# target = 3
# low=0
# result=-1
# high=len(arr)-1
# while low<=high:
#     mid=(high+low)//2

#     if arr[mid]>target:
#         high=mid-1
    
#     elif arr[mid]<target:
#             low=mid+1
#     else:
#         result=mid
#         high=mid-1
# print(result)


# arr = [0,1,0]
# low = 0
# result = 0
# high = len(arr) - 1

# while low <= high:
#     mid = (high + low) // 2
    
#     if arr[mid]<arr[mid+1]:
#         low=mid+1
#     else:
#         result=mid
#         high=mid-1
# print(result)
        
        
# arr = [7, 8, 9, 1, 2, 3, 4, 5, 6]
# low = 0
# high = len(arr) - 1        
# while low < high:
#     mid = (high + low) // 2
    
#     if arr[mid]<arr[high]:
#         high=mid
#     elif arr[mid]>arr[high]:
#         low=mid+1
# print(arr[low]) 
        
# arr = [7,8,9,1,2,3,4,5,6]
# target = 1

# low = 0
# high = len(arr) - 1   # ❗ fix

# while low <= high:
#     mid = (low + high) // 2
    
#     if arr[mid] == target:
#         print(mid)
#         break
    
#     elif arr[low] <= arr[mid]:   # ❗ better condition
#         if arr[low] <= target <= arr[mid]:
#             high = mid - 1
#         else:
#             low = mid + 1
    
#     else:
#         if arr[mid] <= target <= arr[high]:   # ❗ missing check
#             low = mid + 1
#         else:
#             high = mid - 1
        
# arr = [3,1,2,3,3,3,3]
# target = 0

# low = 0
# high = len(arr) - 1   # ❗ fix

# while low <= high:
#     mid = (low + high) // 2
    
#     if arr[mid] == target:
#         print(mid)
#         break
#     if arr[mid]==arr[low]==arr[high]:
#         low=low+1
#         high=high-1
#         continue
#     elif arr[mid]>arr[low]:
#         if arr[low]<=target<=arr[mid]:
#             high=mid-1
#         else:
#             low=mid+1
#     else:
#         if arr[high]>=target>=arr[mid]:
#             low=mid+1
#         else:
#             high=mid-1




# piles = [3, 6, 7, 11]
# h = 8

# low = 1
# high = max(piles)

# ans = 0

# while low <= high:
#     mid = (low + high) // 2
#     total_hours=0
#     for i in piles:
#         if i%mid==0:
#             total_hours+=i//mid
#         else:
#             total_hours+=(i//mid)+1
#     if total_hours<=h:
#         ans=mid
#         high=mid-1
#     else:
#         low=mid+1
# print(ans)
# print(ans)


# from typing import List

# class Solution:
#     def singleNonDuplicate(self, arr: List[int]) -> int:
        
#         if len(arr) == 1:
#             return arr[0]

#         low = 0
#         high = len(arr) - 1

#         while low <= high:
#             mid = (low + high) // 2

#             if mid == 0:
#                 if arr[mid] != arr[mid+1]:
#                     return arr[mid]
#                 else:
#                     low = mid + 1
#                     continue

#             if mid == len(arr) - 1:
#                 if arr[mid] != arr[mid-1]:
#                     return arr[mid]
#                 else:
#                     high = mid - 1
#                     continue

#             if arr[mid] != arr[mid-1] and arr[mid] != arr[mid+1]:
#                 return arr[mid]

#             if mid % 2 == 0:
#                 if arr[mid] == arr[mid+1]:
#                     low = mid + 2
#                 else:
#                     high = mid - 2
#             else:
#                 if arr[mid] == arr[mid-1]:
#                     low = mid + 1
#                 else:
#                     high = mid - 1
#                     print("hello")

# arr = [12, 34, 67, 90]
# m = 2

# low = max(arr)
# high = sum(arr)
# ans = 0

# while low <= high:
#     mid = (low + high) // 2
    
#     students = 1
#     pages = 0
    
#     for i in range(len(arr)):
#         if pages + arr[i] <= mid:
#             pages += arr[i]
#         else:
#             students += 1
#             pages = arr[i]
    
#     if students <= m:
#         ans = mid
#         high = mid - 1
#     else:
#         low = mid + 1

# print(ans)


# arr = [1,2,3,4,5,6,7,8,9,10]
# days= 5

# low = max(arr)
# high = sum(arr)
# ans = 0

# while low <= high:
#     mid = (low + high) // 2
#     taken_days=1
#     capacity=0
#     for i in range(len(arr)):
#         if capacity+arr[i]<=mid:
#             capacity+=arr[i]
#         else:
#             taken_days+=1
#             capacity=arr[i]
#     if taken_days<=days:
#         ans=mid
#         high=mid-1
#     else:
#         low=mid+1
# print(ans)
        
        
        
# arr = [1,2,3,3,3,3,3,3,3,3,4,5]

# tar = 3

# # -------- First Occurrence --------
# low = 0
# high = len(arr) - 1
# first = -1

# while low <= high:
#     mid = (low + high) // 2
    
#     if arr[mid] == tar:
#         first = mid
#         high = mid - 1
#     elif arr[mid] < tar:
#         low = mid + 1
#     else:
#         high = mid - 1

# # -------- Last Occurrence --------
# low = 0
# high = len(arr) - 1
# last = -1

# while low <= high:
#     mid = (low + high) // 2
    
#     if arr[mid] == tar:
#         last = mid
#         low = mid + 1
#     elif arr[mid] < tar:
#         low = mid + 1
#     else:
#         high = mid - 1

# print(last ,first)

# arr = [3,4,5,1,2]
# low = 0
# high = len(arr) - 1


# while low <= high:
#     mid = (low + high) // 2
    
#     if arr[mid]<arr[high] and arr[mid]<arr[low]:
#         print([mid])
#         break
#     if arr[mid]>arr[high]:
#             low=mid+1
#     else:
#         high=mid-1
# print(mid)
            
            
# class Solution:
#     def minDays(self, bloomDay, m, k):
        
#         arr = bloomDay
#         low = min(arr)
#         high = max(arr)

#         ans = -1
        
#         while low <= high:
#             mid = (low + high) // 2
            
#             taken = 0     
#             count = 0    
            
#             for i in range(len(arr)):
#                 if arr[i] <= mid:
#                     count += 1
#                     if count == k:    
#                         taken += 1
#                         count = 0
#                 else:
#                     count = 0          
            
#             if m <= taken:
#                 ans = mid
#                 high = mid - 1
#             else:
#                 low = mid + 1
        
#         return ans

        # arr = weights
        
        # low = max(arr)
        # high = sum(arr)

        # while low <= high:
        #     summ = 0
        #     taken = 1
        #     mid = (low + high) // 2
            
        #     for i in range(len(arr)):
        #         if summ + arr[i] <= mid:
        #             summ += arr[i]
        #         else:
        #             summ = arr[i]
        #             taken += 1
            
        #     if taken <= days:
        #         ans = mid
        #         high = mid - 1
        #     else:
        #         low = mid + 1
        
        # return ans
               
               
# arr = [1, 2, 4, 8, 9]
# k = 3
# low = 1
# high = arr[-1] - arr[0]
# ans=-1
# while low <= high:

#     taken=1
#     mid = (low + high) // 2 
#     first=arr[0] 
#     for i in  range(1,len(arr)):
        
#         if arr[i]-first>=mid:
#             taken+=1
#             first=arr[i]
#     if taken>=k:
#         ans=mid
#         low=mid+1
#     else:
#         high=mid-1
# print(ans)
               
               
# arr = [10,20,30,40]
# k = 2
# low = max(arr)
# high =sum(arr)
# ans=0
# while low <= high:

#     taken=1
#     mid = (low + high) // 2 
#     summ=0
#     for i in  range(len(arr)): 
#         if arr[i]+summ<=mid:
#             summ+=arr[i]
#         else:
#             summ=arr[i]
#             taken+=1
#     if taken<=k:
#         high=mid-1
#         ans=mid
#     else:
#         low=mid+1
# print(ans)   

#######linked_list fast and slow pointers ############################################################################################################################################################################################################
# Node class



# class ListNode:
#     def __init__(self, val):
#         self.val = val
#         self.next = None



# a = ListNode(1)
# b = ListNode(2)
# c = ListNode(0)
# d = ListNode(-4)

# a.next = b
# b.next = c
# c.next = d
# d.next = b   #

# head = a


# slow = head
# fast = head

# while fast is not None and fast.next is not None:
#     slow = slow.next
#     fast = fast.next.next

#     if slow == fast:
#         print(True)
#         break
# else:
#     print(False)
    
    
# class ListNode:
#     def __init__(self, val):
#         self.val = val
#         self.next = None


# a = ListNode(1)
# b = ListNode(2)
# c = ListNode(3)
# d = ListNode(3)

# a.next = b
# b.next = c
# c.next = d
# d.next = b  

# head = 0                                 #fast.next is not None:  this is to check is there any cycle or not


# slow = head
# fast = head

# while fast is not None and fast.next is not None:
#     slow = slow.next
#     fast = fast.next.next

#     if slow == fast:
#         slow = head
        
#         while slow != fast:
#             slow = slow.next
#             fast = fast.next
        
#         print(slow.val)  
#         break
    
     
# arr = [1, 1, 3, 5, 7]

# slow = arr[0]
# fast = arr[0]

# # Step 1: detect cycle
# while True:
#     slow = arr[slow]
#     fast = arr[arr[fast]]

#     if slow == fast:
#         break

# # Step 2: find duplicate (cycle start)
# slow = arr[0]

# while slow != fast:
#     slow = arr[slow]
#     fast = arr[fast]

# print(slow)    

# def funn(n):
    
#     summ=0
    
#     while n>0:
#         d=n%10
#         s=d**2
#         summ+=s
#         n=n//10
#     return summ

# arr =20

# slow = arr
# fast = arr

# while (fast!=1):
#     slow=funn(slow)
#     fast=funn(fast)
#     fast=funn(fast)
    
#     if fast==1:
#         print(True)
#         break
#     if fast==slow:
#         print(False)
#         break



    

# def middleNode(self, head):
#     slow = head
#     fast = head

#     while fast is not None and fast.next is not None:
#         slow = slow.next
#         fast = fast.next.next

        
#     return slow


# class Solution:
#     def removeNthFromEnd(self, head, n):
#         dummy = ListNode(0)
#         dummy.next = head

#         slow = dummy
#         fast = dummy

#         # move fast n steps
#         for _ in range(n):
#             fast = fast.next

#         # move both
#         while fast.next:
#             slow = slow.next
#             fast = fast.next

#         # delete node
#         slow.next = slow.next.next

#         return dummy.next


# class Solution:
#     def isPalindrome(self, head: ListNode) -> bool:
#         if not head or not head.next:
#             return True

#         # STEP 1: Find the middle using Fast & Slow pointers
#         slow = head
#         fast = head
#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next

#         # STEP 2: Reverse the second half in place
#         # 'slow' is currently the start of the second half
#         prev = None
#         curr = slow
#         while curr:
#             next_temp = curr.next
#             curr.next = prev
#             prev = curr
#             curr = next_temp
        
#         # 'prev' is now the head of the reversed second half
        
#         # STEP 3: Compare the two halves
#         first_half = head
#         second_half = prev
        
#         result = True
#         while second_half: # We only need to check the length of the second half
#             if first_half.val != second_half.val:
#                 result = False
#                 break
#             first_half = first_half.next
#             second_half = second_half.next
            
#         # (Optional) STEP 4: Restore the list (Good practice!)
#         # You could reverse the second half back to original here
        
#         return result


##########3###########################################################################################################################################################################################################################################################33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

#recursion
# def fun(n):
#     if n == 1:
#         return 1
#     ans = fun(n - 1)
#     return n *ans
# print(fun(4))
  
  
# def fun(n):
#     if n == 1:
#         return 1
#     ans = fun(n - 1)
#     return n +ans
# print(fun(4))   

# def reverse_string(s):
#     if len(s) == 0:   # base case
#         return ""
#     return reverse_string(s[1:]) + s[0]

# print(reverse_string("hello"))

# def fun(n):
#     if n == 1:
#         return 1
#     if n==0:
#         return 0 
#     ans = fun(n - 1)+fun(n-2)
#     return ans
# print(fun(6)) 

# def fun(n):
#     if n == 1:
#         print(1)
#         return
    
#     fun(n-1)
#     print(n)
# fun(6)


# def fun(n):
#     if n==1:
#         print(1)
#         return 
#     print(n)

#     fun(n-1)
# fun(7)

# print(fun(6)) 

# def fun(arr,n):
#     if n==1:
#         return True
#     if arr[n-1]<arr[n-2]:
#         return False
#     return fun(arr,n-1)
# print(fun(arr=[1,2,6,4],n=4))



# def fun(arr,n):
#     if n==1:
#         return 1
#     last=arr[n-1]
#     ans=fun(arr,n-1)
#     return ans+last
    
# print(fun(arr=[1,2,3,4],n=4))


# def fun(s,n):
#     if len(s) == 0:
#         return "" 
#     k = ""
#     k = k + s[-1]

#     return k+fun(s[:-1],n-1)
# print((fun(s="hello",n=5)))

# def fun(arr,n):
#     if n==1:
#         return arr
#     d=arr%10
#     ans=fun(arr//10,n-1)
#     return ans+d

# print(fun(arr=2,n=4)) 


# def fun(s,l):
#     if len(s)==0:
#         return ""
#     k=""
#     last=s[-1]
#     if last!=l:
#         k+=last
#     return k+fun(s[:-1],l)

    
# print(fun(s="apple",l="p"))
        
        
        
# class Solution:
#     def generateParenthesis(self, n: int):
#         res = []

#         def backtrack(curr, open_count, close_count):
#             # base case
#             if len(curr) == 2 * n:
#                 res.append(curr)
#                 return
            
#             # add '('
#             if open_count < n:
#                 backtrack(curr + "(", open_count + 1, close_count)
            
#             # add ')'
#             if close_count < open_count:
#                 backtrack(curr + ")", open_count, close_count + 1)

#         backtrack("", 0, 0)
#         return res        
    
# class Solution:
    
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
    
    
# def searchleftright(n):
#     res = []

#     def backtrack(curr, left_count, right_count):
#         # base case
#         if len(curr) == 2 * n:
#             res.append(curr)
#             return    

#         if left_count < 2:
#             backtrack(curr + "L", left_count + 1, right_count)

#         if right_count < 2:
#             backtrack(curr + "R", left_count, right_count + 1)

#     backtrack("", 0, 0)
#     return res


# print(searchleftright(2))



            
# def searchleftright(n):
#     res = []

#     def backtrack(curr, left_count, arr):
#         # base case
#         if len(curr) == 2 * n:
#             res.append(curr)
#             return    


#         if len(arr) < 2:
#             backtrack(curr, left_count + 1, arr)



#     backtrack("", 0, arr=[1,2])
#     return res


# print(searchleftright(2))        


# def letterCombinations(digits):
    
#     if not digits:
#         return []

#     phone = {
#         "2": "abc",
#         "3": "def",
#         "4": "ghi",
#         "5": "jkl",
#         "6": "mno",
#         "7": "pqrs",
#         "8": "tuv",
#         "9": "wxyz"
#     }

#     res = []
#     def backtrack(curr,idx):
#         if len(curr)==len(digits):
#             res.append(curr)
#             return
#         letters=phone[digits[idx]]
#         for ch in letters:
#             backtrack(curr+ch,idx+1)
            
#     backtrack("",0)
#     return res
# print(letterCombinations("23"))    



# def subsets(nums):
#     res = []

#     def backtrack(index, curr):

#         # base case
#         if index == len(nums):
#             res.append(curr[:])
#             return

#         # take
#         curr.append(nums[index])
#         print(curr)
#         backtrack(index + 1, curr)

#         # don't take
#         curr.pop()
#         print(curr)
#         backtrack(index + 1, curr)

#     backtrack(0, [])
#     return res


# print(subsets([3,1,2]))   
 
# def subsets(nums, p):

#     def backtrack(index, curr):

#         if sum(curr) == p:
#             return curr

#         if index == len(nums):
#             return False

#         curr.append(nums[index])

#         if backtrack(index + 1, curr):
#             return True

#         curr.pop()

#         if backtrack(index + 1, curr):
#             return True

#         return False

#     print(backtrack(0, []))


# subsets([1,3], 5)

 
# def subsets(nums, p):
    
#     def backtrack(index, curr,count):
#         if sum(curr) == p:
#             return 1
        
#         if index == len(nums):
#             return 0

#         curr.append(nums[index])

#         left=backtrack(index + 1, curr,count)
            

#         curr.pop()

#         right=backtrack(index + 1, curr,count)
#         return right+left

#     print((backtrack(0, [],0)))
# subsets([1,2,3], 3)      



# def subsets(nums, p):

#     def backtrack(index, curr):

#         if sum(curr) == p:
#             return curr

#         if index == len(nums):
#             return False

#         curr.append(nums[index])

#         if backtrack(index + 1, curr):
#             return True

#         curr.pop()

#         if backtrack(index + 1, curr):
#             return True

#         return False

#     print(backtrack(0, []))


# subsets([1,3], 5)

# def subsets(nums):
#     res = []

#     def backtrack(index, curr,taken=True):

#         # base case
#         if index == len(nums):
#             res.append(curr[:])
#             return

#         # take
#         curr.append(nums[index])
#         taken=True
#         if taken:
            
#             backtrack(index + 1, curr,taken=False)

#         # don't take
#             curr.pop()
#             print(curr)

#     backtrack(0, [])
#     return res


# print(subsets([3,1,2])) 
 
 
 
# def subsets(nums):
#     res = []

#     def backtrack(index, curr, prev_taken):

#         # base case
#         if index == len(nums):
#             res.append(curr[:])
#             return

#         # don't take current element
#         backtrack(index + 1, curr, False)

#         take current element only if previous was not taken
#         if not prev_taken:
#             curr.append(nums[index])

#             backtrack(index + 1, curr, True)

#             curr.pop()

#     backtrack(0, [], False)

#     return res


# print(subsets([3, 1, 2])) 




######TREE######################################################################################################################################################
# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None


# def inorder(root):

#     if root is None:
#         return
#     inorder(root.left)
#     print(root.data)
#     inorder(root.right)

# def preorder(root):

#     if root is None:
#         return

#     print(root.data)

#     preorder(root.left)

#     preorder(root.right)
# def postorder(root):

#     if root is None:
#         return

#     postorder(root.left)

#     postorder(root.right)

#     print(root.data)    
# # creating tree
# root = Node(1)

# root.left = Node(2)
# root.right = Node(3)

# root.left.left = Node(4)
# root.left.right = Node(5)


# inorder(root)
# postorder(root)
# preorder(root)
# from collections import deque

# class Solution:
#     def levelOrder(self, root):

#         if not root:
#             return []

#         q = deque([root])
#         ans = []

#         while q:

#             size = len(q)
#             level = []

#             for _ in range(size):

#                 node = q.popleft()
#                 level.append(node.val)

#                 if node.left:
#                     q.append(node.left)

#                 if node.right:
#                     q.append(node.right)

#             ans.append(level)

#         return ans

# from collections import deque


# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None


# def levelorder(root):

#     if not root:
#         return []


#     q = deque([root])

#     ans=[]
#     left_to_right = True
#     while q:
#         size=len(q)
#         level=[]
#         for _ in range(size):
            
            
#             node=q.popleft()
#             level.append(node.data)
#             if node.left:
#                 q.append(node.left)
#             if node.right:
#                 q.append(node.right)
#         if not left_to_right:
            
#             level.reverse()
            
#         ans.append(level)
#         left_to_right=not left_to_right
            

#     return (ans)
# root = Node(1)

# root.left = Node(2)
# root.right = Node(3)

# root.left.left = Node(4)
# root.left.right = Node(5)

# print(levelorder(root))    



# from collections import deque

# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# def zigzagLevelOrder(root):
#     if not root:
#         return []

#     res = []
#     q = deque([root])
#     left_to_right = True

#     while q:
#         level = []
#         size = len(q)

#         for _ in range(size):
#             node = q.popleft()
#             level.append(node.val)

#             if node.left:
#                 q.append(node.left)
#             if node.right:
#                 q.append(node.right)

#         if not left_to_right:
#             level.reverse()

#         res.append(level)
#         left_to_right = not left_to_right

#     return res


# # Example tree
# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)
# root.right.left = TreeNode(6)
# root.right.right = TreeNode(7)

# print(zigzagLevelOrder(root))


# from collections import deque


# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None


# def levelOrder(root):

#     if not root:
#         return

#     q = deque([root])
#     ans = []
#     nodes = []

#     while q:

#         size = len(q)
#         level = []
#         nodes = []

#         for _ in range(size):

#             node = q.popleft()
#             level.append(node.data)

#             if node.left:
#                 q.append(node.left)
#                 nodes.append(node.left)

#             if node.right:
#                 q.append(node.right)
#                 nodes.append(node.right)

#         ans.append(level)

#     return ans


# # Tree creation
# root = Node(1)

# root.left = Node(2)
# root.right = Node(3)

# root.left.left = Node(4)
# root.left.right = Node(5)

# root.right.left = Node(6)
# root.right.right = Node(7)

# print
# print(levelOrder(root))
# print(levelOrder(root))




#sliding window revision###################33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

# arr = [5, 5, 5, 5]
# k = 2
# left=0
# listt=[]
# right=0
# sum=0
# best=0
# for right in range(len(arr)):
#     sum+=arr[right]
#     if  right-left+1==k:
#         best=max(best,sum)
#         sum-=arr[left]
#         left+=1
# print(best)    
        
        
        
#Maximum Sum of Distinct Subarrays With Length K (LeetCode 2461)            
# from typing import List
# from collections import defaultdict

# class Solution:
#     def maximumSubarraySum(self, nums: List[int], k: int) -> int:
#         left = 0
#         window_sum = 0
#         freq = defaultdict(int)
#         best = 0

#         for right in range(len(nums)):
#             # add current element
#             window_sum += nums[right]
#             freq[nums[right]] += 1

#             # shrink window if size exceeds k
#             if right - left + 1 > k:
#                 freq[nums[left]] -= 1
#                 window_sum -= nums[left]

#                 if freq[nums[left]] == 0:
#                     del freq[nums[left]]

#                 left += 1

#             # check condition: size == k AND all distinct
#             if right - left + 1 == k and len(freq) == k:
#                 best = max(best, window_sum)

#         return best      
    
    
# left = 0
# window_state = 0   # sum / frequency / count etc.
# best = 0           # or float('inf') depending on problem

# for right in range(len(arr)):
    
#     # 1. Expand window
#     window_state += arr[right]

#     # 2. Shrink window while condition is NOT valid
#     while condition_is_invalid:
#         window_state -= arr[left]
#         left += 1

#     # 3. Update answer when window is valid
#     best = max(best, window_state)    


#Smallest Subarray with a Given Sum (Easy)
# arr = [2, 1, 5, 2, 3, 2]
# S = 7
# left=0
# summ=0
# best=float('inf')
# for right in range(len(arr)):
#     summ+=arr[right]
#     while summ>=S:
#         best=min(best,right-left+1)
#         summ-=arr[left]
#         left+=1
# print(best)

#fruits in basket
# arr = [1, 2, 2,2,2,2,3]
# left=0
# freq={}
# best=0
# for right in range(len(arr)):
#     freq[arr[right]] = freq.get(arr[right], 0) + 1
#     while len(freq)>2:
#         freq[arr[left]] -= 1

#         if freq[arr[left]] == 0:
#             del freq[arr[left]]
#         left+=1    
        
#     best=max(best,right-left+1)    
# print(best)


#Subarray Product Less Than K
# nums = [1, 1, 1]
# k = 2
# prod=1
# summ=0
# left=0
# for right in range(len(nums)):
#     prod*=nums[right]
#     while prod>=k:
#         prod=prod/nums[left]
#         left+=1
#     summ+=right-left+1
# print(summ)  

# Important formula for counting

# When window is valid:

# right - left + 1

# 👉 This gives number of new subarrays ending at right  




# nums = [1,1,1,0,0,0,1,1,1,1,0]
# k = 2
# nums = [1,0,1,0,1,0,1]
# k = 3
# left=0
# zeros=0
# best=0
# for right in range(len(nums)):
#         if nums[right]==0:
#                 zeros+=1
#         while zeros>k:
#                 if nums[left]==0:
                        
#                         zeros-=1
#                 left+=1        
#         best=max(best,right-left+1)
# print(best)
                
        
    
# nums = [2, 1, 5, 1, 4, 2,7]
# k = 3
# left=0
# best=0
# summ=0
# for right in range(len(nums)):
#         summ+=nums[right]
#         while right-left+1>=k:
#                 best=max(best,summ)
#                 summ-=nums[left]
#                 left+=1
# print(best)


# nums=[1,1,0,1,1,1,1,1]
# left=0
# zeros=0
# best=0
# for right in range(len(nums)):
#         if nums[right]==0:
#                 zeros+=1
#         while zeros>=1:
#                 if nums[left]==0:
#                         zeros-=1
#                 left+=1
#         best=max(best,right-left+1)
# print(best)



#####################3prefix sum revision  #########################################################
# nums = [10, 4, -8, 7]
# left=0
# summ=sum(nums)
# count=0
# for i in range(len(nums)):
#     right=summ-nums[i]-left
#     if right>=left:
#         count+=1
#     left+=nums[i]
# print(count) 

# nums = [1,2,3,4] 
# k=[]
# pre=0
# for i in range(len(nums)):
#         pre+=nums[i]
#         k.append(pre)
# print(k)

# nums=[1,2,3]
# k=3
# pre = 0
# count = 0
# freq = {0: 1}

# for num in nums:
#         pre += num

#         if pre - k in freq:
#                 count += freq[pre - k]

#         freq[pre] = freq.get(pre, 0) + 1

# print(count)
# print(freq)
        
        
# nums=[1,2,3]
# k=3
# pre = 0
# count = 0
# freq = {0: 1}
# for i in nums:
#         pre+=i
#         present=pre-k
#         if present in freq:
#                 count+=freq[present]
#         freq[pre]=freq.get(pre,0)+1
# print(count)                
# print(freq)


# nums = [4,5,0,-2,-3,1]
# k = 5
# pre = 0
# count = 0
# freq = {0: 1}
# for i in nums:
#         pre+=i
#         present=pre%k
#         if present in freq:
#                 count+=freq[present]
#         freq[present]=freq.get(present,0)+1
# print(count)  
# print(freq)


# nums = [1,1]
# k = 6

# pre = 0
# count = 0
# freq = {0: 1}
# for i in nums:
#         pre+=i
#         val=pre%k
#         if val in freq:
#                 if i-freq[val]>1:
#                         print(True)
#                         break
                
#         else:
#                 freq[val]=i

# nums = [1,1,1,0,0,0,1]
# freq={0:-1}
# pre=0
# k=0
# for i in range(len(nums)):
#         if nums[i]==0:
#                 nums[i]=-1
#         pre+=nums[i]
#         if pre not in freq:
#                 freq[pre] = i
#         if pre in freq:
#                 k=max(k,i-freq[pre])
                      
# print(k)


# What should I store?
# If question says
# "Count"

# Store

# How many times this prefix has appeared.

# If question says
# "Longest"

# Store

# Where did this prefix appear first?

# Never overwrite it.

# 4. Initialization

# Whenever counting

# Remember

# Empty prefix exists once.

# Whenever longest

# Remember

# Empty prefix starts before index 0.

# 5. If question contains
# "Exactly K"

# Think

# Current Prefix − Previous Prefix = K

# So mentally think

# Subtract K

# 6. If question contains
# "Divisible by K"

# Think

# Same remainder

# Never think about exact sum.

# Always think

# Remainders

# 7. Equal number of 0s and 1s?

# Think

# Make

# 0 → -1

# Now equal 0s and 1s means

# Prefix sum becomes 0

# 8. Continuous Subarray Sum

# Remember only one thing

# Same remainder again

# AND

# Distance between indices ≥ 2



#binary search ###############################################################################################################################################################################
# nums = [-1,0,3,5,9,12]
# target = 9


# left = 0
# right = len(nums) - 1

# while left <= right:
#     mid = left + (right - left) // 2

#     if nums[mid] == target:
#         print(mid)

#     elif nums[mid] < target:
#         left = mid + 1

#     else:
#         right = mid - 1

# print(-1) 


# nums = [-1,0,3,5,9,12]
# target = 9


# left = 0
# right = len(nums) - 1
# while left<=right:
#         mid=left + (right - left) // 2

#         if nums[mid]==target:
#                 print(mid)
#                 break
#         elif target<nums[mid]:
#                 right=mid-1
#         else:
#                 left=mid+1



# nums = [1]
# target = 0
# left = 0
# right = len(nums) - 1
# while left<=right:
#         mid=left + (right - left) // 2

#         if nums[mid]==target:
#                 print(mid)
#                 break
#         elif target<nums[mid]:
#                 right=mid-1
#         elif target>nums[mid]:
#                 left=mid+1
        
# print(left)
# If the target is found → return its index.
# If the target is not found → left is the correct insertion position.




                
# nums = [4,5,6,7,0,1,2]
# target = 0

# left = 0
# right = len(nums) - 1

# while left <= right:
#     mid = left + (right - left) // 2

#     if target == nums[mid]:
#         print(mid)
#         break

#     # Left half is sorted
#     if nums[mid] >= nums[left]:
#         if nums[left] <= target < nums[mid]:
#             right = mid - 1
#         else:
#             left = mid + 1

#     # Right half is sorted
#     else:
#         if nums[mid] < target <= nums[right]:
#             left = mid + 1
#         else:
#             right = mid - 1
# else:
#     print(-1)
                
#first we need to find sorted array
# first we need to find the target in the sorted array array
#we need find both sorted and non sorted



# nums = [6,7,8,1,2,3,4]
# left=0
# right=len(nums)-1
# while left<=right:
#         mid = left + (right - left) // 2
    
#         if nums[mid]<nums[left]:
#                 left=mid
#         else:
#                 right=mid-1  
#         if nums[mid]>nums[right]:
#                 left=mid+1
#         else:
#                 right=mid-1
                
        
# print(nums[left])
        
        
        

# nums=[4,5,6,7,0,1,2]
  
# left=0
# right=len(nums)-1
# while left<right:
#         mid = left + (right - left) // 2
    
#         if nums[mid]>nums[right]:
#                 left=mid+1
                

#         else:
#                 right=mid
                
        
# print(nums[left])    

#no need to comapre both left and right ,one of them is sufficinet to solve the problem ,either compare mid with low or high 
#sometime right=mid occurs  
#sometimes it could be left<right:    
#nums = [6,7,0,1,2,3,4]
#nums = [4,5,6,7,0,1,2] examples to solve 


# nums = [5,4,3,2,1]

# left=0
# right=len(nums)-1 
# while left<=right:
#         mid=left+(right-left)//2
#         if nums[mid-1]<nums[mid]>nums[mid+1]:
#                 print(nums[mid])
#                 break
#         if nums[mid]<nums[mid+1]:
#                 left=mid+1
#         else:
#                 right=mid
            
# When you are not 100% sure that mid cannot be the answer,

# ➡️ don't discard mid. and use like right=mid           
     
# Golden Rule ⭐

# Don't memorize left = mid + 1.

# Instead memorize:

# If mid cannot be the answer → discard it (mid ± 1).

# If mid could still be the answer → keep it (mid).            
# example : right=mid or left=mid
#sometimes you need to search with adjecents of mid        
            
            
# nums = [1,4,7,3,7,3,2,7,2]

# left=0
# right=len(nums)-1 
# while left<right:
#         mid=left+(right-left)//2
        
#         if nums[mid]>nums[mid+1]:
#                 right=mid
#         else:
#                 left=mid+1
# print(nums[left] )



                                
# piles = [3, 6, 7, 11]
# h = 8

# low = 1
# high = max(piles)

# ans = 0

# while low <= high:
#     mid = (low + high) // 2
#     total_hours=0
#     for i in piles:
#             if i%mid==0:
#                     total_hours+=(i//mid)
#             else:
#                 total_hours+=(i//mid)+1
#     if total_hours<=h:
#             ans=mid
#             high=mid-1
#     else:
#             low=mid+1   
# print(ans)    




# weights = [1,2,3,4,5,6,7,8,9,10]
# days = 5

# low = max(weights)
# high = sum(weights)

# ans = 0

# while low <= high:
#     mid = (low + high) // 2
#     total_sum=0
#     day=1
#     for i in weights:
#             total_sum+=i
            
#             if total_sum>mid:
#                     day+=1
#                     total_sum=i
#     if day<=days:
#             ans=mid
#             high=mid-1
#     else:
#             low=mid+1
            
# print(ans)
                    
                    
dist = [1,3,2]
hour = 2.7

left = 1
right = 10**7
ans = 0                    
while left<=right:
        mid=left+(right-left)//2
        summ=0
        for i in range(len(dist)):
                if i!=len(dist)-1:
                        if dist[i]%mid!=0:
                                summ+=(dist[i]//mid)+1
                        else:
                                
                                summ+=(dist[i]//mid)
                else:
                        summ+=(dist[i]/mid)
        if summ<=hour:
                ans=mid
                right=mid-1
        else:
                left=mid+1
print(ans)