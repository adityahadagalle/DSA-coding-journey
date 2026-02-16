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



intervals = [[1,10],[2,3],[4,5],[6,7]]

intervals.sort(key=lambda x:x[1])
count=1
end1=intervals[0][1]
for i in range(1,len(intervals)):
    st2=intervals[i][0]
    end2=intervals[i][1]
    if end1<=st2:
        count+=1
    
        end1=end2
print(count)        