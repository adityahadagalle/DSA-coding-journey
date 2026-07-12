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
        
        
    


nums = [2,1,2]
k = 1
for i in range(len(nums)):
    pre+=
       