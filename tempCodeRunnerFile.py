
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