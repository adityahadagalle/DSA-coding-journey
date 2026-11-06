from typing import List

class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        count = 0
        pre = 0
        freq = {0: 1}
        
        for i in range(len(nums)):
            pre += nums[i]
            check = pre % k
            
            if check in freq:
                count += freq[check]
                
            freq[check] = freq.get(check, 0) + 1
            
        return count
