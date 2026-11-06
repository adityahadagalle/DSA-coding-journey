class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        count = 0
        pre = 0
        freq = {0: 1}

        for i in range(len(nums)):
            pre += nums[i]
            check = pre - goal

            if check in freq:
                count += freq[check]

            freq[pre] = freq.get(pre, 0) + 1

        return count