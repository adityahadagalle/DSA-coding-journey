class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        freq = {0: -1}
        pre = 0

        for i in range(len(nums)):
            pre += nums[i]
            rem = pre % k

            if rem in freq:
                if i - freq[rem] >= 2:
                    return True
            else:
                freq[rem] = i

        return False
        