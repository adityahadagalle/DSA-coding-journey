class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:

        left = 0
        summ = 0
        best = float('inf')

        for right in range(len(nums)):

            summ += nums[right]

            while summ >= target:

                best = min(best, right - left + 1)

                summ -= nums[left]
                left += 1

        if best == float('inf'):
            return 0

        return best