class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        summ = 0
        left = 0
        max_avg = float("-inf")

        for right in range(len(nums)):
            summ += nums[right]

            while right - left + 1 == k:
                avg = summ / k
                max_avg = max(avg, max_avg)

                summ -= nums[left]
                left += 1

        return max_avg