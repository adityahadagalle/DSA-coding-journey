class Solution:
    def maximumSubarraySum(self, nums, k):

        left = 0
        summ = 0
        best = 0
        freq = {}

        for right in range(len(nums)):

            summ += nums[right]
            freq[nums[right]] = freq.get(nums[right], 0) + 1

            while right - left + 1 > k:

                summ -= nums[left]
                freq[nums[left]] -= 1

                if freq[nums[left]] == 0:
                    del freq[nums[left]]

                left += 1

            if right - left + 1 == k and len(freq) == k:
                best = max(best, summ)

        return best
        