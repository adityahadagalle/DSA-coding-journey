class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:

        curr = 0
        freq = {0: 1}
        count = 0

        for i in range(len(nums)):

            curr += nums[i]

            now = curr - k

            if now in freq:
                count += freq[now]

            freq[curr] = freq.get(curr, 0) + 1

        return count
        