class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        best = 0
        pre = 0
        freq = {0: -1}

        for i in range(len(nums)):
            if nums[i] == 0:
                nums[i] = -1

            pre += nums[i]

            if pre in freq:
                best = max(best, i - freq[pre])

            else:
                freq[pre] = i

        return best
        