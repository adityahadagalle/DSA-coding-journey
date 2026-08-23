
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()

        result_sum = nums[0] + nums[1] + nums[2]
        max_diff = abs(result_sum - target)

        for i in range(len(nums) - 2):

            left = i + 1
            right = len(nums) - 1

            while left < right:

                sum = nums[i] + nums[left] + nums[right]

                diff = abs(sum - target)

                if diff < max_diff:
                    max_diff = diff
                    result_sum = sum

                if sum < target:
                    left += 1

                elif sum > target:
                    right -= 1

                else:
                    return sum

        return result_sum