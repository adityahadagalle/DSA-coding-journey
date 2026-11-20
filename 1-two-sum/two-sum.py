from typing import List


class Solution:

  def twoSum(self, nums: List[int], target: int) -> List[int]:
    # Store each number with its original 0-based index
    indexed_nums = [(num, idx) for idx, num in enumerate(nums)]

    # Sort by the number value
    indexed_nums.sort()

    i = 0
    j = len(indexed_nums) - 1

    while i < j:
      current_sum = indexed_nums[i][0] + indexed_nums[j][0]

      if current_sum == target:
        return [indexed_nums[i][1], indexed_nums[j][1]]
      elif current_sum < target:
        i += 1
      else:
        j -= 1

    return []

