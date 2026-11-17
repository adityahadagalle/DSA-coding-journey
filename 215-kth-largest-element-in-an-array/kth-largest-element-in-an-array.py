import heapq
from typing import List

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []

        for i in range(len(nums)):
            if i < k:
                heapq.heappush(heap, nums[i])
                continue

            heapq.heappush(heap, nums[i])
            heapq.heappop(heap)

        return heap[0]