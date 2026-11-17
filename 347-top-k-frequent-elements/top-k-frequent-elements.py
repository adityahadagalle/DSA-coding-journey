import heapq
from collections import Counter
from typing import List

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        heap = []
        freq = Counter(nums)

        for number, count in freq.items():
            if len(heap) < k:
                heapq.heappush(heap, (count, number))
                continue

            heapq.heappush(heap, (count, number))
            heapq.heappop(heap)

        res = []

        while heap:
            res.append(heapq.heappop(heap)[1])

        return res