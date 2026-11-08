import heapq
from collections import Counter
from typing import List

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:

        result = []
        freq = Counter(nums)
        heap = []
        check = 0

        for number, count in freq.items():
            if check < k:
                heapq.heappush(heap, (count, number))
                check += 1
                continue

            if heap[0][0] < count:
                heapq.heappop(heap)
                heapq.heappush(heap, (count, number))

        while heap:
            result.append(heapq.heappop(heap)[1])

        return result