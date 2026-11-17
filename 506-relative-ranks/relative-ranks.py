import heapq
from typing import List

class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        heap = []
        count = 1

        for index, value in enumerate(score):
            heapq.heappush(heap, (-value, index))

        res = [""] * len(score)

        while heap:
            val, idx = heapq.heappop(heap)

            if count == 1:
                res[idx] = "Gold Medal"
            elif count == 2:
                res[idx] = "Silver Medal"
            elif count == 3:
                res[idx] = "Bronze Medal"
            else:
                res[idx] = str(count)

            count += 1

        return res