import math
import heapq
from typing import List

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:

        check = 0
        result = []
        heap = []

        def fun(point):
            output = math.dist(point, [0, 0])
            return output

        for i in range(len(points)):
            if check < k:
                count = fun(points[i])
                heapq.heappush(heap, (-count, points[i]))
                check += 1
                continue

            count = fun(points[i])
            heapq.heappush(heap, (-count, points[i]))
            heapq.heappop(heap)

        while heap:
            result.append(heapq.heappop(heap)[1])

        return result