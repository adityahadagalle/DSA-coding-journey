import heapq
from typing import List

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:

        result = []
        heap = []
        check = 0

        for i in range(len(arr)):
            if check < k:
                dist = abs(arr[i] - x)

                heapq.heappush(heap, (-dist, arr[i]))
                check += 1
                continue

            dist = abs(arr[i] - x)

            if dist == -(heap[0][0]):

                if arr[i] < heap[0][1]:
                    heapq.heappush(heap, (-dist, arr[i]))
                else:
                    continue

            heapq.heappush(heap, (-dist, arr[i]))
            heapq.heappop(heap)

        while heap:
            result.append(heapq.heappop(heap)[1])

        result.sort()

        return result