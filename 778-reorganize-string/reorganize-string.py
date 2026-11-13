import heapq
from collections import Counter

class Solution:
    def reorganizeString(self, s: str) -> str:
        freq = Counter(s)
        heap = [(-count, char) for char, count in freq.items()]
        heapq.heapify(heap)

        result = []

        while heap:
            count, char = heapq.heappop(heap)

            if result and result[-1] == char:
                if not heap:
                    return ""

                count2, char2 = heapq.heappop(heap)
                result.append(char2)
                count2 += 1

                if count2 < 0:
                    heapq.heappush(heap, (count2, char2))

                heapq.heappush(heap, (count, char))

            else:
                result.append(char)
                count += 1

                if count < 0:
                    heapq.heappush(heap, (count, char))

        return "".join(result)