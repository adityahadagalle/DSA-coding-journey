import heapq
from collections import Counter

class Solution:
    def frequencySort(self, s: str) -> str:

        result = ''
        freq = Counter(s)
        heap = []

        check = 0

        for word, count in freq.items():
            heapq.heappush(heap, (count, word))

        while heap:
            count, char = heapq.heappop(heap)
            for i in range(count):
                result += char

        result = result[::-1]

        return result