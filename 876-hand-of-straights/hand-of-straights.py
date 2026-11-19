from collections import Counter
import heapq
from typing import List

class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False

        freq = Counter(hand)

        heap = list(freq.keys())
        heapq.heapify(heap)

        while heap:
            start = heap[0]

            for num in range(start, start + groupSize):

                if freq[num] == 0:
                    return False

                freq[num] -= 1

                if freq[num] == 0:

                    if num != heap[0]:
                        return False

                    heapq.heappop(heap)

        return True