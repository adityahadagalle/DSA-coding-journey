from collections import Counter
import heapq
from typing import List

class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:

        freq = Counter(hand)

        heap = [(number, count) for number, count in freq.items()]
        heapq.heapify(heap)

        ress = []

        while heap:

            temp = []

            for _ in range(groupSize):

                if not heap:
                    return False

                number, count = heapq.heappop(heap)
                count -= 1
                if not ress:
                    ress.append(number)

                elif ress[-1] + 1 == number:
                    ress.append(number)

                else:
                    return False

            

                if count > 0:
                    temp.append((number, count))

            for item in temp:
                heapq.heappush(heap, item)

            ress = []

        return True