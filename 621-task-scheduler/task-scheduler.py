import heapq
from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = Counter(tasks)

        heap = [(-count, task) for task, count in freq.items()]
        heapq.heapify(heap)

        next_available = {task: 1 for task in freq}

        time = 0

        while heap:
            time += 1

            temp = []

            while heap:
                count, task = heapq.heappop(heap)

                if time >= next_available[task]:
                    count += 1

                    next_available[task] = time + n + 1

                    if count < 0:
                        heapq.heappush(heap, (count, task))

                    break

                temp.append((count, task))

            for item in temp:
                heapq.heappush(heap, item)

        return time