from collections import deque

class Solution:
    def largestValues(self, root):
        if not root:
            return []

        queue = deque([root])
        res = []

        while queue:
            best = float('-inf')

            for _ in range(len(queue)):
                node = queue.popleft()

                if node.left:
                    queue.append(node.left)

                if node.right:
                    queue.append(node.right)

                best = max(best, node.val)

            res.append(best)

        return res