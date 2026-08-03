from collections import deque

class Solution:
    def isCousins(self, root, x, y):

        queue = deque([root])

        while queue:

            level = []
            size = len(queue)

            for _ in range(size):

                node = queue.popleft()

                level.append(node.val)

                if node.left and node.right:

                    if node.left.val == x and node.right.val == y:
                        return False

                    if node.left.val == y and node.right.val == x:
                        return False

                if node.left:
                    queue.append(node.left)

                if node.right:
                    queue.append(node.right)

            if x in level and y in level:
                return True

            if (x in level) != (y in level):
                return False

        return False
        