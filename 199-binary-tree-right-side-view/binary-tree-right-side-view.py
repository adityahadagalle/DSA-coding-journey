# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
from collections import deque

class Solution:
    def rightSideView(self, root):

        if root is None:
            return []

        queue = deque([root])
        ans = []

        while queue:

            level = []

            size = len(queue)

            for _ in range(size):

                node = queue.popleft()

                level.append(node.val)

                if node.left:
                    queue.append(node.left)

                if node.right:
                    queue.append(node.right)

            ans.append(level[-1])

        return ans
        