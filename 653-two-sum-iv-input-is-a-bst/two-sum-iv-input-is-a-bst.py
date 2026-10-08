class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        succ = []

        def find(root):
            if root is None:
                return

            find(root.left)
            succ.append(root.val)
            find(root.right)

        find(root)

        i = 0
        j = len(succ) - 1

        while i < j:
            if succ[i] + succ[j] == k:
                return True
            elif succ[i] + succ[j] < k:
                i += 1
            else:
                j -= 1

        return False