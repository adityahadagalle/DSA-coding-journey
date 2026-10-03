class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        res = []

        def inorder(root):
            if root is None:
                return

            inorder(root.left)
            res.append(root.val)
            inorder(root.right)

        inorder(root)

        return res[k - 1]