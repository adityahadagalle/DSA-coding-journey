class Solution:
    def bstToGst(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        count = 0

        def find(root):
            nonlocal count

            if root is None:
                return

            find(root.right)

            count += root.val
            root.val = count

            find(root.left)

        find(root)
        return root