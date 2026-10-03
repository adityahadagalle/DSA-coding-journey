class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        org = root

        if root is None:
            return TreeNode(val)

        while root:
            if root.val > val:
                if root.left is None:
                    root.left = TreeNode(val)
                    break
                else:
                    root = root.left

            elif root.val < val:
                if root.right is None:
                    root.right = TreeNode(val)
                    break
                else:
                    root = root.right

        return org