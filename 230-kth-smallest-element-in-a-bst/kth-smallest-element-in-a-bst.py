class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        count = 0

        def inorder(root):
            nonlocal count

            if root is None:
                return None

            result = inorder(root.left)

            if result is not None:
                return result

            count += 1

            if count == k:
                return root.val

            return inorder(root.right)

        return inorder(root)