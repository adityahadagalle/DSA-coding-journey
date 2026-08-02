class Solution:
    def preorderTraversal(self, root):

        ans = []

        def preorder(root):

            if root is None:
                return

            ans.append(root.val)

            preorder(root.left)

            preorder(root.right)

        preorder(root)

        return ans
        