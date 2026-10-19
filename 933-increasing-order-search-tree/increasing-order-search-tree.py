
class Solution:
    def increasingBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        arr = []

        def find(node):
            if node is None:
                return

            find(node.left)
            arr.append(node.val)
            find(node.right)

        find(root)

        new_root = TreeNode(arr[0])
        curr = new_root

        for i in range(1, len(arr)):
            curr.right = TreeNode(arr[i])
            curr = curr.right

        return new_root
