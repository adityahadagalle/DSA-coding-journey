class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        prev = None
        first = None
        mid = None
        last = None

        def find(node):
            nonlocal prev, first, mid, last

            if node is None:
                return

            find(node.left)

            if prev is not None and prev.val > node.val:
                if not first:
                    first = prev
                    mid = node
                else:
                    last = node

            prev = node

            find(node.right)

        find(root)

        if first and last:
            first.val, last.val = last.val, first.val
        elif first and mid:
            first.val, mid.val = mid.val, first.val