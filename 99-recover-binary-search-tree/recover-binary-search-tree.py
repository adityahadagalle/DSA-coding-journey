class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        prev = None
        first = None
        mid = None
        last = None
        count = 0

        def find(root):
            nonlocal prev, first, mid, last, count

            if root is None:
                return

            find(root.left)

            if prev is not None and prev.val > root.val:
                if count == 0:
                    first = prev
                    mid = root
                    count += 1
                if count == 1:
                    last = root

            prev = root

            find(root.right)

        find(root)

        if first and last:
            first.val, last.val = last.val, first.val
        elif first and mid:
            first.val, mid.val = mid.val, first.val