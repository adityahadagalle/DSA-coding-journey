class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:

        def fun(l, r):
            if l > r:
                return None

            mid = (l + r) // 2
            root = TreeNode(nums[mid])

            root.left = fun(l, mid - 1)
            root.right = fun(mid + 1, r)

            return root

        return fun(0, len(nums) - 1)