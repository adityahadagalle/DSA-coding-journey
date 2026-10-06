class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def find(root, high, low):
            if root is None:
                return True

            if root.val >= high or root.val <= low:
                return False

            lef = find(root.left, root.val, low)
            rig = find(root.right, high, root.val)

            if lef == False or rig == False:
                return False

            return True

        return find(root, float('inf'), float('-inf'))

