class Solution:
    def getAllElements(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> List[int]:
        list1 = []
        list2 = []

        def inorder(root, arr):
            if root is None:
                return
            inorder(root.left, arr)
            arr.append(root.val)
            inorder(root.right, arr)

        inorder(root1, list1)
        inorder(root2, list2)

        i = 0
        j = 0
        result = []

        while i < len(list1) and j < len(list2):
            if list1[i] <= list2[j]:
                result.append(list1[i])
                i += 1
            else:
                result.append(list2[j])
                j += 1

        result.extend(list1[i:])
        result.extend(list2[j:])

        return result