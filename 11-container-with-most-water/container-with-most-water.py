class Solution:
    def maxArea(self, height: List[int]) -> int:
        l = 0
        best = 0
        r = len(height) - 1

        while l < r:
            width = r - l
            h = min(height[l], height[r])
            mul = width * h

            best = max(best, mul)

            if height[l] < height[r]:
                l += 1
            else:
                r -= 1

        return best