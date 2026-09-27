class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        n = len(arr)
        destination = [0] * (2 * n)
        d = 0

        for s in range(n):
            if arr[s] == 0:
                destination[d] = 0
                d += 1
                destination[d] = 0
            else:
                destination[d] = arr[s]

            d += 1

        for i in range(n):
            arr[i] = destination[i]