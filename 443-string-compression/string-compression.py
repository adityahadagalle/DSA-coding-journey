class Solution:
    def compress(self, chars: List[str]) -> int:
        i = 0
        count = 1

        for right in range(1, len(chars) + 1):

            if right == len(chars) or chars[right] != chars[right - 1]:

                chars[i] = chars[right - 1]
                i += 1

                if count > 1:
                    for digit in str(count):
                        chars[i] = digit
                        i += 1

                count = 1

            else:
                count += 1

        return i