piles = [3, 6, 7, 11]
h = 8

low = 1
high = max(piles)

ans = 0

while low <= high:
    mid = (low + high) // 2
    total_hours=0
    for i in piles:
        if i%mid==0:
            total_hours+=i//mid
        else:
            total_hours+=(i//mid)+1
    if total_hours<=h:
        ans=mid
        high=mid-1
    else:
        low=mid+1
print(ans)
print(ans)