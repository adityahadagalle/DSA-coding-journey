def fun(n):
    if n == 1:
        print(1)
        return
    
    fun(n-1)
    print(n)

print(fun(6)) 