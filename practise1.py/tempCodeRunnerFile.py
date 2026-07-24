#             ans.append(path[:])
#             return
#     for end in range(start, min(start+3,len(s))):
#         peace = s[start:end + 1]
#         if int(peace)<=255:
#             continue
        
            
#         path.append(peace)
#         backtrack(end+3)
#         path.pop()
# backtrack(0)
# print(ans)            
            
        

# path=[]
# ans=[]   

# def backtrack(start,s,k):
#     if start == len(s):
#         ans.append("".join(path))
#         return
#     for i in range(len(k)):
#         a=s[i]+k[start]
#         path.append(a)
#         backtrack(start+1,s,k)
#         path.pop()
        
        
        


            
# backtrack(0,"abc","def")
# print(ans)        


path = []
ans = []

def backtrack(start, s, k):
    if start == len(s):
        return

    first = s[start]
