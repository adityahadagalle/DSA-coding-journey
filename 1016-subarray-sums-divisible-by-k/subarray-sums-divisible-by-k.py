class Solution:
    def subarraysDivByK(self, nums, k):
        dictt={0:1}
        pre=0
        count=0
        for i in range(len(nums)):
            pre+=nums[i]
            val=pre%k
            if val in dictt:
                count+=dictt[val]
            
            dictt[val]=dictt.get(val,0)+1  
                
        return count