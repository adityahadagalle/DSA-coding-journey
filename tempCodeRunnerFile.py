class ListNode:
#     def __init__(self, val):
#         self.val = val
#         self.next = None


# a = ListNode(1)
# b = ListNode(2)
# c = ListNode(3)
# d = ListNode(3)

# a.next = b
# b.next = c
# c.next = d
# d.next = b  

# head = 0                                 #fast.next is not None:  this is to check is there any cycle or not


# slow = head
# fast = head

# while fast is not None and fast.next is not None:
#     slow = slow.next
#     fast = fast.next.next

#     if slow == fast:
#         slow = head
        
#         while slow != fast:
#             slow = slow.next
#             fast = fast.next
        
#         print(slow.val)  
#         break
    