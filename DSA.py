######	Next greater number to given number with the same set of digits

def swap(number,i,j):
    temp=number[i]
    number[i]=number[j]
    number[j]=temp
# Given number as int array, this function finds the  
# greatest number and returns the number as integer
def findNextNumber(number,n): 
    #Practise Yourself :  Write your code Here 
    i=n-1
    min_p=n-1
    
    while i > 0:
        if number[i]>number[i-1]:
            min_p=i-1
            break
        i -=1
    
    print(min_p+1,n)
    
    min_n=number[min_p+1]
    min_np=min_p+1
    
    for i in range(min_p+1,n):
        if number[i]<=min_n:
            if number[i]>=number[min_p]:
                min_n=number[i]
                min_np=i
    
    print(min_p,min_n,min_np)
    
    swap(number,min_p,min_np)
    
    
    
    print(number[:min_p+1] + sorted(number[min_p+1:]))
    

    

digits = "218765"         
  
number = list(map(int ,digits)) 
print(number)
findNextNumber(number, len(digits))






######	Stock Buy -Sell Problem in Linear time

def findProfit(array, n): 
    #Practise Yourself :  Write your code Here 
    i=0
    l_min=[]
    l_max=[]
    
    for i in range(n):
        if i==0:
            if array[i]<array[i+1]:
                l_min.append(i)
                continue
        if array[i]<array[i+1] and array[i]<array[i-1]:
                l_min.append(i)
    
    for i in range(1,n+1):
        if i == n:
            if array[i]>array[i-1]:
                l_max.append(i)
        elif array[i]>array[i+1] and array[i]>array[i-1]:
                l_max.append(i)
    
    profit =0 
    
    if len(l_min)==len(l_max) and l_min!=0:
        for i in range(len(l_min)):
            profit+=(array[l_max[i]]-array[l_min[i]])
    
    return profit
          
if __name__ == '__main__':
    price = [98, 178, 250, 300, 40, 540, 690];
    n = len(price);
 
    print(findProfit(price, n - 1));

######	Remove duplicate char in string in O(n)

def removeDuplicatesFromString(str2): 
  
    #Write your Code here
    ascii_arr=[0]*256

    for i in range(len(str2)):
        if ascii_arr[ord(str2[i])] == 0:
            ascii_arr[ord(str2[i])] = -1
        elif ascii_arr[ord(str2[i])] == -1:
            str2=str2[:i]+' '+str2[i+1:]
            
    return str2.replace(' ','')

######	Rotate Matrix by 90 degree clockwise – O(1) space

def matrixRotation(matrix): 
    #Practise Yourself :  Write your code Here 
    
    m = len(matrix)-1
    
    for j in range(int(m/2)):
        for i in range(m):
            a = matrix[0+j][i+j]
            b = matrix[i+j][m-j]
            c = matrix[m-j][m-i-j]
            d = matrix[m-i-j][0+j]
            
            #print(a,b,c,d)
            matrix[i+j][m-j],matrix[m-j][m-i-j],matrix[m-i-j][0+j],matrix[0+j][i+j]=a,b,c,d
            
    print(matrix)



######	Element search in sorted Matrix

def searchElement(matrix, n, x): 

    #Write your Code here 
    i=0
    j=n-1
    
    while i<n and j>=0:
        if matrix[i][j] == x:
            return True
        elif matrix[i][j] < x:
            i+=1
        elif matrix[i][j] > x:
            j-=1
    
    return False










######	Next Smallest Palindrome number 

def All9check(array):
    for i in array:
        if i !=9:
            return False
    return True

def findNextPalindrome(array, n) :
    #Practise Yourself :  Write your code Here
    mid=int(n/2)
    
    if All9check(array):
        return [1]+ (n-1)*[0] + [1]
    
    if n%2 == 0:
        
        for k in range(mid):
            if array[mid-1-k] != array[mid+k]:
                break
        
        if array[mid-1-k] > array[mid+k]:
            for i in range(mid):
                array[mid+i] = array[mid-1-i]
                
        elif array[mid-1-k] < array[mid+k]:
            if array[mid-1] != 9:
                array[mid-1] += 1
                for i in range(mid):
                    array[mid+i] = array[mid-1-i]
            
            else:
                #array[mid-1] = 0
                carry = 1
                for i in range(mid-1):
                    if carry ==1:
                        array[mid-1-i] += carry
                        if array[mid-1-i]>9:
                            array[mid-1-i] =0 
                            carry = 1
                        else:
                            carry = 0
                    
                    array[mid+i] = array[mid-1-i]
                
                if carry == 1:
                    array[0] += carry
                    
                if array[0] <=9: 
                    array[n-1] = array[0]%10
                else:
                    array[n-1] = array[0]/10

        
    # not single digit
    else:
        for k in range(1,mid+1):
            if array[mid-k] != array[mid+k]:
                break
        
        if array[mid-k] > array[mid+k]:
            for i in range(1,mid+1):
                array[mid+i] = array[mid-i]
                
        elif array[mid-k] < array[mid+k]:
            if array[mid] != 9:
                array[mid] += 1
                for i in range(1,mid+1):
                    array[mid+i] = array[mid-i]

            else:
                array[mid] = 0
                carry = 1
                for i in range(1,mid):
                    if carry ==1:
                        array[mid-i] += carry
                        if array[mid-i]>9:
                            array[mid-i] =0 
                            carry = 1
                        else: 
                            carry = 0
                    
                    array[mid+i] = array[mid-i]
                
                if carry == 1:
                    array[0] += carry
                
                if array[0] <=9: 
                    array[n-1] = array[0]%10
                else:
                    array[n-1] = array[0]/10
                            
    return array            
    
# The function that prints next palindrome of a given number array[] with n digits.		
def generateNextPalindrome(array, n ) : 

	#Practise Yourself :  Write your code Here
    
    print(findNextPalindrome(array, n))

if __name__ == "__main__": 
	array = [4,5,3,5,5]
	n = len(array) 
	generateNextPalindrome( array, n )
######	Max Sum Sub-array : Kadane algorithm

def maxSubArraySum(array,size): 
    #Write your code Here
    resultantSum = 0
    interSum = 0
    
    for i in range(size):
        interSum += array[i]
        
        if array[i] > interSum:
            interSum = array[i]
        
        if interSum > resultantSum:
            resultantSum = interSum
        
    return resultantSum


######	Celebrity Part Problem -> Relationship matrix

# Returns id of celebrity 
def findCelebrity(n): 
    #Practise Yourself :  Write your code Here
    
    X=0;
    Y=n-1
    
    while X<Y:
        if knows(X,Y) == 1:
            X+=1
        else:
            Y-=1
    
    celeb_checkX=1
    celeb_checkY=1
    
    for i in range(n):
        if knows(X,i) == 1:
            celeb_checkX = 0
            break
    
    for i in range(n):
        if X != i :
            if knows(i,X) == 0 :
                celeb_checkY = 0;
                break
    
    if (celeb_checkX and celeb_checkY):
        return X


######	Quick Select Method – find kth smallest element in array
def pivotSet(array,low,high):
    pivot = array[high]
    j = low
    
    for i in range(low,high+1):
        if array[i] <= pivot:
            array[i] , array[j] = array[j] , array[i]
            j+=1
    
    return j-1
    

# finds the kth position   
# in a given unsorted array i.e this function  
# can be used to find both kth largest and  
# kth smallest element in the array.  
def kthSmallest(array, low, high, k): 
    #Practise Yourself :  Write your code Here 
    if k> 0 and k <= len(array):
        pivot_index = pivotSet(array,low,high) 
        
        if pivot_index == k -1 :
            return array[pivot_index]
        elif pivot_index > k -1 :
            return kthSmallest(array, low, pivot_index - 1, k)
        else:
            return kthSmallest(array, pivot_index + 1, high, k)
    return -1


######	Square Root using Binary Search

def binaryS_root(low,high,Num):
    mid = (low+high)/2

    if Num ==0 or Num==1:
        return Num
    
    if Num > 0:        
        if Num-0.01<=mid*mid<=Num+0.01 :
            return int(mid)
        #if mid*mid == Num :
        #    return mid
        elif mid*mid > Num:
            return binaryS_root(low,mid,Num)
        elif mid*mid < Num:
            return binaryS_root(mid,high,Num)
        
def Sqrt(Num) : 
    #Practise Yourself :  Write your code Here 
    return binaryS_root(0,Num,Num)
######	2D array of sorted 0-1. Find row with max 1s – use binary search
def binarySearch( arr, low, high): 
    #Write your code here
    if high>low:
        mid = int((low+high)/2)
        
        if arr[mid] == 1 and (arr[mid-1] == 0 or mid == 0):
            return mid
        elif arr[mid] == 0:
            return binarySearch(arr, mid +1, high)
        else:
            return binarySearch(arr,low,mid-1)
    return -1
    
def findRow( matrix): 
    #Write your code here 
    row_count=len(matrix)
    row=0
    for i in range(len(matrix)):
        index = binarySearch(matrix[i],0,len(matrix)-1)
        if  index < row_count:
            row_count = index
            row=i
    return row

######	Trapping Rain Problem
def findWater(array, n): 
    #Practise Yourself :  Write your code Here 
    l = [0] * n
    r = [0] * n
    result = [0] * n
     
    l_max = array[0]
    for i in range(n):
        if array[i] > l_max:
            l_max = array[i]
        l[i]=l_max
     
    r_max = array[n-1]
    for i in range(n-1,0,-1):
        if array[i] > r_max:
            r_max = array[i]
        r[i]=r_max
        
    for i in range(n):
        if min(l[i],r[i]) > array[i] :
            result[i] = min(l[i],r[i]) - array[i]
    
    sum_i = 0
    for i in range(n):
        sum_i += result[i]
    return sum_i
######	Find Max (j-i) such that arr[j]>arr[i]
def findDiff(array):
    #Write your code here
    n= len(array)
    r=[0]*n
    l=[0]*n
    
    r_max = array[n-1]
    r_max_index = n-1
    for i in range(n-1,-1,-1):
        if array[i]>=r_max:
            r_max = array[i]
            r_max_index = i
        r[i] = r_max_index
    
    l_min = array[0]
    l_min_index = 0
    for i in range(n):
        if array[i] <= l_min:
            l_min = array[i]
            l_min_index = i
        l[i] = l_min_index
        
    print(l,r,array)
    
    i=0
    j=0
    max_diff = 0
    while (i<n and j<n):
        if array[r[j]] > array[l[i]]:
            max_diff = r[j] - l[i]
            j+=1
        else:
            i+=1
        
    return max_diff


######	Inversion counts to sort an array – Merge Sort Application -> O(nlogn)
def mergeOp(array,temp,low,mid,high):
    
    i= low
    j= mid+1
    k= low
    inv_count = 0
    
    while i<=mid and j<= high:
        if array[i] <= array[j]:
            temp[k] = array[i]
            i+=1
            k+=1
        else:
            temp[k] = array[j]
            inv_count+= mid -i +1 #imp.
            j+=1
            k+=1
    
    while i<=mid:
        temp[k] = array[i]
        k+=1
        i+=1
    
    while j<=high:
        temp[k] = array[j]
        j+=1
        k+=1
    
    for p in range(low,high+1):
        array[p] = temp[p]
        
    return inv_count
    

def mergeSort(array,temp,low,high):
    
    inv_count = 0
    
    if low < high:
        
        mid = int((low+high)/2)
        
        inv_count += mergeSort(array, temp , low ,mid)
        inv_count += mergeSort(array, temp , mid+1 ,high)
        inv_count += mergeOp(array, temp, low, mid, high)
        
    return inv_count        


def mergeSortInversion(array, n): 
    #Practise Yourself :  Write your code Here 
    temp = [0] * n
    print(mergeSort(array, temp, 0, n-1))












######	Min window Substring – 1

def findMinWindow(string, pat):  
    #Practise Yourself :  Write your code Here 
    string_ascii = [0] * 256
    pat_ascii = [0] * 256
    
    for i in range(len(pat)):
        pat_ascii[ord(pat[i])] += 1
    
    count = 0
    window_len = len(string)
    start_index=-1
    start = 0
    i=0
    for i in range(len(string)):
        
        string_ascii[ord(string[i])] += 1
        
        if pat_ascii[ord(string[i])] == 1 and string_ascii[ord(string[i])] <= pat_ascii[ord(string[i])] :
            count +=1
            
        if count == len(pat):
            while string_ascii[ord(string[start])] > pat_ascii[ord(string[start])] or pat_ascii[ord(string[start])] == 0:
                
                if string_ascii[ord(string[start])] > pat_ascii[ord(string[start])] :
                    string_ascii[ord(string[start])] -= 1
                
                start += 1
                    
            len_window = i+1 - start
                
            if len_window < window_len:
                window_len = len_window
                start_index = start
                    
    return string[start_index:start_index + window_len]



  











######	Median of two sorted arrays of same odd length – merge approach -> O(n)

def mergeOp(array1, array2):
    i=0
    j=0
    count =0
    n_1 = len(array1)-1
    n_1_set = 0 
    n_set = 0
    n= len(array1)
    
    #print(n_1,n)
    
    while i < len(array1) and j < len(array2):
        if array1[i] < array2[j]:
            if count == n_1 and n_1_set == 0:
                n_1 = array1[i]
                n_1_set = 1
            if count == n and n_set == 0:
                n = array1[i]
                n_set = 1
            i+=1
            count+=1
            
        else:
            if count == n_1 and n_1_set == 0:
                n_1 = array2[j]
                n_1_set = 1
            if count == n and n_set == 0:
                n = array2[j]
                n_set = 1
            j+=1
            count+=1
    
    while i < len(array1):
        if count == n_1 and n_1_set == 0:
            n_1 = array1[i]
            n_1_set = 1
        if count == n and n_set == 0:
            n = array1[i]
            n_set = 1
        i+=1
        count +=1
    
    while j < len(array2):
        if count == n_1 and n_1_set == 0:
            n_1 = array2[j]
            n_1_set = 1
        if count == n and n_set == 0:
            n = array2[j]
            n_set = 1
        j+=1
        count+=1
    
    median = (n_1 + n)/2
    
    #print(n_1,n,median, result)
    
    return median    


#________look for optimised solution in O(logn) and one  that works with even length/ generic(any length)________





######	Given 2 sorted arrays, find element of the final merged sorted array.

#(This approach is in O(n); better can be done in O(logn+logm) -> similar to prev. problem optimised approach)
def kth(array1, array2, m, n, k): 
    #write your code here
    i=0
    j=0
    count=0
    
    
    while i < m and j < n:
        if array1[i] < array2[j]:
            if count == k-1:
                return array1[i]
            count+=1
            i+=1
        else:
            if count == k-1:
                return array2[j]
            count+=1
            j+=1
    
    while i < m:
        if count == k-1:
            return array1[i]
        count+=1
        i+=1
    
    while j< n:
        if count == k-1:
            return array2[j]
        count+=1
        j+=1


   
#Linked List

######	Singly LinkedList

class Node:
    def __init__(self,data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        
    def insert (self,value):
        newNode = Node(value)
        if self.head == None:
            self.head = newNode
            self.tail = newNode
        else:
            self.tail.next = newNode
            self.tail = newNode
    
    def search (self,value):
        ptr = self.head
        
        while ptr != None:
            if ptr.data == value:
                return True
            else:
                ptr = ptr.next
        return False
    
    def delete (self,value):
        ptr = self.head
        prev = None
        if ptr = None : 
            return
        
        while ptr != None:
            if ptr.data != value:
                prev = ptr
                ptr = ptr.next
            elif ptr.data == value:
                if ptr == head:
                    head = ptr.next
                    return
                else:
                    if ptr.next == None:
                        self.tail = prev
                    prev.next = ptr.next
######	Doubly LinkedList
# Doubly LinkedList

class DNode:
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right= None

class DLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def insert(self,value):
        newNode =  DNode(value)
        if head == None:
            self.head = newNode
            self.tail = newNode
        else:
            self.tail.next = newNode
            newNode.prev = self.tail
            self.tail = newNode
    
    def search(self,value):
        ptr = self.head
        while ptr != None:
            if ptr.data == value:
                return True
            else:
                ptr = ptr.next
        return False
    
    def delete(self.value)
        ptr = self.head
        if ptr == None:
            return
        while ptr != None:
            if ptr.data != value:
                ptr = ptr.next
            else:
                if ptr == self.head:
                    self.head = ptr.next
                    return
                else:
                    if ptr == self.tail:
                        self.tail = ptr.prev
                        ptr.prev.next = ptr.next
                    else:
                        ptr.prev.next = ptr.next
                        ptr.next.prev = ptr.prev
                    
######	Rearrange singly Linked List by selecting one node from start and one from end – O(n) time and O(1) space ---------------(Timeout!) -> check and fix find middle 

# Function to print given linked list
def printList(head,result):
 
    ptr = head
    while ptr:
        result.append(ptr.data)
        ptr = ptr.next
 
    return result
  
# Iterative function to reverse nodes of linked list
def reverse(head):
 
    result = None
    current = head
 
    # Iterate through the list and move/insert each node to the
    # front of the result list (like a push of the node)
    while current:
        # tricky: note the next node
        next = current.next
 
        # move the current node onto the result
        current.next = result
        result = current
 
        # process next node
        current = next
 
    # fix head pointer
    return result
 
 
# Recursive function to construct a linked list by merging
# alternate nodes of two given linked lists
#Note in video we dicussed iterative approch to merge Linked List, this is another recursion approch to merge
def mergeLinkedList(a, b):
    #Practise Yourself :  Write your code Here 
    current = Node(0)
    
    while a or b:
        if a:
            current.next = a
            current = a
            a = a.next


        
        if b:
            current.next = b
            current = b
            b = b.next
    
    return current
 
 
# Function to split the linked list into two equal parts and return
# pointer to the second half
def findMiddle(head):
    #Practise Yourself :  Write your code Here 
    slow=head
    fast=head
    prev = None
    while fast and fast.next:
        #prev = slow
        slow = slow.next
        fast = fast.next.next
    return slow
 
 
# Function to rearrange given linked list in specific way
def rearrange(head):
    #Practise Yourself :  Write your code Here 
    mid = findMiddle(head)
    list2 = reverse(head)
    mergeLinkedList(head,list2)

    
	


######	Reverse linked list ‘k’ at a time

class LinkedList: 
    def __init__(self): 
        self.head = None
    def reverse(self, head, k): 
         #Practise Yourself : Write your code Here
        current = head
        prev=None
        next=None
        count =0 
        
        while current!=None and count < k:
            next=current.next
            current.next = prev
            prev= current
            current = next
            count+=1
        
        if next != None:
            head.next = self.reverse(next,k)
        return prev
        
    # Function to insert a new node at the beginning 
    def insert(self, new_data): 
        new_node = Node(new_data) 
        new_node.next = self.head 
        self.head = new_node 
    
    def printList(self,result): 
        temp = self.head 
        while(temp): 
            result.append(temp.data) 
            temp = temp.next
        return result



######	Clone a linked list with random pointer

def traverse(head):
    #Practise Yourself :  Write your code Here 
    result = []
    current = head
    
    while current != None:
        result.append(current.data)
        current = current.next
    
    print(result)

def cloneLinkedList(head):
    #Practise Yourself :  Write your code Here 
    current = head
    headClone = None
    #current_Clone = headClone
    
    while current!=None:
        newNode = Node(current.data)
        current = current.next
        if headClone == None:
            headClone = newNode
            current_Clone = headClone
        else:
            current_Clone.next = newNode
            current_Clone = current_Clone.next
            
    str = head
    ptr = headClone
    temp = str.next
    
    while temp!=None:
        str.next = ptr
        ptr.random = str
        ptr = ptr.next
        str = temp
        temp = temp.next
    str.next = ptr
    ptr.random = str
        
    currentC= headClone
    
    while currentC != None:
        currentC.random = currentC.random.random.next
        currentC = currentC.next
        
    return headClone







######	Print Nth element from end of linked list  --------------(Attribute Error!)

def printNthFromLast(head, n): 
    first = head
    second = head
    for i in range(n):
        if first != None:
            first = first.next
    
    while first != None:
        second = second.next
        first = first.next
    
    return second.data


######	Intersection point of a the linked list

class Solution(object):
    def findLength(self,head):
        count = 0
        current = head
        while current!=None:
            count+=1
            current = current.next
        return count

    
    def traverseCheck(self,diff,largerList,smallerList):
        
        for i in range(diff):
            largerList = largerList.next
        
        while largerList :
            if largerList.data == smallerList.data:
                return largerList
            largerList = largerList.next
            smallerList = smallerList.next
        return -1
   
    def getIntersectionNode(self, headA, headB):
      #Write Your Code here 
        len1 = self.findLength(headA)
        len2 = self.findLength(headB)
        
        if len1 >= len2:
            diff = len1 - len2
            ele = self.traverseCheck(diff, headA, headB)
        else:
            diff = len2 - len1
            ele = self.traverseCheck(diff, headB, headA)
        
        if ele != None:
            return ele



######	Merge Sort in Linked List – in place

class LinkedList: 
    def __init__(self): 
        self.head = None
    # insert new value to linked list 
    # using insert method 
    def insert(self, new_value):
        # Allocate new node 
        new_node = Node(new_value) 
        # if head is None, initialize it to new node
        if self.head is None: 
            self.head = new_node 
            return
        curr_node = self.head 
        while curr_node.next is not None: 
            curr_node = curr_node.next
        # Append the new node at the end 
        # of the linked list 
        curr_node.next = new_node 
        
    def findMiddleList(self,head):
        slow = head
        fast = head
        prev = None
        
        while fast !=None and fast.next !=None:
            prev=slow
            slow=slow.next
            fast=fast.next.next
        return prev
        
    def mergeOp(self,first,second):
        tempNode = Node(0)
        first_c = first
        second_c = second
        head_merged = None
        
        while first_c != None and second_c !=None:
            if first_c.data < second_c.data:
                tempNode.next = first_c
                tempNode = first_c
                first_c = first_c.next
                
                if head_merged == None:
                    head_merged = first
                
            else:
                tempNode.next = second_c
                tempNode = second_c
                second_c = second_c.next
                
                if head_merged == None:
                    head_merged = second
                    
        while first_c != None:
            tempNode.next = first_c
            tempNode = first_c
            first_c = first_c.next
            
        while second_c != None:
            tempNode.next = second_c
            tempNode = second_c
            second_c = second_c.next
        
        return head_merged
      
    def mergeSort(self, h):
        #Practise Yourself :  Write your code Here 
        if h == None or h.next == None:
            return h
        
        middle = self.findMiddleList(h)
        middle_next = middle.next
        middle.next = None
        
        first = self.mergeSort(h)
        second = self.mergeSort(middle_next)
        sortedList = self.mergeOp(first,second)
        
        return sortedList

       
######	Check if linked list is a palindrome


def ispalindrome(head): 
    #Write Your code here 
    mid = None
    
    slow = head
    fast = head
    prev =None
    
    if fast.ptr == None:
        return True
        
    while fast != None and fast.ptr!=None:
        prev = slow
        slow = slow.ptr
        fast = fast.ptr.ptr
    
    if fast == None:
        mid = slow
    else:
        mid = slow.ptr
    
    prev.ptr = None
    
    # reverse one half of the string
    current = mid
    new_prev = None
    
    while current != None:
        new_next = current.ptr
        current.ptr = new_prev
        new_prev = current
        current = new_next
    
    head_h = new_prev

    
    while new_prev and head:
        if new_prev.data != head.data:
            return False
        new_prev = new_prev.ptr
        head = head.ptr
    return True



######	Segregate List into Even – Odd

#return the list from resultant Linkedlist
def reverse(head):
    current = head
    prev = None
    while current:
        next = current.next
        current.next = prev
        prev = current
        current = next
    return prev
    
def createList( head): 
    #Write your code here
    result =[]
    
    while head:
        result.append(head.data)
        head = head.next
    
    return result 
    
#return linkedlist of even and odd Segregate 
def evenOdd( head): 
    head = reverse(head)    # req. since in insert method head ref is at last inserted node 
    #Write your code here
    even_head = None
    odd_head = None
    
    even = Node(0)
    odd = Node(0)
    current = head
    
    while current:
        if current.data %2 == 0:
            even.next = current
            even = current
            if even_head == None:
                even_head = current
        else:
            odd.next = current
            odd = current
            if odd_head == None:
                odd_head = current
        current = current.next
        
    odd.next = None
    even.next = None
    
    even.next = odd_head
    
    return even_head



######	Sum List of two linked list

def reverse(head):
    prev = None
    current = head
    next = None
    
    while current != None :
        next = current.next
        current.next = prev
        prev = current
        current = next
    return prev
  
# Add contents of two linked lists and return the head 
# node of resultant list 
def addTwoLists(first, second): 
    # Here reverse of head ref. of first and second not required as head points to last inserted node
    first = first.head  # need to get ref. of head of linked list - an attribute of linked list
    second = second.head
    
    head_result = None
    current_result = None
    
    carry = 0
    
    while first != None or second != None:
        if first.data : 
            d1 = first.data
        else: 
            d1 = 0
        if second.data :
            d2 = second.data
        else:
            d2 = 0
        
        add = d1 + d2 + carry
        
        if add > 9 :
            store = add % 10
            carry = int(add/10)
        else:
            store = add
        
        newNode = Node(store)
        
        if head_result and current_result :
            current_result.next = newNode
            current_result = current_result.next
        
        if head_result == None:
            head_result = newNode
            current_result = head_result
            
        first = first.next
        second = second.next
        
    if carry > 0:
        finalNode = Node(carry)
        current_result.next = finalNode
        current_result = current_result.next
        
    sum_list = reverse(head_result)
    
    return sum_list    
        
  
# Utility function to return list of resultant linkedlist 
def printList(head): 
    result = []
    temp = head 
    while(temp): 
        result.append(temp.data) #
        print (temp.data) 
        temp = temp.next
    return result






######	LRU Cache implementation – get() & set() in O(1)

class DLinkedNode(): 
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None
            
class LRUCache():
    def _add_node(self, node):
        #Practise Yourself :  Write your code Here 
        if self.front == None and self.rear == None:    #if empty case
            self.front = node
            self.rear = node
        else:
            node.prev = self.rear
            self.rear.next = node
            self.rear = node
        

    def _remove_node(self, node):
        #Practise Yourself :  Write your code Here 
            
        if node != self.front:
            node.next.prev = node.prev
            node.prev.next = node.next
            
        elif node == self.front:
            node.next.prev = node.prev #None
            self.front = node.next
            
        node.prev = self.rear
        node.next = None
        self.rear.next = node
        self.rear = node  
        

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        #Practise Yourself :  Write your code Here 
        self.front = None
        self.rear = None
        self.cache = {}
        self.capacity = capacity
        self.size = 0
        



    def get(self, key):
        #Practise Yourself :  Write your code Here 
        if self.cache.get(key):
            valueNode = self.cache.get(key)
            value = valueNode.value
            
            if valueNode != self.rear:
                self._remove_node(valueNode)
            
            return value
        else:
            return -1
        

    def put(self, key, value):
        #Practise Yourself :  Write your code Here 
        newNode = DLinkedNode()
        newNode.key = key
        newNode.value = value
        
        if self.size <= self.capacity:
            self.cache.setdefault(key,newNode)
            self.size += 1
            self._add_node(newNode)
        else:
            temp = self.front
            temp_key = temp.key
            
            self.front = self.front.next
            self.front.prev = None
            temp.next = None
            
            self.cache.pop(temp_key)
            
            self.cache.setdefault(key,newNode)
            self._add_node(newNode)
            
            
cache = LRUCache(3)
 
 
cache.put("1","111")
cache.put("2","222")
cache.put("3","333")
 
 
print(cache.get("1"))
print(cache.get("3"))
 
cache.put("4","444") 

cache.put("5","544")

######	Group Anagrams Together

def HashString(word):
    ascii_array = [0] * 26
    for i in word:
        ascii_array[ord(i)-ord('a')] += 1
    
    string_val =''
    for i in ascii_array:
        string_val = string_val + str(i)
    return string_val

def groupAnagrams(words,result):
	#Practise Yourself :  Write your code Here 
    HashStringsDict = {}
    
    for word in words:
        HashStringsDict.setdefault(HashString(word),[]).append(word)
    
    for i in HashStringsDict:
        result.append(HashStringsDict[i])
    
if __name__ == '__main__':
    result = []
    words = ["cat", "dog", "tac", "got", "act"]
    groupAnagrams(words,result)
    print(result)
    


















































######	Binary Tree Level Order Traversal

def LevelOrderTraversal(root):
    #Practise Yourself :  Write your code Here 
    result = []
    if root == None:
        return
    queue = deque()
    queue.append(root)
    
    while queue:
        curr = queue.popleft()
        
        if curr.left:
            queue.append(curr.left)
        if curr.right:
            queue.append(curr.right)
        
        result.append(curr.key)
    
    return result


######	Binary Tree Reverse Level Order Traversal

def reverseLevelOrderTraversal(root):
    #Practise Yourself :  Write your code Here 
    result=[]
    
    if root == None:
        return
    
    queue = deque()
    queue.append(root)
    
    stack = deque()
    
    while queue : 
        curr = queue.popleft()
        
        stack.append(curr.key)
        
        if curr.right:
            queue.append(curr.right)
        if curr.left:
            queue.append(curr.left)
    
    while stack : 
        result.append(stack.pop())
    return result
    
    
######	Iterative Preorder Traversal of Binary Tree

def preOrder(root): 
   #Write your code here    
    result = []
    if root == None:
        return
    
    stack = []
    stack.append(root)
    
    while stack:
        curr = stack.pop()
        
        if curr.right:
            stack.append(curr.right)
        if curr.left:
            stack.append(curr.left)
        
        result.append(curr.data)
    return result


######	Vertical sum of Binary Tree

sumMap={}
def verticalLine(root,d=0):
    if root == None:
        return
    verticalLine(root.left,d-1)
    
    sumMap.setdefault(d,[]).append(root.data)
    
    verticalLine(root.right,d+1)

# Function to find and print the vertical sum of given binary tree
def verticalSum(root):
    #Practise Yourself :  Write your code Here 
    verticalLine(root)
    result = []
    for key in sumMap:
        print(key,sumMap[key])
        result.append(sum(sumMap[key]))
    return result




######	Spiral Order of Binary Tree

def printSpiralOrder(root): 
    #Practise Yourself :  Write your code Here 
    result= []
    if root == None:
        return
    s1 = []
    s2 = []
    s1.append(root)
    
    while s1 or s2:
        while s1:
            n1 = s1.pop()
            result.append(n1.data)
            
            if n1.right:
                s2.append(n1.right)
            if n1.left:
                s2.append(n1.left)
            
        while s2:
            n2 = s2.pop()
            result.append(n2.data)
            
            if n2.left:
                s1.append(n2.left)
            if n2.right:
                s1.append(n2.right)
    
    return result

######	Iterative Inorder Traversal of Binary Tree

def inOrder(root): 
   #Write your code here    
    result = []
    if root == None:
        return
    stack = []
    curr = root
    while stack or curr != None:
        while curr != None:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        
        result.append(curr.data)
        curr = curr.right
    
    return result
######	Distance Between two Nodes – LCA, Level, distance

def LowestCommonAncestor(root, n1, n2): 
    #Write your code Here  
    if root == None:
        return 
    if root.data == n1 or root.data == n2:
        return root
    
    left = LowestCommonAncestor(root.left,n1,n2)
    right = LowestCommonAncestor(root.right,n1,n2)
    
    if left and right:
        return root
    if left:
        return  left
    if right:
        return right
    return None

def findLevel(root, n , level):
    if root == None:
        return 
    if root.data == n:
        return level
    left = findLevel(root.left,n,level+1)
    right = findLevel(root.right,n,level+1)
    
    if left:
        return left
    if right:
        return right

#ok for all test cases required  
def findDistance(root, n1, n2): 
    #Write your Code Here
    lca = None
    lca = LowestCommonAncestor(root, n1,n2)
    print('LCA : ',lca.data)
    d1 = findLevel(lca,n1,0)
    d2 = findLevel(lca,n2,0)    
    print('Distances',d1,d2)
    distance = d1 + d2
    
    return distance





######	Bottom View of Binary Tree

def bottomView(root,d,level,hashMap):
    if root == None:
        return
    if d not in hashMap or level >= hashMap[d][1]:
        hashMap[d] = [root.key, level]
    
    bottomView(root.left,d-1,level+1,hashMap)
    bottomView(root.right,d+1,level+1,hashMap)
        
    
# Function to print the bottom view of given binary tree
def bottamViewView(root):
    #Practise Yourself :  Write your code Here 
    result = []
    hashMap={}
    bottomView(root,0,0,hashMap)
    
    for d in sorted(hashMap.keys()):
        result.append(hashMap[d][0])
        print(d,hashMap[d][0])
    
    return result


######	Print all the boundary nodes of Binary Tree

nodeList = []
def leftIntNodes(node):
    if node != None:
        if node.left != None:
            nodeList.append(node.data)
            leftIntNodes(root.left)
        elif node.right != None:
            nodeList.append(node.data)
            leftIntNodes(root.right)

def leafNodes(root):
    if root != None:
        leafNodes(root.left)
        
        if root.right == None and root.left == None:
            nodeList.append(root.data)
        
        leafNodes(root.right)

def rightIntNodes(node):
    if node != None:
        if node.right != None:
            rightIntNodes(node.right)
            nodeList.append(node.data)
        elif node.left != None:
            rightIntNodes(node.left)
            nodeList.append(node.data)

# Function to do boundary traversal of a given binary tree 
def printBoundary(root):
    #Practise Yourself :  Write your code Here 
    nodeList.append(root.data)
    leftIntNodes(root.left)
    leafNodes(root)
    rightIntNodes(root.right)
    
    return nodeList


######	Check whether Binary Tree is Binary Search Tree

def isBST(root,mini,maxi): 

  # Write Your Code here
    if root==None:
        return True
    
    if root.data > maxi or root.data < mini:
        return False
    
    left = isBST(root.left,mini,root.data-1)
    right = isBST(root.right,root.data+1,maxi)
    
    return (left and right)












######	Print nodes at k distance from root

def findNode_DistanceDown(root, k,result): 
    #Practise Yourself :  Write your code Here 
    if root == None:
        return
    if k == 0:
        result.append(root.data)
    
    findNode_DistanceDown(root.left,k-1,result)
    findNode_DistanceDown(root.right,k-1,result)
  

root = None
root = Node(1)  
root.left = Node(2)  
root.right = Node(3) 
result = []
result = findNode_DistanceDown(root,1,result)
print(result)


######	Deletion in Binary Search Tree

class Node: 

	def __init__(self, key): 
		self.key = key 
		self.left = None
		self.right = None


def inorder(root): 
	if root is not None: 
		inorder(root.left) 
		print (root.key) 
		inorder(root.right) 


def insert( node, key): 

	if node is None: 
		return Node(key) 
	if key < node.key: 
		node.left = insert(node.left, key) 
	else: 
		node.right = insert(node.right, key) 

	return node 

def findInorderSuccessor( node): 
     #Write your code here 
	while node.left != None:
	    node= node.left
	return node

def Delete(root, key): 
    #Write your code here 
	if root == None:
	    return 
	
	if key < root.key:
	    root.left = Delete(root.left,key)
	elif key > root.key:
	    root.right = Delete(root.right,key)
	else:
	    if root.left == None and root.right == None:
	        return None
	    elif root.left == None:
	        temp = root.right
	        root = None
	        return temp
	    elif root.right == None:
	        temp = root.left
	        root = None
	        return temp
	    else:
	        minRightValue = findInorderSuccessor(root.right)
	        root.key = minRightValue.key
	        Delete(root.right,minRightValue.key)
	
	return root


root = None
root = insert(root, 50) 
root = insert(root, 30) 
root = insert(root, 20) 
root = insert(root, 40) 
root = insert(root, 70) 
root = insert(root, 60) 
root = insert(root, 80) 

print ('Inorder traversal of the given tree')
inorder(root) 

print ('Delete 20')
root = Delete(root, 20) 
print ('Inorder traversal of the modified tree')
inorder(root) 

print ('Delete 30')
root = Delete(root, 30) 
print ('Inorder traversal of the modified tree')
inorder(root) 

root = Delete(root, 50) 
print ('Inorder traversal of the modified tree')
inorder(root) 





######	Find All Nodes Distance K in Binary Tree

# Recursive function to print all the nodes at distance k 
# int the tree(or subtree) rooted with given root
def printNodesDown(root, k): 
    # Base Case 
    if root is None or k< 0 : 
        return 
    # If we reach a k distant node, print it
    if k == 0 : 
        print (root.data)  
        return 
    # Recur for left and right subtee
    printNodesDown(root.left, k-1) 
    printNodesDown(root.right, k-1) 

# Prints all nodes at distance k from a given target node 
# The k distant nodes may be upward or downward. This function 
# returns distance of root from target node, it returns -1  
# if target node is not present in tree rooted with root
def printNodes(root, target, k):
    # Base Case 1 : IF tree is empty return -1  
    if root is None: 
        return -1
    # If target is same as root. Use the downward function 
    # to print all nodes at distance k in subtree rooted with 
    # target or root 
    if root == target: 
        printNodesDown(root, k) 
        return 0 
    # Recur for left subtree
    dLeft = printNodes(root.left, target, k) 
    # Check if target node was found in left subtree 
    if dLeft != -1:
        # If root is at distance k from target, print root 
        # Note: dLeft is distance of root's left child  
        # from target 
        if dLeft +1 == k : 
            print (root.data) 
        else: 
            printNodesDown(root.right, k-dLeft-2) 
        return 1 + dLeft 
    dRight = printNodes(root.right, target, k) 
    if dRight != -1: 
        if (dRight+1 == k): 
            print (root.data) 
        else: 
            printNodesDown(root.left, k-dRight-2) 
        return 1 + dRight 
    return -1


######	Print All Nodes K Distance from Leaf Node in Binary Tree

def printKDistantfromLeaf(node, k): 
   MAX_HEIGHT = 10
   path = [None] * MAX_HEIGHT 
   visited = [False] * MAX_HEIGHT 
   result=[]
   pathList = printKDistance(node, k, path, visited, 0,result) 
   return pathList

def printKDistance(node, k, path, visited, pathLen,result): 
     #Write your Code Here 
    if node == None:
        return
    
    path[pathLen] = node.key
    visited[pathLen] = False
    
    if (node.left==None and node.right==None and pathLen-k>=0 and visited[pathLen-k]==False):
        print(path[pathLen-k])
        result.append(path[pathLen-k])
        visited[pathLen-k]=True
        return result
    
    tLeft = printKDistance(node.left,k,path,visited,pathLen+1,result)
    tRight = printKDistance(node.right,k,path,visited,pathLen+1,result)

    if tLeft and tRight:
        if len(tLeft)>len(tRight):
            return tLeft
        else:
            return tRight
    if tLeft : 
        return tLeft
    if tRight : 
        return tRight
        

######	Maximum Path Sum in a Binary Tree

result = [0]         

def findMaxPathSumBT(root):
    #Practise Yourself :  Write your code Here 
    
    if root == None:
        return 0
    
    left = findMaxPathSumBT(root.left)
    right = findMaxPathSumBT(root.right)
    
    maxValue = max(root.data,root.data+left,root.data+right)
    
    topMax = max(left+right+root.data,maxValue)
    
    result[0] = max(topMax,result[0])
    
    return maxValue
  

######	Convert a Binary Tree to Sum Tree

def sumTree(Node):
    if Node == None:
        return 0
    
    oldValue = Node.data
    
    left = sumTree(Node.left)
    right = sumTree(Node.right)
    
    Node.data = left + right
    
    return Node.data + oldValue

def convertSumTree(Node) : 
    #Write your Code here
    sumTree(Node)
    return Node
    
    
  
def inorderTraverse(node,result):
    if node == None:
        return
    inorderTraverse(node.left,result)
    result.append(node.data)
    inorderTraverse(node.right,result)
  
def Inorder(Node) :
    #Write your code here
    # This function will return the actual
    result=[]
    inorderTraverse(Node,result)
    return result


######	Find right node of the given Key

desiredLevel = [0]

def findRightNode(root, value, level):
    #Write Your Code Here
    if root == None:
        return
    
    if root.data == value:
        desiredLevel[0] = level
        return 
    
    if desiredLevel[0] != 0:
        if level == desiredLevel[0]:
            return root.data
    
    left = findRightNode(root.left,value,level+1)
    if left!=None:
        return left
    return findRightNode(root.right,value,level+1)


######	Mirror Image of Binary Tree

def convertToMirror(root):
    #Write your code here
    if root == None:
        return
    
    left = convertToMirror(root.left)
    right = convertToMirror(root.right)
    
    #swap op
    root.left = right
    root.right = left
    
    return root

######	Binary Tree Traversal Implementation

def printInorder(root): 
    #Practise Yourself :  Write your code Here 
    if root:
        printInorder(root.left)
        print(root.val)
        printInorder(root.right)
  
def printPostorder(root): 
    #Practise Yourself :  Write your code Here 
    if root:
        printPostorder(root.left)
        printPostorder(root.right)
        print(root.val)
  
def printPreorder(root): 
    #Practise Yourself :  Write your code Here 
    if root:
        print(root.val)
        printPreorder(root.left)
        printPreorder(root.right)


######	Find the kth Largest element in BST

import unittest
class Node:

	def __init__(self, data, left=None, right=None):
		self.data = data
		self.left = left
		self.right = right

#ok for all test cases required
def insert(root, key):

	if root is None:
		return Node(key)

	if key < root.data:
		root.left = insert(root.left, key)

	else:
		root.right = insert(root.right, key)

	return root



def kthLargest(root, i, k):

	#Write your code Here
    if root == None:
        return None,i
    
    right,i = kthLargest(root.right,i,k)
    if right:
        return right,i
    print(root.data,i)
    i += 1
    if i == k+1:
        return root.data,i
    left,i = kthLargest(root.left,i,k)
    return left,i


def findKthLargest(root, k):

	i = 0
    #Write your code Here
	return kthLargest(root,i,k)[0]
	

class Test(unittest.TestCase):
  def test_findKthLargest1(self):
    root = None
    keys = [13, 14, 22, 25, 23, 32, 26,28,40]
    for key in keys:
      root = insert(root, key)
    actual = findKthLargest(root,3)
    expected = 26
    self.assertEqual(actual, expected)

  def test_findKthLargest2(self):
    root = None
    keys = [5,3,6,2,4,1]
    for key in keys:
      root = insert(root, key)
    actual = findKthLargest(root,3)
    expected = 3
    self.assertEqual(actual, expected)


unittest.main(verbosity=2)





######	Binary Tree to Doubly Linked List Conversion
#Represent a node of binary tree  
class Node:  
    def __init__(self,data):  
        self.data = data;  
        self.left = None;  
        self.right = None;  
          
class BinaryTreeToDLL:  
    def __init__(self):  
        #Represent the root of binary tree  
        self.root = None;  
        #Represent the head and tail of the doubly linked list  
        self.head = None;  
        self.tail = None;  
          
    head = None
    #This function will convert the given binary tree to corresponding doubly linked list  
    def convertbtToDLL(self, node):  
        #Practise Yourself :  Write your code Here 
        if node == None:
            return
        
        self.convertbtToDLL(node.left)
        
        if self.head == None:
            self.head = node
            self.root = node
            self.tail = node
        
        else:
            self.tail.right = node
            node.left = self.root
            self.root = node        
        
        self.convertbtToDLL(node.right)
      
    #display() will print out the nodes of the list  
    def display(self,result):  
        #Node current will point to head  
        current = self.head;  
        if(self.head == None):  
            print("List is empty");  
            return;  
        print("Nodes of generated doubly linked list: ");  
        while(current != None):  
            #Prints each node by incrementing pointer.  
            result.append(current.data),  
            current = current.right;  
          
         
bt = BinaryTreeToDLL();  
#Add nodes to the binary tree  
bt.root = Node(1);  
bt.root.left = Node(2);  
bt.root.right = Node(3);  
bt.root.left.left = Node(4);  
bt.root.left.right = Node(5);  
bt.root.right.left = Node(6);  
bt.root.right.right = Node(7);  
   
#Converts the given binary tree to doubly linked list  
bt.convertbtToDLL(bt.root);  
result = []
#Displays the nodes present in the list  
bt.display(result);
print(result)



######	Height Balanced Tree

import unittest
class Node: 
	def __init__(self, data): 
		self.data = data 
		self.left = None
		self.right = None

#ok for all test cases required 
def findHeight(root): 
	if root is None: 
		return 0
	return max(findHeight(root.left), findHeight(root.right)) + 1

def isTreeBalanced(root): 
	# Write Your Code here
    if root == None:
        return True
    
    left = findHeight(root.left)
    right = findHeight(root.right)
    
    if (abs(left-right)<= 1 and isTreeBalanced(root.left) and isTreeBalanced(root.right)):
        return True
    return False

class Test(unittest.TestCase):
    def test_isTreeBalanced_1(self):
        root = Node(2) 
        root.left = Node(3) 
        root.right = Node(4) 
        root.left.left = Node(5) 
        root.left.right = Node(6) 
        root.left.left.left = Node(9)
        actual = isTreeBalanced(root)
        expected = False
        self.assertEqual(actual, expected)
    
    def test_isTreeBalanced_2(self):
        root = Node(2) 
        root.left = Node(3) 
        root.right = Node(4) 
        root.left.left = Node(5) 
        root.left.left.left = Node(6) 
        root.left.left.left.left = Node(9)
        actual = isTreeBalanced(root)
        expected = False
        self.assertEqual
    
    def test_isTreeBalanced_3(self):
        root = Node(2) 
        root.left = Node(3) 
        root.right = Node(4) 
        root.left.left = Node(5) 
        root.left.right = Node(6) 
    
        actual = isTreeBalanced(root)
        expected = True
        self.assertEqual(actual, expected)


unittest.main(verbosity=2)



######	Diameter of Binary tree

# Data structure to store a Binary Tree node
class Node:
	def __init__(self, data, left=None, right=None):
		self.data = data
		self.left = left
		self.right = right

def findHeight(root):
    if root == None:
        return 0
    return max(findHeight(root.left),findHeight(root.right)) + 1



maxDiameter = [0]
def getBTDiameter(root):
    #Practise Yourself :  Write your code Here 
    
    left = findHeight(root.left)
    right = findHeight(root.right)
    
    diameter = left + right + 1
    
    if diameter >= maxDiameter[0]:
        maxDiameter[0] = diameter
    

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.right = Node(4)
root.right.left = Node(5)
root.right.right = Node(6)
root.right.left.left = Node(7)
root.right.left.right = Node(8)
getBTDiameter(root)
print(maxDiameter[0])




######	Print Left View of Binary Tree

# Data structure to store a Binary Tree node
class Node:
	def __init__(self, key, left=None, right=None):
		self.key = key
		self.left = left
		self.right = right
	
level = [0]
# Recursive function to print left view of given binary tree
def printLeftView(root,k=0):
	#Write your code here
	if root == None:
	    return
	
	if level[0] == k:
	    print(root.key)
	    level[0] += 1
	
	printLeftView(root.left,k+1)
	printLeftView(root.right,k+1)


if __name__ == '__main__':

	root = Node(1)
	root.left = Node(2)
	root.right = Node(3)
	root.left.right = Node(4)
	root.right.left = Node(5)
	root.right.right = Node(6)
	root.right.left.left = Node(7)
	root.right.left.right = Node(8)

	printLeftView(root)





######	Check Nodes of Binary Tree is Cousins

class Node:
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right

def findElemRelation(root,elem,level=0,parent=None):
    if root == None:
        return 
    
    left = findElemRelation(root.left,elem,level+1,root)
    right = findElemRelation(root.right,elem,level+1,root)
    
    if root.key == elem:
        return level,parent
        
    if left:
        return left
    if right:
        return right
    

def isTheyCousins(root, elem1, elem2):
    #write your code here
    level1,parent1 = findElemRelation(root,elem1)
    level2,parent2 = findElemRelation(root,elem2)
    
    if level1 == level2 and parent1 != parent2:
        return True
    else:
        return False
 
if __name__ == '__main__':
 
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.right.left = Node(6)
    root.right.right = Node(7)
 
    if isTheyCousins(root, 5, 6):
        print("The given nodes are cousins")
    else:
        print("The given nodes are not cousins")
 



