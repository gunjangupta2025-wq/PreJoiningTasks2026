import numpy as np

######################################
######################################
###########EXERICISE 1 A##############
zero_a = np.zeros([4,5])
print(zero_a)

one_a = np.ones([4,5])
print(one_a)
print("\n", one_a.shape)
arange_a = np.arange(0, 5, 1)
print(arange_a)

linspace_a = np.linspace(0, 5, 10)
print(linspace_a)

identity_matrix = np.eye(4)
print(identity_matrix)

two_d_array = np.array([[1,2,3], [4,5,6]])
print(two_d_array)
print(two_d_array.shape)

print(two_d_array[0,0])
print(two_d_array[0,1])
print(two_d_array[1][0])

#3d array
three_d_array = np.ones((2,3,2))
print(three_d_array)
print(three_d_array.shape)
print(f"Sum over axis 0: {np.sum(three_d_array, axis=0)}")
print(f"Sum over axis 1: {np.sum(three_d_array, axis=1)}")

######Slicing##########
#[[1 2 3]
# [4 5 6]
# [7 8 9]]
#If i want to print just 2 3 5 6 
# then I wanna print 2nd coloumn and upto 2nd row
a1 = np.array([[1, 2, 3],[4,5,6], [7,8,9]])
print(a1)
print(a1[:2, 1:3])

#Reversed rows and coloumns
print(a1[:-1, :-1])

###Reshape and Transpose operations
arr = np.arange(12)
print(arr)
arr1 = arr.reshape(3,4)
print(arr1)
print(arr1.T)

####Concatenation and Splitting
arr = np.arange(16)
arr1 = arr.reshape(4,4)
print(arr1)

v_stack = np.concatenate([arr1, arr1], axis=0)
# print(v_stack)

h_stack = np.concatenate([arr1, arr1], axis=1)
# print(h_stack)

upper, lower = np.split(arr1, [2], axis=0)
print(upper)
print(lower)

######################################
######################################
###########EXERICISE 1 B##############

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

print(x + y)
#There is a distinction between np.add and np.sum
print("add", np.add(x,y))
print("sum", np.sum(x))
#elementwise multiplication
print(x * y)
#matrix multiplication
print(np.dot(x,y))
print(x.dot(y))
print(np.dot(y,x))
print(y.dot(x))

######Statistiscal#########
data = np.array([[1, 5, 3], 
                 [4, 2, 6]])

print("Global stats")
total_mean = np.mean(data)
print(total_mean)
global_max = np.max(data)
print(global_max)
global_min = np.min(data)
print(global_min)

print("Axis-specific stats")
col_means = np.mean(data, axis=0)  # [2.5, 3.5, 4.5]
row_mins = np.min(data, axis=1)    # [1, 2]
std_deviation = np.std(data)
print(col_means)
print(row_mins)
print(std_deviation)



import numpy as np
########1B#############
x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

print(x + y)
#There is a distinction between np.add and np.sum
print("add", np.add(x,y))
print("sum", np.sum(x))
#elementwise multiplication
print(x * y)
#matrix multiplication
print(np.dot(x,y))
print(x.dot(y))
print(np.dot(y,x))
print(y.dot(x))

######Statistiscal#########
data = np.array([[1, 5, 3], 
                 [4, 2, 6]])

print("Global stats")
total_mean = np.mean(data)
print(total_mean)
global_max = np.max(data)
print(global_max)
global_min = np.min(data)
print(global_min)

print("Axis-specific stats")
col_means = np.mean(data, axis=0)  # [2.5, 3.5, 4.5]
row_mins = np.min(data, axis=1)    # [1, 2]
std_deviation = np.std(data)
print(col_means)
print(row_mins)
print(std_deviation)

######Sorting and searching in arrays##########
a = np.array([12, 13, 6, 7, 1, 3, 90, 2])

s_a = np.sort(a, )
print(s_a)
print(s_a[::-1])
s_a_a = np.argsort(a)
print(s_a_a)  

g_t_s = np.where(a > 7)
print(g_t_s)
replace = np.where(a > 7, -1, a)
print(replace)