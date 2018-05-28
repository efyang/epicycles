# Python Program illustrating
# numpy.reshape() method

import numpy as geek

array = geek.arange(8)
print("Original array : \n", array)

# shape array with 2 rows and 4 columns
array = geek.arange(8).reshape(2, 4)
print("\narray reshaped with 2 rows and 4 columns : \n", array)

# shape array with 2 rows and 4 columns
array = geek.arange(8).reshape(4 ,2)
print("\narray reshaped with 2 rows and 4 columns : \n", array)

# Constructs 3D array
array = geek.arange(8).reshape(2, 2, 2)
print("\nOriginal array reshaped to 3D : \n", array)

array = geek.arange(8).reshape(4, 2)
print("\narray reshaped with 4 rows and 2 columns : \n", array)
