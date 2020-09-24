import numpy as np

#creates an 8-element vector
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)

#creates a two-dimensional matrix,
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)

#generates a sequence that includes the lower bound (5) but not the upper bound (12).
sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)

#populates a 6-element vector with random integers between 50 and 100.
random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_integers_between_50_and_100)

#To create random floating-point values between 0.0 and 1.0, call np.random.random
random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1) 

#the following operation uses broadcasting to add 2.0 to the value of every item in the vector created in the previous code cell:
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)

random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)

#task 1
feature = np.arange(6,21)
print(feature)
label = 3 * feature + 4
print(label)

#task 2
noise = (np.random.random(15) - 0.5) * 4  
print(noise)
label = label + noise  
print(label)