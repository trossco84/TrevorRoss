# i=1
# while i <= 100:
#     if i%3==0:
#         print("Fizz", end="")
#         if i%5==0:
#             print("Buzz", end="")
#     elif i%5==0:
#         print("Buzz", end="")
#     else:
#         print(i, end="")
#     print()
#     i+=1

# Write a short program that prints each number from 1 to 100 on a new line. 

# For each multiple of 3, print "Fizz" instead of the number. 

# For each multiple of 5, print "Buzz" instead of the number. 

# For numbers which are multiples of both 3 and 5, print "FizzBuzz" instead of the number.

for x in range(1,101):
    if x%15==0:print('FizzBuzz')
    elif x%3==0:print('Fizz')
    elif x%5==0:print('Buzz')
    else:print(x)

a,b='Fizz'