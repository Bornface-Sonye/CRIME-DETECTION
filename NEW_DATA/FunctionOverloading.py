def overloaded_function(arg1, arg2=None):
    if arg2 is None:
        # Handle the case with one argument
        return arg1
    else:
        # Handle the case with two arguments
        return arg1 + arg2

# Usage
result1 = overloaded_function(5)
result2 = overloaded_function(3, 7)

print(result1)  # Output: 5
print(result2)  # Output: 10