# Assuming you have known criminal profiles
known_criminals = {"John Doe": 35, "Jane Smith": 28, "Mike Johnson": 40}
accused_age = 46 # Age of the accused
# Check if the accused's age matches any known criminal profiles
is_known_criminal = any(abs(age - accused_age) <= 5 for age in known_criminals.values())
#print(any(abs(age - accused_age)))
if is_known_criminal:
     print("The accused matches a known criminal profile.")
else:
     print("The accused does not match any known criminal profiles.")