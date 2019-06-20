# with open("pi.txt") as file_object:
#     contents = file_object.read()
#     print(contents)

# filename = "pi.txt"
# with open(filename) as file_project:
#     for line in file_project:
#         print(line)
#
# with open(filename) as file_project:
#     lines = file_project.readlines()
#
# pi_string = ''
# for line in lines:
#     pi_string += line.strip()

# print(pi_string)

# filename = "pi.txt"
# with open(filename, 'w') as file_object:
#     file_object.write("i love niu\n")
#
# with open(filename, 'a') as file_object:
#     file_object.write("he love me")
#
# with open("pi.txt") as file_object:
#     contents = file_object.read()
#     print(contents)

# try:
#     result = 5/0
# except ZeroDivisionError:
#     print("can not divide by zero")
# else:
#     print(result)

# title = "alice in wonderland"
# print(title.split())

import json
numbers = [1, 2, 3, 4, 5]
filename = 'number.json'
# with open(filename, 'w') as f_obj:
#     json.dump(numbers, f_obj)

with open(filename) as f_obj:
    numbers = json.load(f_obj)
print(numbers)
