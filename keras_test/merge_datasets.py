import random

arr = list(range(1, 150))

random.shuffle(arr)

dev = arr[0:95]
validate = arr[95:120]
test = arr[120:150]

# print(dev)
# print(validate)
# print(test)

dev = [88, 84, 87, 131, 19, 47, 23, 48, 108, 45, 96, 55, 57, 26, 70, 128, 97, 140, 135, 66, 139, 138, 24, 106, 71, 18, 142, 34, 115, 134, 32, 2, 60, 86, 77, 54, 52, 46, 101, 43, 68, 81, 51, 13, 29, 56, 126, 6, 121, 69, 123, 63, 133, 9, 59, 76, 104, 33, 112, 103, 83, 114, 14, 122, 7, 39, 144, 85, 99, 113, 130, 36, 62, 93, 74, 3, 148, 89, 94, 27, 75, 21, 12, 64, 127, 98, 149, 65, 4, 15, 8, 16, 145, 38, 44]
validate = [117, 90, 80, 41, 105, 5, 102, 49, 129, 143, 137, 11, 91, 132, 100, 82, 72, 17, 22, 118, 42, 125, 50, 67, 124]
test = [20, 35, 136, 73, 110, 31, 1, 120, 78, 141, 61, 147, 58, 109, 37, 116, 79, 111, 53, 146, 95, 92, 30, 119, 25, 40, 28, 10, 107]

def write_man(dataset, filename):
  data = ''
  for idx, num in enumerate(dataset):
      with open('dataset/' + str(num).zfill(3) + '.txt') as f:
        data += '-DOCSTART- ' + str(num).zfill(3) + '\n\n'
        data += f.read() 
        if idx < len(dev) - 1:
          data += '\n' 
  with open('dataset/' + filename + '.txt', 'w') as fw:
    fw.write(data)

write_man(test, 'test')
write_man(dev, 'dev')
write_man(validate, 'validate')
