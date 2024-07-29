import csv
with open('enjoysport.csv', 'r') as csvfile:
    data = list(csv.reader(csvfile))
print(data)

print("\nThe total number of training instances are:", len(data))

num_attributes = len(data[0]) - 1
hypothesis = ['0'] * num_attributes
print("\nThe initial hypothesis is:", hypothesis)

for i, instance in enumerate(data):
    if instance[num_attributes] == 'yes':
        for j in range(num_attributes):
            if hypothesis[j] == '0' or hypothesis[j] == instance[j]:
                hypothesis[j] = instance[j]
            else:
                hypothesis[j] = '?'
    print(f"\nThe hypothesis after training instance {i + 1} is:", hypothesis)

print("\n\nThe Maximally specific hypothesis is:", hypothesis)