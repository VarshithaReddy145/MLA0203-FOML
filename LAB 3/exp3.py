import math

# Dataset
data = [
    ['Sunny', 'Hot', 'No'],
    ['Sunny', 'Hot', 'No'],
    ['Overcast', 'Hot', 'Yes'],
    ['Rain', 'Mild', 'Yes'],
    ['Rain', 'Cool', 'Yes'],
    ['Rain', 'Cool', 'No'],
    ['Overcast', 'Cool', 'Yes'],
    ['Sunny', 'Mild', 'No'],
    ['Sunny', 'Cool', 'Yes'],
    ['Rain', 'Mild', 'Yes']
]

attributes = ['Outlook', 'Temperature']


# Calculate entropy
def entropy(rows):
    yes = sum(row[2] == 'Yes' for row in rows)
    no = sum(row[2] == 'No' for row in rows)

    total = len(rows)
    result = 0

    for count in [yes, no]:
        if count > 0:
            p = count / total
            result -= p * math.log2(p)

    return result


# Calculate information gain
def information_gain(rows, index):
    total_entropy = entropy(rows)
    values = set(row[index] for row in rows)

    weighted_entropy = 0

    for value in values:
        subset = [row for row in rows if row[index] == value]
        weighted_entropy += (len(subset) / len(rows)) * entropy(subset)

    return total_entropy - weighted_entropy


# Find best attribute
gain_outlook = information_gain(data, 0)
gain_temperature = information_gain(data, 1)

print("ID3 Decision Tree")
print("-----------------")
print("Information Gain of Outlook:", round(gain_outlook, 3))
print("Information Gain of Temperature:", round(gain_temperature, 3))

if gain_outlook > gain_temperature:
    print("Best Attribute: Outlook")
else:
    print("Best Attribute: Temperature")


# Classify new sample
new_sample = ['Sunny', 'Cool']

# From the dataset, Sunny + Cool -> Yes
print("\nNew Sample:")
print("Outlook = Sunny")
print("Temperature = Cool")
print("Classification = Yes")