import math
import matplotlib.pyplot as plt

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

print("\nNew Sample:")
print("Outlook = Sunny")
print("Temperature = Cool")
print("Classification = Yes")


# ---------------- GRAPH ----------------

plt.figure(figsize=(10, 7))

# Root node
plt.text(0.5, 0.90, "Outlook",
         ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightblue"))

# Second level
plt.text(0.20, 0.65, "Temperature",
         ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightgreen"))

plt.text(0.50, 0.65, "Yes",
         ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightgreen"))

plt.text(0.80, 0.65, "Temperature",
         ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightgreen"))

# Leaf nodes
plt.text(0.10, 0.35, "Yes",
         ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.text(0.30, 0.35, "No",
         ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.text(0.70, 0.35, "No",
         ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.text(0.90, 0.35, "Yes",
         ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Connections from Outlook
plt.plot([0.5, 0.20], [0.87, 0.68], 'k-')
plt.plot([0.5, 0.50], [0.87, 0.68], 'k-')
plt.plot([0.5, 0.80], [0.87, 0.68], 'k-')

# Connections from Sunny Temperature
plt.plot([0.20, 0.10], [0.62, 0.38], 'k-')
plt.plot([0.20, 0.30], [0.62, 0.38], 'k-')

# Connections from Rain Temperature
plt.plot([0.80, 0.70], [0.62, 0.38], 'k-')
plt.plot([0.80, 0.90], [0.62, 0.38], 'k-')

# Branch labels
plt.text(0.32, 0.78, "Sunny")
plt.text(0.51, 0.78, "Overcast")
plt.text(0.66, 0.78, "Rain")

plt.text(0.12, 0.50, "Cool")
plt.text(0.27, 0.50, "Hot/Mild")

plt.text(0.72, 0.50, "Cool")
plt.text(0.87, 0.50, "Mild")

plt.title("Decision Tree using ID3 Algorithm")

plt.axis("off")
plt.show()