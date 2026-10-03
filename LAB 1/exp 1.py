def find_s_algorithm(dataset, target):
    # Determine the number of attributes
    num_attributes = len(dataset[0])
    
    # Step 1: Initialize the hypothesis to the most specific hypothesis (all nulls)
    hypothesis = ['0'] * num_attributes
    print(f"Initial Hypothesis (H0): {hypothesis}\n")
    
    # Track if we have encountered the first positive example
    first_positive = True
    
    # Step 2: Iterate through each training sample
    for i, instance in enumerate(dataset):
        if target[i] == "Yes":  # Only process positive examples
            if first_positive:
                # Initialize hypothesis with the first positive example's values
                hypothesis = list(instance)
                first_positive = False
                print(f"Instance {i+1} (Positive) -> Initialized Hypothesis: {hypothesis}")
            else:
                # Generalize the hypothesis based on the new positive example
                for j in range(num_attributes):
                    if hypothesis[j] != instance[j]:
                        hypothesis[j] = '?'  # Generalize to wildcard
                print(f"Instance {i+1} (Positive) -> Updated Hypothesis:     {hypothesis}")
        else:
            print(f"Instance {i+1} (Negative) -> Ignored")
            
    return hypothesis

# --- Demonstration Data ---
# Attributes: [Sky, AirTemp, Humidity, Wind, Water, Forecast]
training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"],
    ["Sunny", "Warm", "High",   "Strong", "Warm", "Same"],
    ["Rainy", "Cold", "High",   "Strong", "Warm", "Change"],
    ["Sunny", "Warm", "High",   "Strong", "Cool", "Change"]
]
# Target Concept: EnjoySport
target_labels = ["Yes", "Yes", "No", "Yes"]

# Execute the algorithm
print("--- Running FIND-S Algorithm ---")
final_hypothesis = find_s_algorithm(training_data, target_labels)

print("\n--- Final Maximally Specific Hypothesis ---")
print(final_hypothesis)