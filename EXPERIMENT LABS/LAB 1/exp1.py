def find_s_algorithm(dataset, target):
    
    num_attributes = len(dataset[0])
    
    
    hypothesis = ['0'] * num_attributes
    print(f"Initial Hypothesis (H0): {hypothesis}\n")
    
    
    first_positive = True
    
   
    for i, instance in enumerate(dataset):
        if target[i] == "Yes":  
            if first_positive:
              
                hypothesis = list(instance)
                first_positive = False
                print(f"Instance {i+1} (Positive) -> Initialized Hypothesis: {hypothesis}")
            else:
                
                for j in range(num_attributes):
                    if hypothesis[j] != instance[j]:
                        hypothesis[j] = '?'  # Generalize to wildcard
                print(f"Instance {i+1} (Positive) -> Updated Hypothesis:     {hypothesis}")
        else:
            print(f"Instance {i+1} (Negative) -> Ignored")
            
    return hypothesis


training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"],
    ["Sunny", "Warm", "High",   "Strong", "Warm", "Same"],
    ["Rainy", "Cold", "High",   "Strong", "Warm", "Change"],
    ["Sunny", "Warm", "High",   "Strong", "Cool", "Change"]
]

target_labels = ["Yes", "Yes", "No", "Yes"]


print("--- Running FIND-S Algorithm ---")
final_hypothesis = find_s_algorithm(training_data, target_labels)

print("\n--- Final Maximally Specific Hypothesis ---")
print(final_hypothesis)