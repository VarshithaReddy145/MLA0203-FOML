import csv

def load_csv(filename):
    """Reads the CSV file and separates features from the target label."""
    concepts = []
    target = []
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        
        for row in reader:
            if row: 
                concepts.append(row[:-1])
                target.append(row[-1])
    return concepts, target

def learn(concepts, target):
    """Implements the Candidate-Elimination Algorithm."""
    num_attributes = len(concepts[0])
    
    
    specific_h = ["0"] * num_attributes
    general_h = [["?"] * num_attributes]
    
    print("Initial Specific Boundary (S0):", specific_h)
    print("Initial General Boundary (G0) :", general_h)
    print("-" * 60)
    
    for i, val in enumerate(target):
        if val.strip().lower() in ["yes", "1", "true"]:
            specific_h = concepts[i].copy()
            break

    for i, instance in enumerate(concepts):
        is_positive = target[i].strip().lower() in ["yes", "1", "true"]
        print(f"\nProcessing Instance {i+1}: {instance} | Positive: {is_positive}")
        
        if is_positive:
            
            for x in range(num_attributes):
                if instance[x] != specific_h[x]:
                    specific_h[x] = '?'
        
            general_h = [g for g in general_h if all(g[x] == '?' or g[x] == instance[x] for x in range(num_attributes))]

        else:
           
            new_general_h = []
            for g in general_h:
                for x in range(num_attributes):
                   
                    if g[x] == '?':
                        if instance[x] != specific_h[x]:
                            g_new = g.copy()
                            g_new[x] = specific_h[x]
                            if g_new not in new_general_h:
                                new_general_h.append(g_new)
                    else:
                        if g not in new_general_h:
                            new_general_h.append(g)
            
            general_h = [g for g in new_general_h if all(g[x] == '?' or g[x] == specific_h[x] for x in range(num_attributes))]

        print(f"S[{i+1}]: {specific_h}")
        print(f"G[{i+1}]: {general_h}")
        
    return specific_h, general_h

if __name__ == "__main__":
    csv_filename = "training_data.csv"
    
    print(f"Loading data from '{csv_filename}'...\n")
    try:
        concepts, target = load_csv(csv_filename)
        final_s, final_g = learn(concepts, target)
        
        print("\n" + "="*60)
        print("FINAL VERSION SPACE BOUNDARIES")
        print("="*60)
        print("Final Specific Boundary (S):", final_s)
        print("Final General Boundary (G) :", final_g)
        
    except FileNotFoundError:
        print(f"Error: Please create the '{csv_filename}' file in the same directory first.")
