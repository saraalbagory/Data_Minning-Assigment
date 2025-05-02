import itertools
import gradio as gr
from collections import defaultdict
from itertools import combinations
import itertools
from itertools import chain
#this function will take a percentage and
# return a list of transactions that make up that percentage of the file as a set
def read_file(percentage: int,file_path: str):

    with open("data\categories.txt", "r") as file:
        #get the file size
        file.seek(0, 2)
        file_size = file.tell()
        byte_limit = file_size * (int(percentage)/ 100)
        file.seek(0)
        processed_bytes = 0
        dataset = []
        for index, line in enumerate(file):
            processed_bytes += len(line.encode('utf-8'))
            line = line.strip() # Remove the newline character
            categories_list= line.split(";")
            dataset.append(set(categories_list))
            # print(f"Line {index + 1}: {line.strip()}")
            if processed_bytes >= byte_limit:
                break

    return dataset


def support_count(dataset, item):
    count = 0
    for transaction in dataset:
        if item.issubset(transaction):
            count += 1
    return count

def support(dataset, item):
    count = 0
    for transaction in dataset:
        if item.issubset(transaction):
            count += 1
    return count/len(dataset)

def confidence(dataset, item1, item2):
    return support_count(dataset, item1.union(item2)) / support_count(dataset, item1)

def generate_association_rules(dataset,frequent_item_set, min_confidence):
    association_rules = []
    for item_set in frequent_item_set:
        if(len(item_set)==1):
            continue
        elif len(item_set) == 2:
            print(f"Generating association rules len =2 for  {item_set}")
            item1, item2 = map(frozenset, [list(item_set)[0:1], list(item_set)[1:2]])
            if confidence(dataset, item1, item2) >= min_confidence:
                association_rules.append((item1, item2, confidence(dataset, item1, item2)))
            if confidence(dataset, item2, item1) >= min_confidence:
                association_rules.append((item2, item1, confidence(dataset, item2, item1)))

        else:
            print(f"Generating association rules for {item_set}")
            for r in range(1, len(item_set)):
                combinations = itertools.combinations(item_set, r)
                for combo in combinations:
                    item1=frozenset(combo)
                    item2=item_set-item1
                    if len(item2)==0:
                        continue
                    # print(f"item1: {item1}, item2: {item2}")
                    candidate_rule_confidence=confidence(dataset, item1, item2)
                    if candidate_rule_confidence >= min_confidence:
                        association_rules.append((item1, item2,candidate_rule_confidence))
    return association_rules
def generate_candidates(dataset, k):
    # Generate the candidate item sets
    candidates=set()
    if k == 1:
        unique_categories = set(chain(*dataset))
        candidates = {frozenset([category]) for category in unique_categories}  # Generate sets for k=1
    else:
        for itemset1 in dataset:
            for itemset2 in dataset:
                if len(itemset1.union(itemset2)) == k:
                    candidates.add(itemset1.union(itemset2))

    return candidates

def association_rules_to_string(asst_rules):
    rules_str = []
    for item1, item2, confidence in asst_rules:
        item1_str = ", ".join(item1)
        item2_str = ", ".join(item2)
        rule_str = f"{item1_str} -> {item2_str}, Confidence: {confidence} \n"
        rules_str.append(rule_str)
        
    return "\n".join(rules_str)


def apriori(min_support_count, min_confidence, percentage,file_path):
    dataset = read_file(percentage,file_path)
    i=1
    prev_frequent_itemset = dataset
    while True:
        new_frequent_itemset = []
        generated_candidates= generate_candidates(prev_frequent_itemset, i)
        print(f"Number of candidates of length {i}: {len(generated_candidates)}")
        if not generated_candidates:
            print("entered break 1")
            break

        for category in generated_candidates:
            count = support_count(dataset, category)
            if count >= min_support_count:
                new_frequent_itemset.append(category)
                category_str = " , ".join(category)
                print(f"Support count for {category_str}: {count}")
        # there is no new frequent itemset
        print(f"Number of frequent itemsets of length {i}: {len(new_frequent_itemset)}")
        if not new_frequent_itemset:
            print("entered break 2")
            break
        prev_frequent_itemset= new_frequent_itemset
        i += 1
    
    return association_rules_to_string( generate_association_rules(dataset,prev_frequent_itemset,min_confidence))



interface = gr.Interface(
    fn=apriori,  # Function to be called
    inputs=[
        gr.Number(label="Minimum Support Count"),
        gr.Number(label="Minimum Confidence"),
        gr.Number(label="Percentage of Dataset to Use"),
        gr.Textbox(label="file path")  # Hidden input for file path
    ],
    outputs="text",  # Output type
    title="Find Association Rules",
    description="Enter the minimum support count, minimum confidence, and percentage of the dataset to analyze."
)

# Launch the interface
interface.launch()