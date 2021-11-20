# import packages
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Read the gorcery data
groc_data= pd.read_csv("./Grocery_data.csv",header=None)
#print("Dataset:", groc_data.head(15),"\n")


items_data=groc_data.transpose()
unique_items=[]

for i in range(1000):
    transaction=list(set(items_data[i]))
    [unique_items.append(x) for x in transaction if x not in unique_items]



encoded_vals=[]
# one hot encoding for supplying data in the apriori algorithm
for i, rows in groc_data.iterrows():
    labels={}
    uncommons=list( set(unique_items)-set(rows))
    commons=list( set(unique_items).intersection(rows))

for un in uncommons:
    labels[un] = 0

for com in commons:
    labels[com] = 1
    encoded_vals.append(labels)

# create a dtaframe for enocoded data
encoded_data = pd.DataFrame(encoded_vals)
print("\nOne hot encoded data:\n", encoded_data.head(10),"\n")

frequent_items = apriori(encoded_data, min_support=0.0085, use_colnames=True, verbose=1)
print("Top 15 frequent items:\n",frequent_items.head(15),"\n")

association_rule_generated_confidence= association_rules(frequent_items, metric="confidence", min_threshold=0.25)
print("association rules generated through confidence metrics:\n", association_rule_generated_confidence.head(15),"\n")

"""
# saving the dataframe to a csv file
association_rule_generated_confidence.to_csv("association_rule_generated_confidence.csv")
association_rule_generated_support= association_rules(frequent_items, metric="support", min_threshold=0.005)
print("association rules generated through support metrics:\n",association_rule_generated_support.head(15),"\n")
# saving the dataframe to a csv file
association_rule_generated_support.to_csv("association_rule_generated_support.csv")
"""
