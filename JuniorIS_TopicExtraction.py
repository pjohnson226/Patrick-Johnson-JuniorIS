!pip install datasets
# Datasets load_dataset function for Huggingface
from datasets import load_dataset

dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather",
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-21',
    val_filing_start_date='2016-01-22',
    val_filing_end_date='2016-01-31',
)

#The below code follows the same preprocessing steps as the ModelTraining.py file
import random
# Label-to-index mapping for the decision status field
decision_to_str = {'REJECTED': 0, 'ACCEPTED': 1, 'PENDING': 2, 'CONT-REJECTED': 3, 'CONT-ACCEPTED': 4, 'CONT-PENDING': 5}

# Helper function
def map_decision_to_string(example):
    return {'decision': decision_to_str[example['decision']]}

train_set = dataset_dict['train'].map(map_decision_to_string)
val_set = dataset_dict['validation'].map(map_decision_to_string)
# Filter out labels 2, 3, 4 and 5
filtered_train_set = train_set.filter(lambda example: example['decision'] not in [2, 3, 4, 5])
filtered_val_set = val_set.filter(lambda example: example['decision'] not in [2, 3, 4, 5])
num_accepted = len(filtered_train_set.filter(lambda example: example['decision'] == 0))
num_rejected = len(filtered_train_set.filter(lambda example: example['decision'] == 1))
print(f'Accepted: {num_accepted}')
print(f'Rejected: {num_rejected}')

# Separate data by label (0 and 1)
accepted_examples = [ex for ex in filtered_train_set if ex['decision'] == 0]
rejected_examples = [ex for ex in filtered_train_set if ex['decision'] == 1]

# Calculate the desired split size (50% of the smaller group)
split_size = min(len(accepted_examples), len(rejected_examples)) // 2

# Randomly sample split_size elements from each label group
train_accepted = random.sample(accepted_examples, split_size)
train_rejected = random.sample(rejected_examples, split_size)

# Combine train examples while maintaining order
balanced_train_set = train_accepted + train_rejected

# Print the sizes of the balanced train set
print(f"Balanced train set size: {len(balanced_train_set)}")

# Calculate the test set size (15% of the total data)
test_size_balanced = int(0.15 * len(balanced_train_set))

# Create the test set by taking the remaining examples
test_set_balanced = balanced_train_set[-test_size_balanced:]

# Remove the test examples from the balanced train set
train_set_balanced = balanced_train_set[:-test_size_balanced]

# Print the sizes of balanced train and test sets
print(f"Balanced train set size: {len(train_set_balanced)}")
print(f"Balanced test set size: {len(test_set_balanced)}")
#^The above code follows the same preprocessing steps as the ModelTraining.py file

filtered_TSB = []
for item in train_set_balanced:
  filtered_item = {'claims': item['claims'], 'decision': item['decision']}
  filtered_TSB.append(filtered_item)
pprint(filtered_TSB[0])
#^Isolate only the claims and decision for supervised BERTopic for train set


filtered_VSB = []
for item in test_set_balanced:
  filtered_item = {'claims': item['claims'], 'decision': item['decision']}
  filtered_VSB.append(filtered_item)
pprint(filtered_VSB[0])
#^Do the same for validation set

train_docs = [item['claims'] for item in filtered_TSB]
train_y = [item['decision'] for item in filtered_TSB]
test_docs = [item['claims'] for item in filtered_VSB]
test_y = [item['decision'] for item in filtered_VSB]
#^Further partitioning to isolate claims from decisions into document variable and target variable for train and val sets, a prerequisite for BERTopic

!pip install bertopic
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
#^Install BERTopic for supervised classification and other necessary packages for BERTopic

vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
# Skip over dimensionality reduction (not needed for supervised BERTopic, replace cluster model with classifier (necessary for supervised BERTopic),
# and reduce frequent words (best practice for topic extraction)
empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
#^instantiating the "layers" of BERTopic


# Create a fully supervised BERTopic instance
topic_model= BERTopic(
        umap_model=empty_dimensionality_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model=clf,
        ctfidf_model=ctfidf_model
)
topics, probs = topic_model.fit_transform(train_docs, y=train_y)
#^Establishing supervised BERTopic model and running topic extraction

