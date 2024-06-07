paragraph = """
Imagine we have a dataset for a rare disease screening. The dataset has 1000 samples, where only 10 samples are positive (have the disease) and 990 samples are negative (do not have the disease).

Now, let's build a binary classifier for this dataset. Suppose the classifier is very conservative and predicts all samples as negative (i.e., it never predicts a positive case). Here’s how the performance would look:

True Positives (TP): 0 (no positive case is correctly identified)
False Negatives (FN): 10 (all actual positive cases are missed)
True Negatives (TN): 990 (all actual negative cases are correctly identified)
False Positives (FP): 0 (no negative case is incorrectly identified)
Let's calculate the metrics:

Accuracy = 99%
TPR (True Positive Rate) = 0%

In this example:

The accuracy is very high at 99%, because the classifier correctly identifies all 990 negative samples out of 1000 total samples.
However, the TPR is 0 because the classifier fails to identify any of the positive samples.
This scenario demonstrates a binary classifier with high accuracy but low TPR. This can happen in imbalanced datasets where the number of negative samples far outweighs the number of positive samples. In such cases, a classifier that predicts all samples as negative can achieve high accuracy but will have a very low TPR, failing to detect the positive cases altogether. This is why metrics like ROC curves, which consider the balance between TPR (sensitivity) and FPR (false positive rate), are often preferred for evaluating classifiers, especially in imbalanced datasets.
"""

print(paragraph)
