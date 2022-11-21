# ImpactCite Sentiment Analysis

## Files
The data folder contains the **complete_corpus** file that stores the **Citation Sentiment Corpus**.
The **Duplicates_handling** files stores information about the processing and the invalid instances.
The **dataset** folder covers the files created with the notbooks included in this repository.
The **dataset_full** file consists of a list with the data samples and a list with the textual labels (o: neutral, p: positive, n: negative).
The **dataset_folds** file has a similar shape but the data and label lists are divided into the different folds.

## Usage
To create the clean dataset from scratch run the **dataset_cleaner** notebook followed by the **dataset_converter** notebook.

Train the XLNet in a similar fashion as the **ImpactCite Intent** does.
