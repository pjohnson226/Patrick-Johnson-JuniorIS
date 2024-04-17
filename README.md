# Patrick-Johnson-JuniorIS

Name: Topic Extraction for Feature Enrichment in Patent Classification

The software will be the beginnings of a broader research project that aims to utilize extracted topics for a given section of a patent claim to enrich the feature set inputted into a model. As such, in this current project, I acquire accuracy results from a BERT model trained on 21 days worth of patent claim data (specifically the background and claims sections) from the HUPD (Harvard University Patent Dataset) dataset. The model is then tested on 9 days worth of data. The accuracy metric exhibits how well the model predicts whether a given claim will be accepted (0) or rejected (1). I then use Latent Dirichlet Allocation to extract topics from the background and claims section to enrich the train_set of the BERT model. The results are then evaluated to see if the extracted topics benfit or hinder the models performance on the binary classification task. 

It is important to note that the results will always be underwhelming due to the small size of the sample dataset provided by the HUPD which is what is used in this project. This is primarily for debugging purposes. Given a larger time range and a better understanding of the computational requirements of years worth of patent data would make the use of the larger dataset offered by the HUPD more feasible. 
