# FinTextSim: Enhancing Financial Text Analysis with BERTopic: A comparative study of topic modeling techniques
Repository containing the code for training and evaluating FinTextSim as outlined in Jehnen et al. (2025) (https://doi.org/10.48550/arXiv.2504.15683).
FinTextSim is a sentence-transformer optimized for clustering and semantic search of financial texts. 
Compared to the most widely used sentence-transformers, FinTextSim increases intratopic similarity by up to 71% while simultaneously reducing intertopic similarity by 102%, allowing unprecedented separation between financial topics.
The main code and results are located in src/. The processing of the code is organized into several modules.

## Data Origin
We source our data from the Notre Dame Software Repository for Accounting and Finance in text-file format, which underwent a 'Stage One Parse' to remove all HTML tags. 
The data can be found at: https://sraf.nd.edu/data/stage-one-10-x-parse-data/.

## Main Components
### documents
- extract 10-K reports from S&P 500 companies
- isolation of the MDA&A section (Item 7 and Item 7A)
- detection of outlier documents
- tokenization into sentences and text cleaning
- perform necessary preprocessing steps to conduct topic modeling with BERTopic as well as LDA and NMF

### labeled_dataset
- create a labeled dataset which will be used for the training of FinTextSim
- follows a keyword- and substring-based logic to create a labeled dataset

### fintextsim
- train FinTextSim with BatchHardTripletLoss based on the created labeled dataset
- create test and train datasets - perform the test-/train-split topic-wise to ensure balanced learning by reducing the impact of class imbalance issues

### bertopic_models
- fit BERTopic models with varying embedding models
- generation of embeddings with varying embedding models
- generate parameters to evaluate NPMI coherence of BERTopic models

### classical_models
- fit classical topic models (LDA, NMF)

### evaluation
- quantitative evaluation of the quality of the embedding models on the test set
- quantitatively evaluate NPMI coherence, topic-precision as well as inter- and intratopic similarity of both BERTopic and classical topic models
- qualitative assessment of the generated topic representations

## Expected Results

### FinTextSim vs. other embedding models on financial text (testset from training)
FinTextSim will significantly outperform other embedding models on financial text, creating clear and distinct clusters of financial topics.
Other embedding models will fail to capture the semantic concepts of financial text, leading to an indistinguishable mash of datapoints.


### Topic Quality
Using BERTopic in conjunction with FinTextSim will be able to identify financial topics, clearly separating topic domains. Classical topic modeling approaches and BERTopic with other embedding models will fail to do so.
Hence, FinTextSim is necessary to distinguish between financial topics in a large corpus of documents.

### Organizing Power
Only when using BERTopic in combination with FinTextSim, a high intratopic similarity and low intertopic similarity can be expected.
Without FinTextSim, BERTopic will struggle with misclassification and overlapping topics. 
Thus, FinTextSim is pivotal for advancing financial text analysis.

## Implications
FinTextSim’s enhanced contextual embeddings, tailored for the financial domain, elevate the quality of future research and financial information. 
This improved quality of financial information will enable stakeholders to gain a competitive advantage, streamlining resource allocation and decision-making processes. 
Moreover, the improved insights have the potential to leverage business valuation and stock price prediction models.
