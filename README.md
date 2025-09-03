# FinTextSim: Enhancing Financial Text Analysis with BERTopic: A Comparative Study of Topic Modeling Techniques
Repository containing the code for training and evaluating FinTextSim as outlined in Jehnen et al. (2025) (https://doi.org/10.48550/arXiv.2504.15683).
FinTextSim is a sentence-transformer optimized for clustering and semantic search of financial texts. 
Compared to the most widely used sentence-transformers, FinTextSim increases intratopic similarity by up to 71% while simultaneously reducing intertopic similarity by 102%, allowing unprecedented separation between financial topics.
The processing of the code is organized into several modules.

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

### Results
- contains results presented in the paper FinTextSim: Enhancing Financial Text Analysis with BERTopic: A Comparative Study of Topic Modeling Techniques by Jehnen et al. (2025) (https://doi.org/10.48550/arXiv.2504.15683)
  - evaluation of embedding models on the test set
  - quantitative results from BERTopic and classical topic modeling approaches
  - qualitative results: topic assignments from BERTopic and classical topic modeling approaches to a selection of sentences  

## Results

### FinTextSim vs. other embedding models on financial text (testset from training)
FinTextSim will significantly outperform other embedding models on financial text, creating clear and distinct clusters of financial topics.
| Model                | Intratopic Similarity ↑ | Intertopic Similarity ↓ | Outliers within BERTopic↑ |
|-----------------------|--------------------------|---------------------------|------------------|
| FinTextSim            | **0.997**                | **-0.011**                  | **295,706**             |
| all-MiniLM-L6-v2 (AM) | 0.583                     | 0.559                      | 781,965             |
| all-mpnet-base-v2     | 0.626                     | 0.611                      | 784,225             |


In the following plots, each datapoint represents a sentence from FinTextSim's testset. The color of the datapoint is associated with its financial topic.

**FinTextSim**
<img width="2160" height="1440" alt="red_embeddings_FinTextSim" src="https://github.com/user-attachments/assets/f7ad0f33-7f1a-4c18-bde9-1ff2afcbe498" />
Other embedding models will fail to capture the semantic concepts of financial text, leading to an indistinguishable mash of datapoints.

**all-MiniLM-L6-v2 (AM)**
<img width="2160" height="1440" alt="red_embeddings_AM" src="https://github.com/user-attachments/assets/82c4fb4d-14cc-423e-a2a8-f1865ecb3ef3" />

**all-mpnet-base-v2 (MPNET)**
<img width="2160" height="1440" alt="red_embeddings_MPNET" src="https://github.com/user-attachments/assets/3705ed6a-70a3-4c54-b157-d8cd21be95cb" />

### Topic Quality
Using BERTopic in conjunction with FinTextSim will be able to identify financial topics, clearly separating topic domains. 
The following plots show the created topic representations of the topic models. 
The color of each word represents its associated unique topic from the keyword list. Words colored in black are not present in the keyword list.

**BERTopic-FinTextSim**
<img width="2000" height="1200" alt="wordcloud_bertopic_htl_modern_bert" src="https://github.com/user-attachments/assets/db386555-3149-4666-a158-fc86f77d5b37" />

Classical topic modeling approaches and BERTopic with other embedding models will fail to do so.

**BERTopic-AM**
<img width="2000" height="1200" alt="wordcloud_bertopic_AM" src="https://github.com/user-attachments/assets/1194d7cc-a1ae-427d-9dc6-4dbe60c0723a" />

**LDA**
<img width="1000" height="800" alt="wordcloud_lda_tfidf" src="https://github.com/user-attachments/assets/9ba96a05-58ac-4912-8ca2-29e6793b545c" />

Hence, FinTextSim is necessary to distinguish between financial topics in a large corpus of documents.


### Organizing Power
Only when using BERTopic in combination with FinTextSim, a high intratopic similarity and low intertopic similarity can be expected.
Without FinTextSim, BERTopic will struggle with misclassification and overlapping topics. 
Thus, FinTextSim is pivotal for advancing financial text analysis.
The following plots show examples of topic assignments on selected sentences across the different topic modeling techniques.

**Example 1:** 
Sentence: acquisitionrelated expenses and restructuring costs are recognized separately from the business combination and are expensed as incurred
<img width="2500" height="500" alt="cost_57400" src="https://github.com/user-attachments/assets/18f1f302-4915-4e71-8739-36c374f5c4bb" />

**Example 2:** 
Sentence: our commercial customers consist primarily of delivery customers for whom we deliver products from our store locations to our commercial customers places of business including independent garages service stations and auto dealers
<img width="2500" height="500" alt="operations_7350" src="https://github.com/user-attachments/assets/b5e9b2f5-0114-4c77-b626-930ce6375d8c" />

**Example 3:** 
Sentence: currencyneutral comparable operating profit declined due to the weakness in the australian cereal category and our performance in south africa
<img width="2500" height="500" alt="profit_loss_4500" src="https://github.com/user-attachments/assets/310f2055-bfa2-4198-9e90-f7eb4b4facf8" />


## Implications
FinTextSim’s enhanced contextual embeddings, tailored for the financial domain, elevate the quality of future research and financial information. 
This improved quality of financial information will enable stakeholders to gain a competitive advantage, streamlining resource allocation and decision-making processes. 
Moreover, the improved insights have the potential to leverage business valuation and stock price prediction models.

## Citation
To cite the FinTextSim paper, please use the following bibtex reference:

**Jehnen, S., Ordieres-Meré, J., & Villalba-Díez, J. (2025).**  
*FinTextSim: Enhancing Financial Text Analysis with BERTopic.*  
arXiv:2504.15683. https://doi.org/10.48550/arXiv.2504.15683  

```bibtex
@article{jehnen2025,
  title={FinTextSim: Enhancing Financial Text Analysis with BERTopic},
  author={Jehnen, Simon and Ordieres-Mer{\'e}, Joaqu{\'\i}n and Villalba-D{\'\i}ez, Javier},
  journal={arXiv preprint arXiv:2504.15683},
  year={2025}
}
