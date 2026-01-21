# FinTextSim: A Domain-Specific Sentence-Transformer for Extracting Predictive Latent Topics from Financial Disclosures
Repository containing the code for training and evaluating FinTextSim as outlined in Jehnen et al. (2026) (https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1752103/abstract).
FinTextSim is a sentence-transformer optimized for clustering and semantic search of financial texts. 
Compared to the most widely used sentence-transformers and financial baselines, FinTextSim increases intratopic similarity by up to 71% while simultaneously reducing intertopic similarity by more than 108%, allowing unprecedented separation between financial topics.
Crucially, these qualitative gains translate into quantitative predictive benefits: incorporating FinTextSim-derived topic features into a logistic regression framework for corporate performance prediction leads to a statistically significant two-percentage-point increase in both ROC-AUC and F1-score over a purely financial baseline. 
In contrast, off-the-shelf sentence-transformers and classical topic models introduce noise that degrades predictive performance. 
For non-linear classifiers, several textual representations yield modest gains, reflecting their greater capacity to absorb noisier features. 
However, FinTextSim remains the most stable and consistently strong performer across both linear and non-linear settings.

The processing of the code is organized into several modules.

## Data Origin
We source our data from the Notre Dame Software Repository for Accounting and Finance in text-file format, which underwent a 'Stage One Parse' to remove all HTML tags. 
The data can be found at: https://sraf.nd.edu/data/stage-one-10-x-parse-data/.
Financial data is sourced from FinancialModelingPrep (https://site.financialmodelingprep.com/)

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
- create test and train datasets - perform the test-/train-split temporal to avoid data leakage

### feature_creation
- create financial and textual features

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

### ml
- preprocessing of ML-based corporate performance prediction framework
- running of lightweight Logistic Regression and XGBoost ML framework, predicting normalized ROA direction changes
- evaluation of ML results

### Results
- contains results presented in the paper FinTextSim: Enhancing Financial Text Analysis with BERTopic: A Comparative Study of Topic Modeling Techniques by Jehnen et al. (2026) (https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1752103/abstract)
  - evaluation of embedding models on the test set
  - quantitative results from BERTopic and classical topic modeling approaches
  - qualitative results: topic assignments from BERTopic and classical topic modeling approaches to a selection of sentences
  - predictive validity: incorporating FinTextSim-derived topic features into a logistic regression framework for corporate performance prediction leads to a statistically significant two-percentage-point increase in both ROC-AUC and F1-score over a purely financial baseline. In contrast, off-the-shelf sentence-transformers and classical topic models introduce noise that degrades predictive performance. For non-linear classifiers, several textual representations yield modest gains, reflecting their greater capacity to absorb noisier features. However, FinTextSim remains the most stable and consistently strong performer across both linear and non-linear settings.

## Results

### FinTextSim vs. other embedding models on financial text (testset from training)
FinTextSim will significantly outperform other embedding models on financial text, creating clear and distinct clusters of financial topics.
| Model                | Intratopic Similarity ↑ | Intertopic Similarity ↓ | Outliers within BERTopic ↓ |
|-----------------------|--------------------------|---------------------------|------------------|
| FinTextSim            | **0.998**                | **-0.075**                  | **240,823**             |
| all-MiniLM-L6-v2 (AM) | 0.584                     | 0.563                      | 781,965             |
| all-mpnet-base-v2     | 0.614                     | 0.625                      | 784,225             |
| distil-RoBERTa      | 0.773                    | 0.883                        | 1,332,620          |


In the following plots, each datapoint represents a sentence from FinTextSim's testset. The color of the datapoint is associated with its financial topic.

**FinTextSim**
![red_embeddings_FinTextSim](https://github.com/user-attachments/assets/66001b4c-1467-49a4-bba1-48fdbda49b7b)
Other embedding models will fail to capture the semantic concepts of financial text, leading to an indistinguishable mash of datapoints.

**all-MiniLM-L6-v2 (AM)**
<img width="2160" height="1440" alt="red_embeddings_AM" src="https://github.com/user-attachments/assets/82c4fb4d-14cc-423e-a2a8-f1865ecb3ef3" />

**all-mpnet-base-v2 (MPNET)**
<img width="2160" height="1440" alt="red_embeddings_MPNET" src="https://github.com/user-attachments/assets/3705ed6a-70a3-4c54-b157-d8cd21be95cb" />

**distilroberta-finetuned-financial-news-sentiment-analysis (DR)**
![red_embeddings_DR](https://github.com/user-attachments/assets/fafdb1e0-d18d-477a-b788-2e9d4fe62c65)


### Topic Quality
Using BERTopic in conjunction with FinTextSim will be able to identify financial topics, clearly separating topic domains. 

| Model                  | Topic-Accuracy ↑   | NPMI Coherence ↑ |
|-----------------------|---------------------|------------------|
| BERTopic-FinTextSim   | **0.81**            | 0.287            | 
| BERTopic-AM           | 0.06                | **0.387**        |
| BERTopic-MPNET        | 0.23                | 0.382            |
| BERTopic-DR           | 0.09                | 0.368            |
| LDA                   | 0                   | 0.039            |
| NMF                   | 0.11                | 0.239            |

The following plots show the created topic representations of the topic models. 
The color of each word represents its associated unique topic from the keyword list. Words colored in black are not present in the keyword list.

**BERTopic-FinTextSim**
![wordcloud_htl_temporal](https://github.com/user-attachments/assets/0607cdb9-8476-4096-be2f-554e0e97a97c)


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

| Model                  | Intertopic Similarity ↓ | Intratopic Similarity ↑ |
|------------------------|-------------------------|---------------------------|
| BERTopic-FinTextSim    | **-0.034**              | **0.939**                  | 
| BERTopic-AM            | 0.465                   | 0.596                           |
| BERTopic-MPNET         | 0.511                   | 0.656                     | 
| BERTopic-DR            | 0.745                   | 0.948                      | 
| LDA                    | 1                       | 0                       |
| NMF                    | 0.202                   | 0.881                      | 
The following plots show examples of topic assignments on selected sentences across the different topic modeling techniques.

**Example 1:** 
Sentence: we calculate revpar by dividing hotel room revenue by total number of room nights available to guests for a given period
![sales_130000](https://github.com/user-attachments/assets/8cecfdfd-2183-4d31-a24a-d94be2a4cc32)


**Example 2:** 
Sentence: reported operating profit of million in was million or higher than reported operating profit of million in
![profit_loss_884295](https://github.com/user-attachments/assets/0fd039e7-3459-4ac2-9f60-fd8bc65fb9e8)


**Example 3:** 
Sentence: in the fourth quarter we recognized our frontline employees for their commitment and contributions to their communities during the pandemic with a award that was paid in january
![hr_temporal_1769440](https://github.com/user-attachments/assets/2ce932b5-e298-4ed4-b440-d63481fc28b4)



### Predictive Validity

#### Logistic Regression
Our results for LR show that improved textual representations translate into tangible predictive benefits.
Only the topics generated by BERTopic in combination with FinTextSim yield a significant improvement in predictive power over purely financial features, as evidenced by a two-percentage-point increase in ROC-AUC and F1-score. 
In contrast, OTS sentence-transformers and classical topic models provide no improvement and, in some cases, even degrade performance, indicating that their latent features introduce noise rather than signal.

| Model                | Accuracy ↑ | F1-Score ↑ | AUC-ROC ↑ |
|-----------------------|--------------------------|---------------------------|---------------------------|
| Financial             | 0.692               |0.578                  | 0.688 |
| Financial + AM       | 0.638                     | 0.533                      | 0.646             |
| Financial + MPNET     | 0.669                     | 0.565                      | 0.658 |
| Financial + FinTextSim     | 0.686                     | **0.599**                      | **0.708** |
| Financial + DR     | 0.665                     | 0.539                      | 0.667 |
| LDA     | 0.674                     | 0.556                      | 0.677             |
| NMF     | 0.662                     | 0.564                      | 0.690 |


#### XGBoost
For XGBoost, several textual representations yield modest gains, reflecting their greater capacity to absorb noisier features. However, FinTextSim remains the most stable and consistently strong performer across both linear and non-linear settings.

| Model                | Accuracy ↑ | F1-Score ↑ | AUC-ROC ↑ |
|-----------------------|--------------------------|---------------------------|---------------------------|
| Financial             | 0.636               |0.603                  | 0.672 |
| Financial + AM       | 0.663                     | **0.626**                      | 0.674             |
| Financial + MPNET     | 0.648                     | 0.580                      | 0.667 |
| Financial + FinTextSim     | 0.660                     | 0.612                      | **0.686** |
| Financial + DR     | 0.657                     | 0.594                      | 0.676 |
| LDA     | **0.670**                     | 0.622                      | 0.676             |
| NMF     | 0.667                     | 0.608                      | 0.682 |


## Implications
Our work offers several key contributions.
First, we advance contextual embeddings for the financial domain with FinTextSim, which functions as a domain-adapted information filter, addressing the fundamental information processing and retrieval bottleneck in financial text analysis. By transforming unstructured narratives into structured, semantically rich representations, FinTextSim enhances the quality of extracted information and enables ML models to detect economically meaningful signals often overlooked by human analysts and generic models.
Second, FinTextSim strengthens the informational content of textual data, allowing analysts and researchers to derive actionable insights that support efficient resource allocation and more informed decision-making.
Third, by bridging classical and contemporary topic modeling techniques, we establish a foundation for methodologically consistent and empirically validated model selection in financial text analysis.
Finally, we demonstrate the practical value of FinTextSim in a downstream corporate performance prediction task. Thus, our research lays the foundation for integrating narrative information into valuation and forecasting frameworks, highlighting that qualitative disclosures can complement quantitative financial metrics in predictive applications.


## Citation
To cite the FinTextSim paper, please use the following bibtex reference:

**Jehnen, S., Ordieres-Meré, J., & Villalba-Díez, J. (2026).**  
*FinTextSim: Enhancing Financial Text Analysis with BERTopic.*  
arXiv:2504.15683. https://doi.org/10.48550/arXiv.2504.15683  

```bibtex
@article{jehnen2026,
  title={FinTextSim: A Domain-Specific Sentence-Transformer for Extracting Predictive Latent Topics from Financial Disclosures},
  author={Jehnen, Simon and Ordieres-Mer{\'e}, Joaqu{\'\i}n and Villalba-D{\'\i}ez, Javier},
  journal={Frontiers in Artificial Intelligence},
  year={2026}
}
