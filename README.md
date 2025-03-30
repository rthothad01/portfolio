<!-- https://rthothad01.github.io/portfolio/ -->
# Portfolio

## LLM Projects
### Study Assistant
- Developed a Study Assistant using LangChain that:
1. Summarizes study material into concise points.
2. Automatically generates multiple-choice quiz questions based on the summarized content.
3. Functions without the need for external retrieval mechanisms or vector database.


### Search engine retriever
- Built a search engine on Wikipedia articles & some research papers such as Attention is all you need
	1. Processed both text and PDF documents using LangChain
	2. Created Document and Contextual Chunks using Recursive Character Text Splits & Chunking strategy and also used Contextual Chunking with the help of gpt4o-mini
	3. Indexed Chunks and embeddings in a chroma vector database using Langchain
	4. Experimented with different retrievers listed below
		1. Similarity based Retrieval
		2. Multi Query Retrieval to rephrase the question in different ways using gpt4o-mini. For a question about CNN, was able to get back documents related to CNN, the cable news channel and CNN, the convolutional neural network.
		3. Contextual Compression Retrieval to filter out irrelevant information.
		4. Chained Retrieval Pipeline, which inlcudes basic retrieval strategies such as cosine similarity, filtering out noise using LLM chain filtering compression and then use HF's BGE re-ranker model to re-rank the results.

### Evaluate Generated Content
- Used RAGAS and DeepEval to evaluate context that was retrieved and the response provided.
	1. Used ContextualPrecisionMetric Co,nextualRecallMetric and ContextualRelevancyMetric to evaluate the context retrieved.
	2. Used to AnswerRelevancyMetric using a LLM approach and a similarity based approach using RAGA framework to evaluate the response.
	
## Datasets Practiced from Kaggle
1. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#imbalanced]() [#BinaryClassification]()
2. [Predict the LLM](https://www.kaggle.com/competitions/h2oai-predict-the-llm) where I fine-tuned a Mistral-7B model &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#NLP]() [#MultiClassClassiification]()
3. [House Price Prediction](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#Regression]()
4. [Quora Question Pair](https://www.kaggle.com/competitions/quora-question-pairs) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#NLP]() [#Regression]()
5. [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#RecommenderSystem]() [#MultiClassClassification]()
6. [Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/overview) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#NLP]() [#Regression]()
7. [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#Regression]()
8. [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#NLP]() [#BinaryClassification]()
9. [Forecasting Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;	[#TimeSeries]() [#Regression]()
10. [Steel Plate Defect Prediction](https://www.kaggle.com/competitions/playground-series-s4e3) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;[#MultiClassClassiification]()

# Published Articles
1. [Santander Product Recommendation](https://medium.com/@ravitee/santander-product-recommendation-ee4122d15072)
2. [Portfolio Optimization using Principal Component Analysis](https://medium.com/@ravitee/portfolio-optimization-using-principal-component-analysis-923f102a8a47)
3. [A Primer on AI Agents](https://medium.com/@ravitee/a-primer-about-ai-agents-1e34f6dc7a4d)

# Contact
ravi.thothadri@hotmail.com