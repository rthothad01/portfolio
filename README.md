<!-- https://rthothad01.github.io/portfolio/ -->
# Portfolio

## LLM Projects
### Financial Chat assistant
- Developed a chat assistant to chat with Raymond James' 1Q25 Results that had lot of charts. The goal was to bring the relevant chart while answering the questions.
1. Used OpenAI's text-embedding-3-large, which would help capture nuaned semantic meanings.
2. Used Llamaparse to extract text and images, using OpenAI's multimodal LLM, gpt-40
3. Mapped text and image using LlamaIndex's data structure, TextNode.
4. Created custom report output to output both Text and Image blocks for the query.
5. Used gpt-4o to query.

### Retrieve News Articles using Hypothetical Answers
- Developed a News assistant to answer broad-themed questions such as "What impact does the anti-trust case against google have on its long term value?"
1. Uses LLM to answer to generate 10 different questions for the user question and have the LLM generate hypothetical answers.
2. Fetches news articles for the ticker using newsapi.org.
3. Uses similarity ranking to rank the fetched news articles against the hypothetical answers.
4. Returns the top n news articles.

### Financial Analyst Agent
- Developed a financial analyst to do sentiment analysis and trend analysis on the market data.
1. Used crewai for orchestration.
2. Build 2 agents, one to act as a Financial News Analyst, whose role is collect and analyze financial news article to identify sentiment.
3. Second agent, acts as a Data Analyst, who role is to analyze historical market data to identify trends.

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
1. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) &nbsp;&nbsp;&nbsp;&nbsp; 	[#imbalanced]() [#BinaryClassification]()
2. [Predict the LLM](https://www.kaggle.com/competitions/h2oai-predict-the-llm) where I fine-tuned a Mistral-7B model &nbsp;&nbsp;&nbsp;&nbsp; 	[#NLP]() [#MultiClassClassiification]()
3. [House Price Prediction](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) &nbsp;&nbsp;&nbsp;&nbsp;	[#Regression]()
4. [Quora Question Pair](https://www.kaggle.com/competitions/quora-question-pairs) &nbsp;&nbsp;&nbsp;&nbsp; [#NLP]() [#Regression]()
5. [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) &nbsp;&nbsp;&nbsp;&nbsp;[#RecommenderSystem]() [#MultiClassClassification]()
6. [Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/overview) &nbsp;&nbsp;&nbsp;&nbsp;[#NLP]() [#Regression]()
7. [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) &nbsp;&nbsp;&nbsp;&nbsp;[#Regression]()
8. [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) &nbsp;&nbsp;&nbsp;&nbsp;[#NLP]() [#BinaryClassification]()
9. [Forecasting Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1) &nbsp;&nbsp;&nbsp;&nbsp;[#TimeSeries]() [#Regression]()
10. [Steel Plate Defect Prediction](https://www.kaggle.com/competitions/playground-series-s4e3) &nbsp;&nbsp;&nbsp;&nbsp;[#MultiClassClassiification]()

# Published Articles
1. [Santander Product Recommendation](https://medium.com/@ravitee/santander-product-recommendation-ee4122d15072)
2. [Portfolio Optimization using Principal Component Analysis](https://medium.com/@ravitee/portfolio-optimization-using-principal-component-analysis-923f102a8a47)
3. [A Primer on AI Agents](https://medium.com/@ravitee/a-primer-about-ai-agents-1e34f6dc7a4d)

# Contact
ravi.thothadri@hotmail.com