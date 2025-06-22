<!-- https://rthothad01.github.io/portfolio/ -->
# Portfolio

## LLM Projects - RAG

<!-- Refer to TechNotes git's LLM/RAG/LlamaIndex folder -->
### Financial Chat assistant 
- Developed a chat assistant to chat with [Raymond James' 1Q25 Results](https://www.raymondjames.com/-/media/rj/dotcom/files/our-company/news-and-media/2025-press-releases/rjf20250129-1q-presentation.pdf) (any PDF can be used) that had lot of charts. The goal was to bring the relevant chart while answering the questions.
1. Used OpenAI's text-embedding-3-large model that creates embeddings with up to 3072 dimensions to capture nuanced semantic meanings.
2. Used Llamaparse to extract text and images, using OpenAI's multimodal LLM, gpt-4o
3. Mapped text and image for each page using LlamaIndex's data structure, TextNode.
4. Created custom report output to output both Text and Image blocks for the query.
5. Used gpt-4o to query.
6. Some of the questions I tried are:
   1. Give me a summary of the financial performance of the different business units.
   2. Give me a summary of whether you think the financial projections are stable, and if not, what are the potential risk factors. Support your research with sources.
   3. What is the Net Interest Income for the current quarter? How does it compare to the previous quarter and the previous year? Support your research with sources.

<!-- Google drive - Hypothetical Answers-->
### Retrieve News Articles using Hypothetical Answers/HyDE
- Developed a News assistant to answer broad-themed questions such as "What impact does the anti-trust case against google have on its long term value?"
1. Used LLM to generate 10 different questions for the user question and LLM to generate hypothetical answers for those questions.
2. Fetched news articles for the ticker using newsapi.org.
3. Used similarity ranking to rank the fetched news articles against the hypothetical answers.
4. Returned the top n news articles.

<!-- Refer to git's LLM/Agentic AI/5.Notebooks.5 -->
### Search engine retriever
- Built a search engine on Wikipedia articles & some research papers such as Attention is all you need
	1. Processed both text and PDF documents using LangChain
	2. Created Document and Contextual Chunks using Recursive Character Text Splits & Chunking strategy and also used Contextual Chunking with the help of gpt4o-mini. The idea here is to split a huge document into managable chunks and store a summary for each chunk that can be used during retrieval.
	3. Indexed Chunks and embeddings in a chroma vector database using Langchain
	4. Experimented with different retrievers listed below with questions such as What is the difference between transformers and vision transformers?, What is a cnn?
		1. Similarity based Retrieval
		2. Multi Query Retrieval to rephrase the question in different ways using gpt4o-mini to overcome the issue when subtle changes in query wording yield different results. This is akin to automating prompt engineering. For a question about what is a CNN, generated similar quesitons such as 'What does CNN stand for and what are its main functions?  ', 'Can you explain the concept and applications of a convolutional neural network?  ', 'What are the key features and uses of CNNs in machine learning?.
		3. Contextual Compression Retrieval to filter out irrelevant information. The information most relevant to a query may be buried in a document with a lot of irrelevant text. Passing that full document through the application can lead to more expensive LLM calls and poorer responses. After retrieving the documents passed it througn LLMChainFilter to retrieve relevant documents.
		4. Chained Retrieval Pipeline, which inlcudes basic retrieval strategies such as cosine similarity, filtering out noise using LLM chain filtering compression (as mentioned above) and then use HuggingFace's BGE re-ranker model to re-rank the results.

<!-- Refer to git's LLM/Agentic AI/5.Notebooks.7 -->
### Content Evaluation
- Used RAGAS and DeepEval to evaluate context that was retrieved and the response provided.
	1. Used ContextualPrecisionMetric,  ContextualRecallMetric and ContextualRelevancyMetric to evaluate the context retrieved.
	2. Used AnswerRelevancyMetric, with both a LLM and a similarity based approach, using RAGA framework to evaluate the response.
   
## LLM Projects - Agentic AI
<!-- Google drive-->
### Financial Analyst Agent (Agentic AI)
- Developed a financial analyst to do sentiment and trend analysis on the market data for a provided ticker.
1. Used crewai for orchestration.
2. Built 2 agents, one to act as a Financial News Analyst, whose role is to collect and analyze financial news article to identify sentiment (and).
3. Another agent to act as a Data Analyst, whose role is to analyze historical market data to identify trends.
<!-- Refer to TechNotes git's LLM/Agentic AI/BuildingAIAgentswithLangChain/notebooks/Module5/M5_Build_a_Financial_Analyst_ReAct_Agentic_AI_System_with_LangChain.ipynb-->
### Financial Analyst Agent (ReAct Based Agent)
Created an updated version of the above and converted it to a ReAct based agent with 2 flows. One to get information about specific stocks and another flow to get general market information. Used GPT-4o, which helps with function calling and LangChain/LangGraph

The tools built are listed below. Used OpenBB to integrate with different data providers to get required data.
- GET_STOCK_TICKER: Validates and fetches stock tickers based on user queries. Integrated with OpenBB to get this data from SEC. This integration also helps convert say NVIDIA to NVDA.
- GET_STOCK_PRICE_METRICS: Retrieves current price, historical price and performance data for specific stocks. This was also integrated with OpenBB and used CBOE to get price data, finviz to get performance data and yfinance for historical price.
- GET_STOCK_NEWS: Extracts recent news articles related to stocks or markets. Used tmx provider via OpenBB to get this information.
- GET_STOCK_FUNDAMENTAL_INDICATOR_METRICS: Provides insights into financial indicators like P/E ratio, ROE, etc. Integrated with OpenBB's fundamental and ratios metrics
- GET_GENERAL_MARKET_DATA: Fetches general market trends and data for the whole market such as most actively traded stocks based on volume, top gainers and top losers. Used yfinance to get the data.

This will help answer questions such as 
1. How is the market doing today?
2. Is it the right time to invest in NVDA?

The key benefit of an agentic financial analyst is to provide quick access to multi-source financial data and insights.

<!-- Refer to TechNotes git's LLM/Agentic AI/4.6 Assignment -->
### Study Assistant
- Developed a Study Assistant using LangChain that:
1. Summarizes study material into concise points.
2. Automatically generates multiple-choice quiz questions based on the summarized content.
3. Functions without the need for external retrieval mechanisms or vector database.
   Used LangChain's ChatPromptTemplate and LangChain Expression Language to chain the prompt, LLM and the output parser.
	
## LLM Projects - Fine Tuning
<!-- Kaggle -->
### Sentiment Analysis for Financial News
- Built a supervised fine-tuned (SFT) model to predict sentiment of the news article and recevied a **gold star** for the [notebook](https://www.kaggle.com/code/ravitee/sentiment-analysis-on-financial-news-using-llama2/notebook) I published in Kaggle.
1. Used accelerate, peft, bitsandbytes, transformers and trl libraries to fine tune a llama-2 base model.
2. Used float16 data type for computations and retrained the non-linear layer of the base model.

<!-- Kaggle -->
### LLM Prediction Model
- Built a multi-class classification model (SFT) to predict the LLM that produced the test data. Dataset can be found in [Kaggle](https://www.kaggle.com/competitions/h2oai-predict-the-llm)
1. Used accelerate, peft, bitsandbytes, transformers to fine tune Mistral-7B.
2. Used Weights & Biases for experiment tracking.
3. Used Lora to target re-training of attention blocks.
4. Used k-bit training, which reduces memory footprint, to improve training speed


## Data Science Projects

### Datasets Practiced from Kaggle
1. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) &nbsp;&nbsp;&nbsp;&nbsp; 	[#imbalanced]() [#BinaryClassification]()
2. [House Price Prediction](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) &nbsp;&nbsp;&nbsp;&nbsp;	[#Regression]()
3. [Quora Question Pair](https://www.kaggle.com/competitions/quora-question-pairs) &nbsp;&nbsp;&nbsp;&nbsp; [#NLP]() [#Regression]()
4. [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) &nbsp;&nbsp;&nbsp;&nbsp;[#RecommenderSystem]() [#MultiClassClassification]()
5. [Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/overview) &nbsp;&nbsp;&nbsp;&nbsp;[#NLP]() [#Regression]()
6. [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) &nbsp;&nbsp;&nbsp;&nbsp;[#Regression]()
7. [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) &nbsp;&nbsp;&nbsp;&nbsp;[#NLP]() [#BinaryClassification]()
8. [Forecasting Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1) &nbsp;&nbsp;&nbsp;&nbsp;[#TimeSeries]() [#Regression]()
9. [Steel Plate Defect Prediction](https://www.kaggle.com/competitions/playground-series-s4e3) &nbsp;&nbsp;&nbsp;&nbsp;[#MultiClassClassiification]()

# Published Articles
1. [Santander Product Recommendation](https://medium.com/@ravitee/santander-product-recommendation-ee4122d15072)
2. [Portfolio Optimization using Principal Component Analysis](https://medium.com/@ravitee/portfolio-optimization-using-principal-component-analysis-923f102a8a47)
3. [A Primer on AI Agents](https://medium.com/@ravitee/a-primer-about-ai-agents-1e34f6dc7a4d)

# Certifications
1. AWS Certified Machine Learning - Specialty
2. AWS Certified Cloud Practitioner

# Contact
ravi.thothadri@hotmail.com