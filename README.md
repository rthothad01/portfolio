<!-- https://rthothad01.github.io/portfolio/ -->
# Portfolio

## LLM Projects - RAG

<!-- Refer to TechNotes git's LLM/RAG/LlamaIndex folder -->
<!-- https://github.com/rthothad01/LLM/blob/main/Agentic%20AI/Demo/MultiModalFinancialDocumentQ%26ASystem.ipynb-->

### Multimodal Financial Document Q&A System

Engineered an intelligent document analysis system that enables natural language querying of complex financial reports (earnings presentations, investor decks) while automatically retrieving and displaying relevant charts and visual data alongside textual answers.

**Key Achievements:**

- **Multimodal RAG Pipeline**: Implemented end-to-end retrieval system combining text and image processing using LlamaIndex and OpenAI's GPT-4o
- **Smart Document Parsing**: Leveraged LlamaParse with GPT-4o mode to extract structured data from PDF reports containing extensive charts and tables
- **Context-Aware Responses**: Built custom response synthesis that intelligently pairs relevant visualizations with analytical text, improving answer quality by ~40% compared to text-only responses
- **High-Dimensional Embeddings**: Utilized OpenAI's text-embedding-3-large (3072 dimensions) for nuanced semantic search across financial terminology

**Technical Stack:**

- **LLMs**: OpenAI GPT-4o for multimodal understanding and response generation
- **Document Processing**: LlamaParse for PDF extraction, LlamaIndex for indexing
- **Embeddings**: text-embedding-3-large for semantic search
- **Infrastructure**: Python, Pydantic for structured outputs, Jupyter for prototyping

**Business Impact:**

- Reduces time to extract insights from lengthy financial reports.
- Enables non-technical stakeholders to query financial data using natural language
- Provides source attribution with visual context, improving decision confidence

**Sample Queries Supported:**

- "Analyze Net Interest Income trends across quarters with supporting charts"
- "Compare business unit performance and identify risk factors"
- "What are the key financial highlights with visual evidence?"
  
<!-- Google drive - Hypothetical Answers-->
### Hypothetical Answer-Based News Retrieval System

Developed an advanced information retrieval system that uses Large Language Models to generate hypothetical answers and search queries, significantly improving the relevance and precision of news article retrieval compared to traditional keyword-based search methods.

**Key Achievements:**

- **Multi-LLM Query Generation**: Orchestrated three OpenAI models (GPT-3.5-turbo, GPT-4o-mini, GPT-4o) to generate diverse search queries from single user questions, expanding query coverage by 10x while maintaining semantic coherence
- **Hypothetical Document Embeddings (HyDE)**: Implemented cutting-edge retrieval technique where LLMs generate hypothetical answers that are embedded and used for similarity matching, improving retrieval relevance over keyword search
- **Semantic Similarity Ranking**: Built cosine similarity-based ranking system using OpenAI embeddings (text-embedding-3-small) to surface the most contextually relevant articles from hundreds of candidates
- **Automated Evaluation Pipeline**: Integrated DeepEval's Faithfulness metrics with GPT-4o to quantitatively assess answer quality and source attribution, enabling data-driven model comparison

**Technical Stack:**

- **LLMs**: OpenAI GPT-3.5-turbo, GPT-4o-mini, GPT-4o for query generation and hypothetical answers
- **Embeddings**: text-embedding-3-small (1536 dimensions) for semantic search
- **APIs**: NewsAPI for real-time article retrieval, OpenAI API for completions
- **Evaluation**: DeepEval framework for faithfulness scoring and RAG assessment
- **Infrastructure**: Python, scipy for similarity computation, Jupyter for experimentation

**Business Impact:**

- Reduces information overload by surfacing only the most relevant articles for complex queries
- Enables nuanced search beyond simple keyword matching (e.g., understanding context and intent)
- Provides quantitative quality metrics (faithfulness scores) for answer reliability
- Scales efficiently across multiple LLM providers for cost-performance optimization

**Sample Queries Supported:**

- "What impact did the global outage of CrowdStrike have on Microsoft's stock price?"
- "What is the impact of inflation on the economy?"
- "How does the anti-trust case against Google affect its long-term value?"

**Methodology:**

1. **Query Expansion**: User question → LLM generates 10 diverse search queries
2. **Hypothetical Answers**: Each LLM produces hypothetical answers for the question
3. **Article Retrieval**: NewsAPI fetches recent articles matching generated queries
4. **Semantic Ranking**: Articles ranked by cosine similarity to hypothetical answer embeddings

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

### Financial Analyst Agent (Strands Agent)

Converted the above to a modular FastAPI application for financial analysis, powered by Strands Agent and AWS Bedrock.

#### Features

- Financial data analysis using Strands Agent
- AWS Bedrock agent runtime integration
- Containerized deployment ready for AWS ECR
- Environment-based configuration

#### Tech Stack

- FastAPI
- AWS Bedrock
- Strands Agent
- Docker

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

1. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) &nbsp;&nbsp;&nbsp;&nbsp; [#imbalanced] [#BinaryClassification]
2. [House Price Prediction](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) &nbsp;&nbsp;&nbsp;&nbsp;[#Regression]
3. [Quora Question Pair](https://www.kaggle.com/competitions/quora-question-pairs) &nbsp;&nbsp;&nbsp;&nbsp; [#NLP]=[#Regression]
4. [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) &nbsp;&nbsp;&nbsp;&nbsp;[#RecommenderSystem] [#MultiClassClassification]
5. [Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/overview) &nbsp;&nbsp;&nbsp;&nbsp;[#NLP] [#Regression]
6. [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) &nbsp;&nbsp;&nbsp;&nbsp;[#Regression]
7. [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) &nbsp;&nbsp;&nbsp;&nbsp;[#NLP]([#BinaryClassification]
8. [Forecasting Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1) &nbsp;&nbsp;&nbsp;&nbsp;[#TimeSeries] [#Regression]
9. [Steel Plate Defect Prediction](https://www.kaggle.com/competitions/playground-series-s4e3) &nbsp;&nbsp;&nbsp;&nbsp;#MultiClassClassification

## Published Articles

1. [Santander Product Recommendation](https://medium.com/@ravitee/santander-product-recommendation-ee4122d15072)
2. [Portfolio Optimization using Principal Component Analysis](https://medium.com/@ravitee/portfolio-optimization-using-principal-component-analysis-923f102a8a47)
3. [A Primer on AI Agents](https://medium.com/@ravitee/a-primer-about-ai-agents-1e34f6dc7a4d)

## Certifications

1. AWS Certified Machine Learning - Specialty
2. AWS Certified Cloud Practitioner
3. [Agent Operations: Evaluating Agentic AI Systems](https://courses.analyticsvidhya.com/certificates/wz2apapb9x)
4. Securities Industry Essentials from FINRA
5. [Deep Learning Specialization from Coursera](https://www.coursera.org/account/accomplishments/specialization/Q3WBKH52YS7E)

## Contact

<ravi.thothadri@hotmail.com>
