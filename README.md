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


# Contact
ravi.thothadri@hotmail.com