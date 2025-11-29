from src import Config, SYSTEM_PROMPT, ReportOutput
from llama_index.core import Settings, SummaryIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class QueryEngineBuilder:
    """
    Builds and manages query engines for document analysis
    
    Handles:
    - LLM and embedding model setup
    - Query engine configuration
    - Structured output generation
    """
    def __init__(self, config: Config):
        """
        Initialize query engine builder
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.query_engine = None
        self._setup_models()
        logger.info("QueryEngineBuilder initialized")

    def _setup_models(self) -> None:
        """Setup LLM and embedding models as global defaults"""
        try:
            # Initialize embedding model
            self.embed_model = OpenAIEmbedding(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
            
            # Initialize LLM
            self.llm = OpenAI(
                model=self.config.llm_model,
                api_key=self.config.openai_api_key,
                temperature=self.config.temperature,
            )
            
            # Register as global defaults for LlamaIndex
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            logger.info(f"Models configured: {self.config.llm_model}, {self.config.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to setup models: {e}", exc_info=True)
            raise
    
    def create_query_engine(self, index: SummaryIndex, system_prompt: Optional[str] = None,
                            similarity_top_k: Optional[int] = None,
                            response_mode: Optional[str] = None):
        """
        Create query engine with structured output
        
        Args:
            index: The LlamaIndex summary index
            system_prompt: Custom system prompt (uses default if None)
            similarity_top_k: Number of chunks to retrieve (uses config default if None)
            response_mode: Response synthesis mode (uses config default if None)
            
        Returns:
            Query engine configured for structured output
        """
        try:
            # Use defaults from config if not provided
            system_prompt = system_prompt or SYSTEM_PROMPT
            similarity_top_k = similarity_top_k or self.config.similarity_top_k
            response_mode = response_mode or self.config.response_mode
            
            # Create LLM with system prompt and structured output
            llm_with_prompt = OpenAI(
                model=self.config.llm_model,
                api_key=self.config.openai_api_key,
                temperature=self.config.temperature,
                system_prompt=system_prompt
            )
            
            # Convert to structured LLM
            structured_llm = llm_with_prompt.as_structured_llm(
                output_cls=ReportOutput
            )
            
            # Create query engine
            self.query_engine = index.as_query_engine(
                similarity_top_k=similarity_top_k,
                llm=structured_llm,
                response_mode=response_mode,
            )
            
            logger.info(
                f"Query engine created with top_k={similarity_top_k}, "
                f"mode={response_mode}"
            )
            
            return self.query_engine
            
        except Exception as e:
            logger.error(f"Failed to create query engine: {e}", exc_info=True)
            raise
      
    def query(self, query_text: str, verbose: bool = False) -> Optional[ReportOutput]:
        """
        Execute a query        
        Args:
            query_text: The question to ask
            verbose: If True, log detailed query info
        Returns:
            ReportOutput or None if query fails
        """
        if self.query_engine is None:
            logger.error("Query engine not created. Call create_query_engine() first.")
            return None
        
        try:
            if verbose:
                logger.info(f"Executing query: {query_text}")
            
            response = self.query_engine.query(query_text)
            
            if verbose:
                logger.info(f"Query successful, got {len(response.source_nodes)} source nodes")
            
            return response.response
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return None
    
    def get_query_engine(self):
        """Get the current query engine instance"""
        return self.query_engine

# Convenience function for simple usage
def create_simple_query_engine(config: Config, index: SummaryIndex):
    """
    Create a query engine with default settings
    
    Args:
        config: Application configuration
        index: Summary index
        
    Returns:
        Configured query engine
    """
    builder = QueryEngineBuilder(config)
    return builder.create_query_engine(index)