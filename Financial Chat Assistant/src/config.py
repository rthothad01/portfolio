import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv

def read_env_file() -> bool:
    load_dotenv()
    dotenv_path = find_dotenv()
    if dotenv_path:
        print(f"Found .env file at: {dotenv_path}")
        loaded = load_dotenv(dotenv_path, verbose=True)
        if loaded:
            print("load_dotenv() successfully processed the file.")
        else:
        # This case is rare if find_dotenv() succeeded, but covers edge cases
            # print("load_dotenv() could not process the file content.")
            raise ValueError(
                    "load_dotenv() could not process the file content. "
                    "Please check the .env file for formatting issues."
                )
        return loaded
    else:
        # print("No .env file found by find_dotenv().")
        raise ValueError(
                "No .env file found by find_dotenv(). "
                "Please ensure the .env file exists in the project directory."
            )

if read_env_file():
    print(".env file loaded successfully.")
else:
    print("Warning: Could not load .env file. Please ensure it exists.")
    exit(1)

@dataclass
class Config:
    """
    Centralized configuration for Financial Chat Assistant
    
    Manages all configuration parameters including API keys, paths,
    model settings, and processing flags.
    """
    
    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    llamaparse_api_key: str = field(default_factory=lambda: os.getenv('LLAMAPARSE_API_KEY', ''))
    phoenix_api_key: Optional[str] = field(default_factory=lambda: os.getenv('PHOENIX_API_KEY'))
    
    # Directories
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    images_dir: Path = field(default_factory=lambda: Path("data_images"))
    storage_dir: Path = field(default_factory=lambda: Path("storage_nodes_summary"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    
    # Model Configuration
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    llm_model: str = "gpt-4o"  
    temperature: float = 0.0
    
    # Query Engine Settings
    similarity_top_k: int = 10
    response_mode: str = "compact"  # Options: "compact", "tree_summarize"
    
    # LlamaParse Settings
    result_type: str = "markdown"  # Options: "markdown", "text"
    gpt4o_mode: bool = True
    
    # Processing Flags
    reuse_existing_data: bool = True
    verbose: bool = True
    
    # Cost Tracking
    track_costs: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        if not self.llamaparse_api_key:
            raise ValueError(
                "LLAMAPARSE_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        # Ensure model name is correct
        if self.llm_model == "gpt-40":
            raise ValueError(
                f"Invalid model name: {self.llm_model}. Did you mean 'gpt-4o'?"
            )
    
    @property
    def pdf_path(self) -> Path:
        """Get the default PDF file path"""
        return self.data_dir / "rjf1q25.pdf"
    
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_dir,
            self.images_dir,
            self.cache_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def clear_cache(self) -> None:
        """Clear all cached data"""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
    
    def clear_storage(self) -> None:
        """Clear persisted index storage"""
        import shutil
        
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
    
    def clear_images(self) -> None:
        """Clear downloaded images"""
        import shutil
        
        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)
            self.images_dir.mkdir()
    
    def get_summary(self) -> dict:
        """Get configuration summary"""
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "llm_model": self.llm_model,
            "similarity_top_k": self.similarity_top_k,
            "response_mode": self.response_mode,
            "data_dir": str(self.data_dir),
            "storage_exists": self.storage_dir.exists(),
        }


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = Config()
    config.setup_directories()
    
    print("Configuration Summary:")
    print("-" * 50)
    for key, value in config.get_summary().items():
        print(f"{key:25s}: {value}")
    
    print("\nDirectories created successfully!")
