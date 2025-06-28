from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config import config

class LLMModel:
    def __init__(self):
        self.llm = None
        self.max_len = 2048  

    def load_model(self):
        if config.LLM_PROVIDER.lower() == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=config.MODEL_NAME,
                temperature=config.TEMPERATURE,
                google_api_key=config.GEMINI_API_KEY,
                max_output_tokens=self.max_len,
                streaming=True
            )
        else:
            self.llm = ChatOpenAI(
                model_name=config.MODEL_NAME,
                temperature=config.TEMPERATURE,
                openai_api_key=config.OPENAI_API_KEY,
                max_tokens=self.max_len,
                streaming=True
            )
        
        return self.llm

    def get_llm(self):
        if self.llm is None:
            self.load_model()
        return self.llm