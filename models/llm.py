from langchain_openai import ChatOpenAI
from config import config

class LLMModel:
    def __init__(self):
        self.llm = None
        self.max_len = 2048  

    def load_model(self):
        # print(f'\nLoading model: gpt-4\n')
        
        # Create OpenAI LangChain model
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