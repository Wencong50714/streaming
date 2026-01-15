from src.memory.memory import Memory
from src.router import Router

class Model:
    def __init__(self, model_path):
        self.memory = Memory()
        self.router = Router()
        pass

    def _video_preprocess(self, video_file_name):
        pass

    def inference(self, video_file_name, prompt) -> str:
        pass
