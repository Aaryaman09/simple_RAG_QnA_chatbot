
import json, os

def track_queries_on_langsmith(flag: bool = False):
    if flag:
        pass_keys = get_config("key.json")
        os.environ["LANGCHAIN_API_KEY"] = pass_keys["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_TRACING_V2"] = pass_keys["LANGCHAIN_TRACING_V2"]
        os.environ["LANGCHAIN_PROJECT"] = pass_keys["LANGCHAIN_PROJECT_NAME"]

def get_config(path: str) -> dict:
    """
    Load configuration from a JSON file.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config
