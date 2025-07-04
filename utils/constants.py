import os
from pathlib import Path

import yaml

def get_root_path():
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_PATH = get_root_path()
CONFIG_PATH = ROOT_PATH / "configs"

SERDESER_PATH = ROOT_PATH / "storage"  # TODO to store `storage` under the individual generated project

MESSAGE_ROUTE_FROM = "sent_from"
MESSAGE_ROUTE_TO = "send_to"
MESSAGE_ROUTE_CAUSE_BY = "cause_by"
MESSAGE_META_ROLE = "role"
MESSAGE_ROUTE_TO_ALL = "<all>"
MESSAGE_ROUTE_TO_NONE = "<none>"
MESSAGE_ROUTE_TO_SELF = "<self>"

MARKDOWN_TITLE_PREFIX = "## "

USE_CONFIG_TIMEOUT = 0  # Using llm.timeout configuration.
LLM_API_TIMEOUT = 300

AGENT = "agent"
IMAGES = "images"
AUDIO = "audio"

IGNORED_MESSAGE_ID = "0"

config = yaml.load(open(f"{CONFIG_PATH}/config.yaml", "r"), Loader=yaml.FullLoader)

COMPLETION_API_KEY = config["llm"]["api_key"]
COMPLETION_BASE_URL = config["llm"]["base_url"]
COMPLETION_MODEL_NAME = config["llm"]["model"]
COMPLETION_API_TYPE = config["llm"]["api_type"]

EMBEDDING_BASE_URL = config['embedding']['base_url']
EMBEDDING_API_KEY = config['embedding']['api_key']
EMBEDDING_MODEL_NAME = config['embedding']['model']
EMBEDDING_API_TYPE = config['embedding']['api_type']

LOCAL_INFERENCE_ENDPOINT_HOST = config["local_inference_endpoint"]["host"]
LOCAL_INFERENCE_ENDPOINT_PORT = config["local_inference_endpoint"]["port"]

LOCAL_INFERENCE_ENDPOINT_URL = f"http://{LOCAL_INFERENCE_ENDPOINT_HOST}:{LOCAL_INFERENCE_ENDPOINT_PORT}"

HUGGINGFACE_HEADERS = {}
if config["huggingface"]["token"] and config["huggingface"]["token"].startswith("hf_"):  # Check for valid huggingface token in config file
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {config['huggingface']['token']}",
    }
elif "HUGGINGFACE_ACCESS_TOKEN" in os.environ and os.getenv("HUGGINGFACE_ACCESS_TOKEN").startswith("hf_"):  # Check for environment variable HUGGINGFACE_ACCESS_TOKEN
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_ACCESS_TOKEN')}",
    }
else:
    raise ValueError(f"Incorrect HuggingFace token. Please check your config.yaml or .env file.")

PROXY = None
if config["proxy"]:
    PROXY = {
        "https": config["proxy"],
    }