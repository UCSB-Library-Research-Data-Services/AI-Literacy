from dotenv import load_dotenv
import os

load_dotenv()

SERVICES = {
    "dreamlab": {
        "url": "https://litellm.dreamlab.ucsb.edu",
        "key": os.getenv("DL_KEY")
        },
    "CIT": {
        "url": "https://api.ai.college.ucsb.edu",
        "key": os.getenv("CIT_KEY")
        },
    "GRIT": {
        "url": "https://llm.grit.ucsb.edu/api/v1",
        "key": os.getenv("GRIT_KEY")
        },
    "AICommons":{
        "url": "https://zkh52rh785.execute-api.us-east-1.amazonaws.com/v1",
        "key": os.getenv("AICOMMONS_KEY")
        }
}