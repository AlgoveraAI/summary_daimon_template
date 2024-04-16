import logging
import requests
from pathlib import Path
from fake_useragent import UserAgent
from readability import Document
from markdownify import markdownify as md
from create_summary.schemas import InputSchema

OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
DEFAULT_FILENAME = "summary.txt"
DEFAULT_MODEL = 'mistral'


SUMMARIZE_PROMPT = """
Summarize the following document:

{document}
"""

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = get_logger()

def scrap_url(url: str):
    user_agent = UserAgent()

    headers = {
        'User-Agent': user_agent.random
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    doc = Document(response.text).summary()
    markdown = md(doc)

    return markdown

def run(job: InputSchema):
    try:
        logger.info(f"Running job with url: {job.url}")

        # get the url
        markdown = scrap_url(job.url)
        prompt = SUMMARIZE_PROMPT.format(document=markdown)

        data = {
            'model': job.ollama_model or DEFAULT_MODEL,
            'prompt': prompt,
            'stream': False
        }

        response = requests.post(
            OLLAMA_ENDPOINT,
            json=data
        )
        response.raise_for_status()
        response_json = response.json()

        summary = response_json['response']

        logger.info(f"Summary: {summary}")

        if job.output_path is not None:
            output_path = Path(job.output_path) / DEFAULT_FILENAME
            output_path.write_text(summary)

        return summary
    
    except Exception as e:
        print(e)


if __name__ == "__main__":
    from create_summary.schemas import InputSchema
    inp = InputSchema(
        url="https://www.twosigma.com/articles/a-guide-to-large-language-model-abstractions/",
        output_path=".",
        ollama_model="mistral:latest"
    )

    run(inp)