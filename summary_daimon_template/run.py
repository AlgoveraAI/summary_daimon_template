import logging
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from summary_daimon_template.schemas import InputSchema
from pathlib import Path

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = get_logger()
DEFAULT_FILENAME = "summary.txt"

def run(job: InputSchema):
    logger.info(f"Running job with url: {job.url}")
    loader = WebBaseLoader(job.url)
    docs = loader.load()

    llm = Ollama(model=job.ollama_model)
    chain = load_summarize_chain(llm, chain_type="stuff")

    result = chain.invoke(docs)
    logger.info(f"Summary: {result['output_text']}")

    if job.output_path is not None:
        output_path = Path(job.output_path) / DEFAULT_FILENAME
        output_path.write_text(result['output_text'])

    return result


if __name__ == "__main__":
    run(
        InputSchema(
            url="https://www.twosigma.com/articles/a-guide-to-large-language-model-abstractions/",
            output_path="."
            )
        )