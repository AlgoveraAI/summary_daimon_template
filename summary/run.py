import logging
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from summary.schemas import InputSchema

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = get_logger()


def run(job: InputSchema):
    logger.info(f"Running job with url: {job.url}")
    loader = WebBaseLoader(job.url)
    docs = loader.load()

    llm = Ollama(model=job.ollama_model)
    chain = load_summarize_chain(llm, chain_type="stuff")

    result = chain.invoke(docs)
    logger.info(f"Summary: {result['output_text']}")

    if job.output_path is not None:
        with open(job.output_path, "w") as f:
            f.write(result["output_text"])

    return result


if __name__ == "__main__":
    run(
        InputSchema(
            url="https://www.twosigma.com/articles/a-guide-to-large-language-model-abstractions/",
            output_path="summary.txt"
            )
        )