from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI as LangchainOpenAI

def generate_summary(transcript, query):
    llm = LangchainOpenAI(temperature=0.5, max_tokens=300)
    template = """
    Summarize the following meeting transcript:
    Transcript: {transcript}
    Query: {query}
    """
    prompt = PromptTemplate(input_variables=["transcript", "query"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(transcript=transcript, query=query)
