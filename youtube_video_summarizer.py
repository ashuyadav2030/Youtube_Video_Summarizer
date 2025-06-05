from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from fastapi import FastAPI, Query

app = FastAPI()



load_dotenv()

@app.post("/summarizer")
async def vid_summarizer(str= Query(..., description="YouTube Video ID")):
    """
    Summarize a YouTube video transcript.
    Provide the YouTube video ID as a query parameter.
    """
    video_id = str 
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print("Transcript fetched successfully!")

    except TranscriptsDisabled:
        print("No captions available for this video.")


    model = ChatGoogleGenerativeAI(
        model= "gemini-2.0-flash",
        temperature=0.2,
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    result = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(result, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables = ['context', 'question']
    )

    retrieved_docs   = retriever.invoke('question')

    def format_docs(retrieved_docs):
      context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
      return context_text

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | model | parser

    result1 = main_chain.invoke('Summarize this video in detail, including all the important points and key takeaways. Make sure to include all the important details and key points in the summary.')
    print(result1)
    return {"summary": result1}








