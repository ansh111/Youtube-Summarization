import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from dotenv import load_dotenv
import nltk
from yt_dlp import YoutubeDL
from langchain.schema import Document
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv()

## sstreamlit APP 
st.set_page_config(page_title="Langchain: Summerize Text From YT or Website",page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


## Get the Groq API Key and url(YT or website)to be summarized
#groq_api_key=st.text_input("Groq API key", value=st.secrets["GROQ_API_KEY"], type="password")
with st.sidebar:
   groq_api_key=st.text_input("Groq API key", value="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", type="password")

generic_url=st.text_input("URL", label_visibility= "collapsed")
## Gemma model
#llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
llm =ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)


def load_youtube_content(url):
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Video")
        description = info.get("description", "No description available.")
        return f"{title}\n\n{description}"

prompt_template="""
Provide a summary of the following content in points having 500 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")
    else:
        try:
            with st.spinner("Waiting.."):
                if "youtube.com" in generic_url:
                     #loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                     text_content = load_youtube_content(generic_url)
                     docs = [Document(page_content=text_content)]
               
                else:
                    loader= UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})

                    docs=loader.load()  
                    print(docs)
            
                ##chain for summerisation
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output=chain.run(docs)

                st.success(output)   
        except Exception as e:
            st.exception(f"Exception:{e}")







