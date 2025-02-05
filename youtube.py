import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from dotenv import load_dotenv
import nltk
from yt_dlp import YoutubeDL
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
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
   st.title("Configurations") 
   groq_api_key=st.text_input("Groq API key", value="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", type="password")
   words = st.sidebar.slider("Choose number of words for summary", 0, 1000, 500)
   # Language selection dropdown in the sidebar
   language_options = ["English", "Portuguese", "Hindi", "German", "Chinese"]
   selected_language = st.sidebar.selectbox("Select Summary Language:", language_options, index=0)  # Default is "English"

   # Display selected language in the main UI
   st.write(f"### Selected Language: {selected_language}")

   



generic_url=st.text_input("URL", label_visibility= "collapsed")
## Gemma model
llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

    
def extract_video_id(url):
    # Regular expression to match YouTube video ID
    pattern = r'(?:youtube\.com\/(?:v\/|watch\?v=|live\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    
    # Search for the video ID in the URL
    match = re.search(pattern, url)
    if match:
        return match.group(1)  # Return the video ID
    else:
        return None  # Return None if no match is found
    
def get_video_transcript(video_id):
        # Fetch the transcript for the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=['hi','en','pt'])

         # Convert the transcript to a string format (plain text)
        transcript_text = ""
        for entry in transcript:
            transcript_text += f"{entry['text']} "  # Combine all caption text with a space
        
        return transcript_text.strip()  # Remove any trailing spaces  


def get_summarization_with_map_reduce(docs, selected_language="English", words="500"):
    try:
        final_doc = RecursiveCharacterTextSplitter(chunk_size=6000,chunk_overlap=100).split_documents(docs)
        print(final_doc)
        chunks_prompt=f"""
            Please summarize the below speech in {selected_language} in  {words}:
            Speech:`{{text}}'
            Summary:
            """
        map_prompt_template=PromptTemplate(input_variables=['text'],
                                    template=chunks_prompt) 
        prompt_template=f"""
        Provide a summary of the following content in points having {words} words in {selected_language}:
        Content:{{text}}

        """  
        
        final_prompt_template=PromptTemplate(input_variables=['text'],template=prompt_template)
        summary_chain = load_summarize_chain(
            llm = llm,
            chain_type = "map_reduce",
            map_prompt = map_prompt_template,
            combine_prompt = final_prompt_template,
            verbose=True
         )
        return summary_chain.run(final_doc)
    except Exception as e:
        print(f"Error: {e}")  


def get_summarization_with_stuff(docs, selected_language="English"):
    try:
        prompt_template="""
        Provide a summary of the following content in points having 500 words in  {language}:
        Content:{text}

        """
        prompt=PromptTemplate(template=prompt_template,input_variables=["text","language"])    
        chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
        return chain.invoke(docs,selected_language)
    except Exception as e:
        print(f"Error: {e}")   


def count_number_of_tokens(text):
    # A rough estimation of token count (1 token ≈ 4 characters, you can adjust as needed)
    return len(text)


if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")
    else:
        try:
            with st.spinner("Waiting.."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = extract_video_id(generic_url)
                    text_content= get_video_transcript(video_id=video_id)
                    docs = [Document(page_content=text_content)]
            
                else:
                    loader= UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()  
            
                if docs[0].page_content is None:
                    docs[0].page_content = ""

                ##chain for summerisation
                
                if(count_number_of_tokens(docs[0].page_content) < 6000):
                   output = get_summarization_with_stuff(docs,selected_language,words)
                   print("stuff")
                else:
                    output =get_summarization_with_map_reduce(docs,selected_language,words)
                    print("map_reduce")

                st.success(output)   
        except Exception as e:
            st.exception(f"Exception:{e}")







