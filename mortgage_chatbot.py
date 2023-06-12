#!pip install flask_restful, flask_cors
#!pip install langchain, PyPDF2, faiss-cpu, tiktoken,sentence_transformers
#!pip install -qU transformers accelerate einops langchain wikipedia xformers

#first setup LLM
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Split the extracted text into chunks so we can avoid hitting the token limit of the LLM 

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)# Split the extracted text into chunks so we can avoid hitting the token limit of the LLM 

from torch import cuda, bfloat16
import transformers

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

model = transformers.AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    trust_remote_code=True,
    torch_dtype=bfloat16,
    max_seq_len=2048
)
model.eval()
model.to(device)

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# mtp-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=64,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}"
)

llm = HuggingFacePipeline(pipeline=generate_text)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Load our Q+A chain
from langchain.chains.question_answering import load_qa_chain
chain2 = load_qa_chain(llm=llm, chain_type="stuff")

#then setup Flask

from flask import Flask, request
from flask_restful import Api, Resource
from flask_cors import CORS, cross_origin

import os
import threading

class WebPage():
	url = ''
	content = ''
	title = ''
	
class QA():
	question = ''
	answer = ''
	references = []

allDocs = []
docsearch = None

class IngestDocument(Resource):
    response = {"status":200, "message":"success"}
    def addContent(self,content):
        global docsearch;
        texts = text_splitter.split_text(content)
        print("split to chunks", len(texts))
        if(docsearch == None):
          print("create new docsearch")
          docsearch = FAISS.from_texts(texts, hf)
        else:
          print("add to docsearch")
          docsearch.add_texts(texts)
        print("add content success")


    def post(self):
        global allDocs;
        page_data = request.get_json()
        url = page_data['url']
        content = page_data['content']
        title = page_data['title']

        doc = WebPage()
        doc.url = url
        doc.content = content
        doc.title = title
        allDocs.append(doc)
        print("now add content to FAISS", title)
        self.addContent(content)

        self.response['status'] = 200
        self.response['message'] = 'success'
        return self.response, 200
    
class AskQuestion(Resource):
    response = {"status":200}

    def searchRefDocs(self, question):
        if(docsearch==None):
          docs=[]
        else:  
          docs = docsearch.similarity_search(question)
        
        #docs = ["docs1","doc2"]
        return docs
	
    def askQuestion(self, refdocs, question):
        answer = chain2.run(input_documents=refdocs, question=question)
        #answer="test answer"
        return answer;
		
    def post(self):
        question_data = request.get_json()
        question = question_data['question']
		
        refdocs = self.searchRefDocs(question)
        print("searchRefDocs found", len(refdocs))
        answer = self.askQuestion(refdocs, question)
        print("askQuestion returns", answer)

        q_response = {
          "question": question,
          "answer": answer,
          "refdocs": []
        }
        self.response['status']=200
        self.response['message']= q_response
        return self.response, 200

class DocumentList(Resource):
    response = {"status":200}
    def title(self,x):
        return x.title;

    def get(self):
        titles =   list(map(self.title, allDocs))

        self.response['status']=200
        self.response['message']= titles
        return self.response, 200

class ResetDocumentList(Resource):
    response = {"status":200}
    def title(self,x):
        return x.title;

    def get(self):
        global allDocs;
        global docsearch;
        allDocs=[]
        docsearch=None

        self.response['status']=200
        self.response['message']= "success"
        return self.response, 200

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}
    def post(self):
        return {'hello': 'post world'}
                    
app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

api.add_resource(IngestDocument, '/api/ingest')
api.add_resource(AskQuestion, '/api/qa')
api.add_resource(HelloWorld, '/api/hello')
api.add_resource(DocumentList, '/api/docs')
api.add_resource(ResetDocumentList, '/api/resetdocs')

if __name__ == '__main__':
    app.run(debug=True)