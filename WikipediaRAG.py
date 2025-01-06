import together
import pymongo
import AIClasses
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import wikipedia
from typing import List
import time

with open('mongotoken.txt','r') as f:
    client = pymongo.MongoClient(f.read())

#Query is fed into mistral to generate a hypothetical wikipedia article name. It is then inputted into 
print('Initialize')
query = "What is Alan Turing's Contributions to Computer Science and cryptography?"
BasePrompt = "You are Jean Crouton, an AI Bot who takes in a question and only returns the shortest hypothetical wikipedia title or topic, you do not answer the question. E.g. <User> What is the size of the Effiel Tower? </User> <AI> Effiel Tower </AI>"
settings = AIClasses.settings(stop=["[INST]", '\n','[/INST]'])
wikiname= AIClasses.AIResponse(BasePrompt, query, settings)['output']['choices'][0]['text']
result =  wikipedia.search(wikiname, results=1, suggestion=False)[0]
print(result)

def get_database():
   CONNECTION_STRING = open('mongoURI.txt',"r").read()
   client = pymongo.MongoClient(CONNECTION_STRING)
   return client['TestingRAG']

dbname = get_database()
collection_name = dbname["Wikipedia3"]

#Creating a list of metadata and isolating the article names to find if article is in database
x = collection_name.distinct("metadata")
EnteredArticles = []
for i in x:
    if i['articlename'] not in EnteredArticles:
        EnteredArticles.append(i['articlename'])

#Embedding info and function
embedding_model_string = 'WhereIsAI/UAE-Large-V1' # model API string from Together.
vector_database_field_name = 'embedded' # define your embedding field name.

def generate_embedding(input_texts: List[str], model_api_string: str) -> List[List[float]]:
    together_client = together.Together()
    outputs = together_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return [x.embedding for x in outputs.data]

if result not in EnteredArticles:
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    print('Scraping Data')
    ArticleName = result
    wikicontent = wikipedia.page(title =ArticleName,auto_suggest=False).content.split("== Notes ==")[0]
    wikicontent = re.split("\n\n\n=== |\n\n\n== |\n\n\n==== ",wikicontent)
    finaldata = []
    metadata = []
    cnt = 0
    index = 0
    for i in range(len(wikicontent)):
        #delete unnecessary information
        if len(wikicontent[i-cnt])<200:
            wikicontent.pop(i-cnt)
            cnt+=1
        else:
            index += 1
            if i == 0:
                topic = "Summary"
            else:
                topic = re.split("==\n| ===\n| ====\n|, ", wikicontent[i-cnt])[0]
            
            texts = text_splitter.create_documents([wikicontent[i-cnt]])
            for text in texts:
                if len(text.page_content) > 100:
                    finaldata.append(text.page_content)
                    metadatadict = {
                        'articlename' : ArticleName ,
                        'topic' : topic,
                        'index' : index,
                    }
                    print(f'{len(text.page_content)} and {metadatadict}')
                    metadata.append(metadatadict)

    for i in range(len(finaldata)):
        tempdict = {
            'original_data': finaldata[i],
            vector_database_field_name : generate_embedding(finaldata[i], embedding_model_string),
            'metadata' : metadata[i]
        }
        collection_name.insert_one(tempdict)
        time.sleep(0.5)

#generate embedding for the query and finding 3 most relevant results
""" vector_index(Used MongoDB Search Index)

{
  "fields": [
    {
      "numDimensions": 1024,
      "path": "embedded",
      "similarity": "dotProduct",
      "type": "vector"
    }
  ]
} """


query_emb = generate_embedding([query], embedding_model_string)[0]
context = ""
for i in collection_name.aggregate([
  {
    "$vectorSearch": {
      "queryVector": query_emb,
      "path": vector_database_field_name,
      "numCandidates": 60, # this should be 10-20x the limit
      "limit": 3, # the number of documents to return in the results
      "index": "vector_index", # the index name you used in Step 4
  }
  }
]):
  context += f'{i["original_data"]} \n'

#feeding it into an LLM to return with a relevant answer
BasePrompt = "[INST]You are Jean Crouton, a helpful and succint AI Bot who answers questions with access to additional context. The context marked as <context> and the question will be marked as <q>[/INST]"
settings1 = AIClasses.settings()
HumanPrompt1 = f'<q>{query}</q> \n <context>{context}</context>'
response = AIClasses.AIResponse(BasePrompt, HumanPrompt1, settings1)['output']['choices'][0]['text']

print(response)



