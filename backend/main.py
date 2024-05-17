from fastapi import FastAPI, Request, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse, Response
from typing import List, Optional
import json
import os
import threading                                                                
from pydantic import BaseModel
from groq import Groq
import requests
import json
import chromadb
from chromadb.utils import embedding_functions
from unstructured.partition.html import partition_html
from unstructured.documents.elements import NarrativeText, Title, Text
from sse_starlette.sse import EventSourceResponse

groq = Groq(
    api_key="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",
)
app = FastAPI()

search = ['Wikipedia and Facts', 'Social Media: Instagram', 'Social Media: Twitter', 'LinkedIn and People', 'Social Media: Facebook', 'Finance and Crypto', 'Stackoverflow and Coding', 'Where to Watch Movies and TV Shows', 'Reddit or Advice, Opinion, Reviews', 'Glassdoor and Business Reviews', 'Unknown']
shopping = ['Shopping and Purchasing']
news = ['News and Current Events', 'Live Sports Updates', 'Live Weather or forecast', 'Pharma & Healthcare']
videos = ['Video Search']
chitchat = ['Casual Chitchat']
images = ['Image Search']
maps = ['Yelp and Restaurants', 'Places and Attractions']

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def get_internet_content(result, html_content_text):
    elements = partition_html(url=result['link'], headers={"User-Agent": "value"})
    html_content = ""
    for element in elements:
        if 'unstructured.documents.html.HTMLTitle' in str(type(element)):
            html_content += element.text
        elif 'unstructured.documents.html.HTMLNarrativeText' in str(type(element)):
            html_content += '. ' + element.text
    
    html_content = html_content.replace('\n', '. ')
    html_content_text.append({'snippet': html_content, 'link': result['link'], 'title': result['title']})

def scrapeURLData(sources):
    threads = []
    html_content_text = []

    for i in range(len(sources)):
        threads.append(threading.Thread(target=get_internet_content, args=(sources[i], html_content_text)))         
        threads[-1].start()

    for t in threads:                                                           
        t.join()

    return html_content_text

def rerank_search_results(query, results, num):
    #creating chromadb client
    client = chromadb.Client()
    
    #creating chromadb collection
    collection = client.create_collection(
        name="my_collection",
        embedding_function=embedding_function
    )

    #upserting data to collection
    collection.add(
        documents=[doc['snippet'] for doc in results],
        metadatas=[{"title": doc['title']} for doc in results],
        ids=[doc['link'] for doc in results]
    )

    #getting similarity search results
    matching_docs = collection.query(
        query_texts=[query],
        n_results=num
    )

    results = []

    for i in range(len(matching_docs['metadatas'][0])):
        results.append({
            'link': matching_docs['ids'][0][i],
            'title': matching_docs['metadatas'][0][i]['title'],
            'snippet': matching_docs['documents'][0][i]
        })

    client.delete_collection(name="my_collection")

    return results

def getSerperResults(query, url, num):
    
    print(query)

    if num:
        payload = json.dumps({
            "q": query,
            "location": "Mumbai, Maharashtra, India",
            "num": num
        })
    else:
        payload = json.dumps({
            "q": query,
            "location": "Mumbai, Maharashtra, India",
        })
    
    headers = {
        'X-API-KEY': '40bb8f50ccb374957cd3b900808f55e56781be88',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    results = []

    if 'search' in url:
        for sitedata in json.loads(response.text, strict = False)['organic']:
            results.append({'title': sitedata.get('title', 'Data Not Available'), 'link': sitedata.get('link', 'Data Not Available'), 'snippet': sitedata.get('snippet', 'Data Not Available')})
    elif 'news' in url:
        for sitedata in json.loads(response.text, strict = False)['news']:
            results.append({'title': sitedata.get('title', 'Data Not Available'), 'link': sitedata.get('link', 'Data Not Available'), 'snippet': sitedata.get('snippet', 'Data Not Available')})
    elif 'maps' in url:
        for sitedata in json.loads(response.text, strict = False)['places']:
            results.append({'title': sitedata.get('title', 'Data Not Available'), 'description': sitedata.get('description', 'Data Not Available'), 'address': sitedata.get('address', 'Data Not Available'), 'website': sitedata.get('website', 'Data Not Available'), 'rating': sitedata.get('rating', 0), 'type': sitedata.get('type', 'Data Not Available'), 'phone': sitedata.get('phone', 'Data Not Available')})
    elif 'images' in url:
        for sitedata in json.loads(response.text, strict = False)['images']:
            results.append({'title': sitedata.get('title', 'Data Not Available'), 'url': sitedata.get('imageUrl', 'Data Not Available'), 'link': sitedata.get('link', 'Data Not Available')})
    elif 'videos' in url:
        for sitedata in json.loads(response.text, strict = False)['videos']:
            results.append({'title': sitedata.get('title', 'Data Not Available'), 'link': sitedata.get('link', 'Data Not Available'), 'snippet': sitedata.get('snippet', 'Data Not Available')})
    elif 'shopping' in url:
        for sitedata in json.loads(response.text, strict = False)['shopping']:
            results.append({'title': sitedata['title'], 'link': sitedata.get('link', 'Data Not Available'), 'rating': sitedata.get('rating', 0), 'ratingCount': sitedata.get('ratingCount', 0), 'delivery': sitedata.get('delivery', 'Data Not Available'), 'price': sitedata.get('price', 'Data Not Available'), 'source': sitedata.get('source', 'Data Not Available')})

    return results

def final_answering(messages, sources):
    if sources:
        messages.insert(0, {
        'role': 'system',
        'content': f'''As a professional writer, your job is to generate a comprehensive and informative, yet concise answer of 400 words or less for the given user's question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text. If there are any images relevant to your answer, be sure to include them as well. Aim to directly address the user's question, augmenting your response with insights gleaned from the search results. 
                Whenever quoting or referencing information from a specific URL, always cite the source URL explicitly. Please match the language of the response to the user's language.
                Always answer in Markdown format. Links and images must follow the correct format.
                Link format: [link text](url)
                Image format: ![alt text](url)
                
                User's Question:
                {messages[-1]['content']}

                Search Result Information:
                {sources}
                '''
        })

        stream = groq.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=2500,
            top_p=1,
            stop=None,
            stream=True,
        )

        answer = ''

        for chunk in stream:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
                yield{
                    'data': chunk.choices[0].delta.content
                }
    else:
        messages.insert(0, {
        'role': 'system',
        'content': f'''
                    You are an advanced language model trained to assist users by answering their questions and queries to the best of your abilities. Your goal is to provide helpful, informative, and truthful responses that are tailored to the user's specific needs.
                    When a user asks you a question or presents you with a query, analyze it carefully to understand the user's intent and information needs. Draw upon your broad knowledge base spanning various domains to formulate a comprehensive response.
                    If you are uncertain about any aspect of the query or do not have enough information to provide a complete answer, acknowledge the limitations of your knowledge and offer suggestions on how the user could find more details.
                    Be polite, empathetic, and respectful in your interactions. Avoid biased or discriminatory language. If a user asks about sensitive topics, respond thoughtfully while maintaining appropriate boundaries.
                    Your responses should be clear, concise, and easy to understand. Use plain language and avoid jargon unless necessary. Provide relevant examples or analogies to illustrate key points.
                    Overall, your goal is to be a knowledgeable, trustworthy, and helpful assistant to the user. Leverage your capabilities to the fullest extent to deliver the best possible response to each query.
                '''
        })

        stream = groq.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=2500,
            top_p=1,
            stop=None,
            stream=True,
        )

        answer = ''

        for chunk in stream:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
                yield{
                    'data': chunk.choices[0].delta.content
                }

    print(answer)

@app.get("/getSources")
async def get_sources(request: Request):
    request_dict = await request.json()

    messages = request_dict["messages"]

    class Query(BaseModel):
        query: str

    query_rewriting = groq.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a rephraser and always respond with response which is rephrased version of the input
                    that is given to a search engine API. Always be succint and use the same words as the input.
                    
                    {
                        'query': '<rewritten query>'
                    }
                    """
            },
            {
                "role": "user",
                "content": f"""Rewrite user query into an independent search query for Google Search in this JSON format - 

                    Chat History: {messages[:-1]}
                    User Query: {messages[-1]['content']}
                    
                    Construct an independent single search query from the user's query in the expected JSON format. Refrain from adding any explanation along with the JSON."""
            },
        ],
        model="mixtral-8x7b-32768",
        temperature=0,
        stream=False,
    )

    query = json.loads(query_rewriting.choices[0].message.content)

    print(query)

    class Intent(BaseModel):
        name: str
        score: float

    class IntentList(BaseModel):
        intent: List[Intent]

    judge_intent = groq.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """Your task is to analyze the user's query and determine the most likely intents behind it. You will be provided with a list of possible intent types, and you must choose 2 that best matches the query, assigning a score from 0 to 1 for each (with 1 being the most certain).
                    
                    The possible intent types are:
                    
                    Wikipedia and Facts
                    Social Media: Instagram
                    Social Media: Twitter
                    Casual Chitchat
                    News and Current Events
                    Shopping and Purchasing
                    LinkedIn and People
                    Social Media: Facebook
                    Live Sports Updates
                    Live Weather or forecast
                    Finance and Crypto
                    Stackoverflow and Coding
                    Where to Watch Movies and TV Shows
                    Reddit or Advice, Opinion, Reviews,
                    Yelp and Restaurants 
                    Glassdoor and Business Reviews
                    Video Search
                    Places and Businesses
                    Image Search
                    Unknown

                    Your output should be a JSON object with the following structure:
                    {
                        "Finance and Crypto": 0.7,
                        "Wikipedia and Facts": 0.15
                    }
                    If the user's query does not clearly fit any of the provided intent types, you should mark it as "Unknown" with a score of 1.0.\n"""
                    
                    f"The JSON object must use the schema: {json.dumps(IntentList.model_json_schema(), indent=2)}"
            },
            {
                "role": "user",
                "content": f"Judge the intent of the user's query - {query}",
            },
        ],
        model="mixtral-8x7b-32768",
        temperature=0,
        stream=False,
        response_format={"type": "json_object"},
    )

    intent_list = IntentList.model_validate_json(judge_intent.choices[0].message.content)
    sorted_intent_list = sorted(intent_list.intent, key=lambda x: x.score, reverse=True)
    intent = sorted_intent_list[0].name

    print(intent)

    sources = {}

    if intent in search:
        print('search')
        sources = getSerperResults(query['query'], 'https://google.serper.dev/search', 10)
        if sources:
            sources = rerank_search_results(query['query'], sources, num=3)
            sources = scrapeURLData(sources)
            print(sources)
    elif intent in shopping:
        print('shopping')
        sources = getSerperResults(query['query'], 'https://google.serper.dev/shopping', 20)
        print(sources)
    elif intent in news:
        print('news')
        sources = getSerperResults(query['query'], 'https://google.serper.dev/news', 10)
        if sources:
            sources = rerank_search_results(query['query'], sources, num=5)
            sources = scrapeURLData(sources)
            print(sources)
    elif intent in images:
        print('images')
        sources = getSerperResults(query['query'], 'https://google.serper.dev/images', 10)
        print(sources)
    elif intent in videos:
        print('videos')
        sources = getSerperResults(query['query'], 'https://google.serper.dev/videos', 10)
        print(sources)
    elif intent in maps:
        print('maps')
        sources = getSerperResults(query['query'], 'https://google.serper.dev/maps', None)
        print(sources)
    else:
        print('none')
        sources = None

    return sources

@app.get("/getAnswer")
async def get_answer(request: Request):

    request_dict = await request.json()

    messages = request_dict["messages"]
    sources = request_dict["sources"]

    if sources:
        return EventSourceResponse(final_answering(messages, sources), media_type="text/event-stream")
    else:
        return EventSourceResponse(final_answering(messages, None), media_type="text/event-stream")