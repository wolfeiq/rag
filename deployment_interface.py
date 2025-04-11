import os
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_aws import BedrockEmbeddings

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
os.makedirs(templates_dir, exist_ok=True)

template_path = os.path.join(templates_dir, "index.html")
with open(template_path, "w", encoding="utf-8") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frag Mich Was - PLZ TU Berlin</title>
    <style>
        :root {
            --main-bg: #f5f0e1;
            --beige-light: #e8dcc5;
            --beige-mid: #d4b996;
            --beige-dark: #b08d67;
            --beige-deep: #8c6a4f;
            --text-primary: #4a3623; 
            --text-secondary: #7a5e47; 
            --shadow-light: #ffffff;
            --shadow-dark: #d8c3a5;
            --error-color: #e07b39; 
            --font-main: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: var(--font-main);
            background-color: var(--main-bg);
            color: var(--text-primary);
            max-width: 900px;
            margin: 0 auto;
            padding: 30px;
            line-height: 1.6;
        }
        
        h1 {
            font-weight: 700;
            font-size: 2.2rem;
            margin-bottom: 1.5rem;
            color: var(--lavender-deep);
            text-align: center;
            letter-spacing: 0.5px;
        }
        
        .container {
            background: var(--main-bg);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 8px 8px 16px var(--shadow-dark), 
                        -8px -8px 16px var(--shadow-light);
        }
        
        .query-section {
            margin-bottom: 20px;
        }
        
        #query-input {
            width: 100%;
            min-height: 120px;
            padding: 16px;
            border: none;
            background: var(--main-bg);
            color: var(--text-primary);
            font-family: var(--font-main);
            font-size: 1rem;
            border-radius: 15px;
            resize: vertical;
            margin-bottom: 20px;
            box-shadow: inset 5px 5px 10px var(--shadow-dark), 
                        inset -5px -5px 10px var(--shadow-light);
            transition: all 0.3s ease;
        }
        
        #query-input:focus {
            outline: none;
            box-shadow: inset 5px 5px 10px var(--shadow-dark), 
                        inset -5px -5px 10px var(--shadow-light);
        }
        
        #query-input::placeholder {
            color: var(--text-secondary);
            opacity: 0.7;
        }
        
        #submit-btn {
            padding: 14px 28px;
            background: var(--main-bg);
            color: var(--lavender-deep);
            font-weight: 600;
            font-size: 1rem;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            display: block;
            margin: 0 auto;
            box-shadow: 5px 5px 10px var(--shadow-dark), 
                       -5px -5px 10px var(--shadow-light);
            transition: all 0.2s ease;
        }
        
        #submit-btn:hover {
            background: linear-gradient(145deg, var(--main-bg), var(--lavender-light));
            color: var(--lavender-deep);
        }
        
        #submit-btn:active {
            box-shadow: inset 3px 3px 6px var(--shadow-dark), 
                        inset -3px -3px 6px var(--shadow-light);
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            color: var(--lavender-dark);
            font-style: italic;
            border-radius: 12px;
            background: var(--main-bg);
            box-shadow: inset 3px 3px 6px var(--shadow-dark), 
                        inset -3px -3px 6px var(--shadow-light);
        }
        
        .result-section {
            margin-top: 30px;
        }
        
        .result-card {
            background: var(--main-bg);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 8px 8px 16px var(--shadow-dark), 
                        -8px -8px 16px var(--shadow-light);
        }
        
        .result-title {
            font-size: 1.2rem;
            color: var(--lavender-deep);
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        #response {
            white-space: pre-wrap;
            line-height: 1.7;
            color: var(--text-primary);
            padding: 10px;
            border-radius: 10px;
            background: var(--main-bg);
            box-shadow: inset 3px 3px 6px var(--shadow-dark), 
                        inset -3px -3px 6px var(--shadow-light);
            min-height: 100px;
        }
        
        #sources {
            margin-top: 15px;
            font-size: 0.9rem;
            color: var(--text-secondary);
            padding: 10px;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 15px;
            }
            
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 1.8rem;
            }
        }
    </style>
</head>
<body>
    <h1>Hi, frag mich etwas über den Inhalt der Dokumente</h1>
            
    
    <div class="container">
        <div class="query-section">
            <textarea id="query-input" rows="4" placeholder="z.B: Wie beeinflusst der Kontakt mit Kühlschmierstoffen (KSS) die Rückhaltefähigkeit von PC-Sichtscheiben? Welche Rolle spielt die TU Berlin im Projekt ExoTherm? Welche Vorteile hätte das CM-System für WZM-Betreiber?"></textarea>
            <button id="submit-btn">Ask Me</button>
        </div>
        
        <div class="loading" id="loading">
            Processing your query with llama3.2... This may take a moment.
        </div>
        
        <div class="result-section">
            <div class="result-card">
                <div class="result-title">What I found...</div>
                <div id="response"></div>
            </div>
            
            <div class="result-card">
                <div class="result-title">Information pulled from</div>
                <div id="sources"></div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('submit-btn').addEventListener('click', async () => {
            const queryText = document.getElementById('query-input').value;
            if (!queryText) return;
            
            const loading = document.getElementById('loading');
            const responseDiv = document.getElementById('response');
            const sourcesDiv = document.getElementById('sources');
            
            loading.style.display = 'block';
            responseDiv.innerHTML = '';
            sourcesDiv.innerHTML = '';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: queryText }),
                });
                
                const data = await response.json();
                if (response.ok) {
                    responseDiv.textContent = data.response;
                    sourcesDiv.textContent = 'Sources: ' + data.sources.join(', ');
                } else {
                    responseDiv.textContent = 'Error: ' + data.error;
                    responseDiv.style.color = 'var(--error-color)';
                }
            } catch (error) {
                responseDiv.textContent = 'Error: ' + error.message;
                responseDiv.style.color = 'var(--error-color)';
            } finally {
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
    """)

FAISS_PATH = ""
PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}"""

def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="eu-central-1"
    )
    return embeddings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        response_text, sources = query_rag(query_text)
        return jsonify({
            "response": response_text,
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def query_rag(query_text: str):

    embedding_function = get_embedding_function()
    db = FAISS.load_local(FAISS_PATH, embedding_function, allow_dangerous_deserialization=True)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model="llama3.2:latest")
    response_text = model.invoke(prompt)
   
    sources = [doc.metadata.get("id", None) for doc, score in results]
    
    return response_text, sources

if __name__ == "__main__":
    print(f"To ask questions paste this into your browser at http://127.0.0.1:5000")
    print(f"MAKE SURE OLLAMA IS RUNNING! by checking with ollama list in cmd. Should return a list of models.")
    print(f"template directory: {templates_dir}")
    print(f"template created: {os.path.exists(template_path)}")
    app.run(host='127.0.0.1', port=5000, debug=True)