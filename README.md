#Create & activate venv, install:

python -m venv .venv
venv/bin/activate
pip install -r requirements.txt


#Set up Ollama & pull mistral:

ollama pull mistral
ollama serve

#Build vectors:

python src/build_vectorstore.py

#Run Q&A:

python src/qa_cli.py
