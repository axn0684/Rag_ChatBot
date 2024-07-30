# RAG_ChatBot
A Chatbot implemented using Retrieval Augmented Generation allowing users to upload a pdf file and query based on the given information in the file.

The text given in the pdf file will be converted to embeddings using Sentence Transformers' all-mpnet-base-v2 and the ChatBot is based on Google's Gemma-2b-it

# Installation and running details
- Install the required libraries in setup.py
- For interactive localhost web server, run python gradio_interface
- For running in command prompt, run python back_end main
