import gradio as gr
import torch
from Augmentation_processing import *
from Retrieval_processing import *
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
global pdf_data
# Define a function to process the PDF and set up the models
def file_processing(pdf):
    global pdf_data
    embeddings, pages_and_chunks = retrieval_setup(pdf_file_path=pdf)
    a=model_config()
    llm_model,tokenizer=a[0],a[1]
    
    pdf_data = {
        "embeddings": embeddings,
        "pages_and_chunks": pages_and_chunks,
        "llm_model": llm_model,
        "tokenizer": tokenizer
    }
    print("------pdf processed-------")
    new_text_box=gr.Textbox(label="Enter text and press ENTER", placeholder="Type your message here...",interactive=True)
    pdf_reset=gr.Button(value="Reupload PDF",visible=True)
    return new_text_box,pdf_reset

def reset_pdf():
    os.remove("upload.csv")
    return gr.File(value=None,label="Upload PDF", file_types=["pdf"])
# Define the chatbot function that will use the proxy function
def chatbot(query, pdf, temperature,max_output_tokens):
    global pdf_data
    # Use the global conversation history
    # Process the PDF file and set up models if this is the first query
    
    # Get the response from the model
    output_text, context_items = ask_query(
        query=query,
        llm_model=pdf_data["llm_model"],
        tokenizer=pdf_data["tokenizer"],
        embeddings=pdf_data["embeddings"],
        pages_and_chunks=pdf_data["pages_and_chunks"],
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

    return output_text,context_items
def user(user_message, history):
    if user_message is None or user_message=="":
        return user_message, history
    return "", history + [[user_message, None]]
# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot")
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large")
    def transcribe(audio):
        try: 
            sr, y = audio
            y = y.astype(np.float32)
            y /= np.max(np.abs(y))
        except:
            return

        return transcriber({"sampling_rate": sr, "raw": y})["text"]
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=["pdf"])
            pdf_reset=gr.Button(value="Reupload PDF",visible=False)
            with gr.Accordion("Parameters", open=False):
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.2)
                max_output_tokens = gr.Slider(label="Max output tokens", minimum=1, maximum=2048, value=1024)
            context_output = gr.Checkbox(label="Relevant info",info="Include relevant information from text.")
        with gr.Column(scale=2):
            chatbot_output = gr.Chatbot(label=None)
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
        
            chatbot_input = gr.Textbox(label="Enter text and press ENTER", placeholder="Please upload your pdf file to begin",interactive=False)
            clear = gr.ClearButton([chatbot_input,chatbot_output])
            
        def submit(chatbot_input, pdf_input, temperature,max_output_tokens,chat_history,context_output):
            if pdf_input is None:
                yield  "Please upload a PDF file."
                return
            chat_history[-1][1] = ""    
            response,context = chatbot(chat_history[-1][0], pdf_input, temperature, max_output_tokens)
            chat_history[-1][1] +=response
            yield chat_history
            if context_output and context is not None:
                chat_history.append([None,'Relevant information from the input file can be found below:\n'+context])
                yield chat_history
            
            
        chatbot_input.submit(user,[chatbot_input,chatbot_output],[chatbot_input,chatbot_output]).then(
            submit, inputs=[chatbot_input, pdf_input, temperature, max_output_tokens,chatbot_output,context_output], outputs=[chatbot_output])
        pdf_input.upload(file_processing,inputs=[pdf_input],outputs=[chatbot_input,pdf_reset])
        pdf_reset.click(reset_pdf,outputs=pdf_input)
        audio_input.change(fn=transcribe, inputs=audio_input, outputs=chatbot_input )
# Launch the Gradio interface

demo.launch()
