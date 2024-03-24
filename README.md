# Chat and validate the response given by the LLM
### A real-time LLM chat App powered by llama-index and RAPTOR

By: [Tanishq Selot](https://github.com/tanishq150802) , Based on the latest tree-organized information retrieval technique [RAPTOR](https://github.com/parthsarthi03/raptor), powered by [llama-index](https://github.com/run-llama/llama_index) for pipelining, [ragas](https://docs.ragas.io/en/stable/) as an assessment framework for the pipeline and [streamlit](https://streamlit.io) as a deployment framework.

Clone this repository. Open Command Prompt. ```cd``` into the cloned repository and use the command ```pip install -r requirements.txt``` to install the requirements inside a virtual environment. Run ```streamlit run app.py``` to find the app running on http://localhost:8501/.

## The App
https://github.com/tanishq150802/chat_and_validate/assets/81608921/20e36343-f0d6-400e-a88e-71a87fa94c50
* Upload the PDF.
* Write your query and  expected answer. Click on "Submit" to generate the LLM's response from the indexed document.
* Click on validate to show the "Answer Correctness" score.

## Requirements
* llama-index
* llama-index-packs-raptor 
* llama-index-vector-stores-qdrant
* llama-index-vector-stores-chroma
* chromadb
* streamlit
* ragas
