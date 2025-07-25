## 1. A Singapore Tourism Chatbot

This is a Streamlit based chatbot prototype, allowing a user to ask questions that'd interest a typical tourist. The backend consists of the following components

- RAG Fusion Module
- Vector DB
- Chat Module based on gpt-4-turbo-preview

A user can ask questions to the chatbot and get responses based on the knowledge available in the form of a vector db (Chroma db). In this case the vector db is made of embeddings produced from a bunch of text files containing Singapore related data.

Once a user posts a question , the following is the general logic:

1. Build 4 similar questions using gemini-2.0-flash. Thus we have 5 similar questions to be sent to the retriever.
2. The retreiver then retrieves top 5 documents from the vector store for each question.
3. A Reciprocal Rank Fusion fuses the retrieved documents and ranks them. In case no documents are received , it produces and empty list of documents.
4. The retrieved documents are injected into the prompt as context , enabling an llm (gpt-4-turbo-preview) to produce an answer.  

## 2. Usage (Windows)

#### 2.1 Clone the Repo

Open powershell and create a folder ```your_path``` or with any other name. Then clone the repo into the created folder.
Use following command<br>

```C:\your_path> git clone https://github.com/Arindam75/ChatAgent.git -b main```


#### 2.2 Creating the Environemnt

In the same powershell terminal, run the following command, to install uv. You can skip this line , if you have uv already installed. 

```C:\your_path\ChatAgent\>powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

Now move (cd ChatAgent) to the folder ChataAgent , that was cloned above.

Run the following command to install the environment

```C:\your_path\ChatAgent\>uv env .venv```

To activate the environment run:<br>
```C:\your_path\ChatAgent\>.venv\Scripts\Activate.ps1```<br>
```C:\your_path\ChatAgent\>uv pip install -r requirements.txt```

#### 2.3 Running the App

This project uses gpt-4-turbo-preview and gemini-2.0-flash. The vector db is downloaded from huggingface hub. So create a .env file in the current folder with the following keys.

```
OPENAI_API_KEY = "sk-Nxxx"
GOOGLE_API_KEY = "AIzxxx"
HF_API_KEY = "hf_xxx"
```

Now that the environment is activated , run the following command:<br>
```(.venv) PS C:\your_path\ChatAgent> streamlit run main.py```

## 3. Acknowledgement

- [Sam Witteveen RAG Fusion](https://youtu.be/GchC5WxeXGc?si=pBAzX3naY_D85UM-): This is a fabulous video on RAG Fusion. I have used the same dataset to produce the vector db.
- [Streamlit Crash Course by Tim](https://youtu.be/o8p7uQCGD0U?si=4fAteqwpkcxP18b6): A very useful course to quickly get acquainted with Streamlit.  
