from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel
from dotenv import load_dotenv
import tempfile, os, re, openai

# Load .env with TOGETHER_API_KEY
load_dotenv()
openai.api_key = os.getenv("TOGETHER_API_KEY")
openai.api_base = "https://api.together.xyz/v1"

app = FastAPI()

# ✅ Allow your deployed frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-kappa-one-91.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_DB_PATH = "chroma_store"
vectorstore = None
user_info = {}

@app.get("/")
def root():
    return {"message": "🚀 Resume Chat API is running"}

# Extract personal info from resume
def extract_personal_info(text: str):
    info = {
        "name": None,
        "email": None,
        "phone": None,
    }

    lines = text.strip().split("\n")
    if lines:
        info["name"] = lines[0].strip()

    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+", text)
    if email_match:
        info["email"] = email_match.group()

    phone_match = re.search(r"(\+?\d[\d\-\s]{8,}\d)", text)
    if phone_match:
        info["phone"] = phone_match.group()

    return info

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Extract plain text
        resume_text = "\n".join([doc.page_content for doc in documents])
        global user_info
        user_info = extract_personal_info(resume_text)

        # Split & embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(
            model_name="intfloat/e5-small-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        global vectorstore
        vectorstore = Chroma.from_documents(chunks, embedding, persist_directory=VECTOR_DB_PATH)

        os.remove(tmp_path)

        # Personalized greeting
        greeting = "👋 Hello"
        if user_info.get("name"):
            greeting += f" {user_info['name']}"
        greeting += "! How may I assist you with your resume today?"

        return {"message": greeting}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        question = body.get("message", "").strip()

        if not question:
            return JSONResponse(status_code=400, content={"error": "Message required."})

        question_lower = question.lower()

        # Handle basic questions
        if "your name" in question_lower or "my name" in question_lower:
            if user_info.get("name"):
                return StreamingResponse(iter([f"data: Your name is {user_info['name']}\n\n"]), media_type="text/event-stream")
        if "email" in question_lower or "mail" in question_lower:
            if user_info.get("email"):
                return StreamingResponse(iter([f"data: Your email is {user_info['email']}\n\n"]), media_type="text/event-stream")
        if "phone" in question_lower or "mobile" in question_lower:
            if user_info.get("phone"):
                return StreamingResponse(iter([f"data: Your phone number is {user_info['phone']}\n\n"]), media_type="text/event-stream")

        global vectorstore
        if vectorstore is None:
            embedding = HuggingFaceEmbeddings(
                model_name="intfloat/e5-small-v2",
                encode_kwargs={"normalize_embeddings": True}
            )
            vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding)

        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs[:2]])

        prompt = f"[INST] Use the following resume content to answer the question.\n\n{context}\n\nQuestion: {question} [/INST]"

        def stream():
            try:
                response = openai.ChatCompletion.create(
                    model="mistralai/Mistral-7B-Instruct-v0.2",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=0.7,
                    max_tokens=300
                )
                for chunk in response:
                    content = chunk.choices[0].delta.get("content", "")
                    if content:
                        yield f"data: {content}\n\n"
            except Exception as e:
                print("[streaming error]", e)
                yield f"data: ERROR: {str(e)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    except Exception as e:
        print("[/chat error]", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# Role-fit score logic
class ScoreRequest(BaseModel):
    job_description: str

@app.post("/score")
async def score_fit(payload: ScoreRequest):
    try:
        job_description = payload.job_description.strip()
        if not job_description:
            return JSONResponse(status_code=400, content={"error": "Job description required."})

        global vectorstore
        if not vectorstore:
            return JSONResponse(status_code=500, content={"error": "Resume not uploaded yet."})
        docs = vectorstore.similarity_search(job_description, k=5)
        resume_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an AI resume scoring assistant. Given the job description and resume content below, give a match score (0 to 100) for how well the resume fits the job.

Resume:
{resume_text}

Job Description:
{job_description}

Respond only with a number between 0 and 100.
"""

        response = openai.ChatCompletion.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.2
        )

        score_text = response.choices[0].message["content"]
        score = int(re.search(r"\d{1,3}", score_text).group())
        score = min(max(score, 0), 100)
        return {"score": score}

    except Exception as e:
        print("[score error]", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
