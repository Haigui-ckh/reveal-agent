import os
import uuid
import asyncio
from typing import Dict, TypedDict, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from langgraph.graph import StateGraph, END
from pyppeteer import launch

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# --- App Setup ---
app = FastAPI()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# --- FastAPI Setup ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- LangGraph State ---
class GraphState(TypedDict):
    markdown_content: str
    title: Optional[str] = None
    presentation_id: Optional[str] = None
    output_html_path: Optional[str] = None
    output_pdf_path: Optional[str] = None

# --- LangGraph Nodes ---
def process_markdown(state: GraphState) -> GraphState:
    """Processes and structures the input markdown content using an LLM."""
    print("---CALLING LLM TO STRUCTURE MARKDOWN---")

    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("---WARNING: OPENAI_API_KEY not found. Skipping LLM structuring.---")
        structured_markdown = state["markdown_content"]
    else:
        try:
            llm = ChatOpenAI(model="deepseek-chat", temperature=0, base_url="https://api.deepseek.com")
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in creating presentations. Your task is to structure raw markdown content for reveal.js slides. You must insert slide separators: '---' on a new line for new horizontal slides (main topics), and '--' on a new line for new vertical slides (sub-points of a main topic). Analyze the content's logical structure (headings, lists, paragraphs) to decide where to place the separators. Ensure the output is only the modified markdown content, without any additional explanations."),
                ("user", "Please structure the following markdown content:\n\n{markdown_input}")
            ])
            parser = StrOutputParser()
            chain = prompt_template | llm | parser
            structured_markdown = chain.invoke({"markdown_input": state["markdown_content"]})
        except Exception as e:
            print(f"---ERROR during LLM call: {e}. Falling back to original content.---")
            structured_markdown = state["markdown_content"]

    return {
        "markdown_content": structured_markdown,
        "title": state.get("title", "My Presentation"),
        "presentation_id": str(uuid.uuid4())
    }

def generate_html(state: GraphState) -> GraphState:
    """Generates the reveal.js HTML file from markdown."""
    print("---GENERATING HTML---")
    presentation_id = state["presentation_id"]
    template = templates.get_template("index.html")
    html_content = template.render(
        title=state["title"],
        content=state["markdown_content"]
    )
    
    output_path = os.path.join(OUTPUTS_DIR, f"{presentation_id}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return {"output_html_path": output_path}

# --- LangGraph Workflow ---
workflow = StateGraph(GraphState)
workflow.add_node("process_markdown", process_markdown)
workflow.add_node("generate_html", generate_html)

workflow.set_entry_point("process_markdown")
workflow.add_edge("process_markdown", "generate_html")
workflow.add_edge("generate_html", END)

app_graph = workflow.compile()

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/generate/", response_class=HTMLResponse)
async def generate_presentation(
    request: Request,
    title: str = Form("My Presentation"),
    markdown_text: Optional[str] = Form(None),
    markdown_file: Optional[UploadFile] = File(None)
):
    markdown_content = ""
    if markdown_file and markdown_file.filename:
        content = await markdown_file.read()
        markdown_content = content.decode("utf-8")
    elif markdown_text:
        markdown_content = markdown_text

    if not markdown_content.strip():
        raise HTTPException(status_code=400, detail="Please provide markdown text or a file with content.")

    initial_state = {"markdown_content": markdown_content, "title": title}
    final_state = app_graph.invoke(initial_state)
    
    presentation_id = final_state["presentation_id"]
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "presentation_id": presentation_id,
        "title": title
    })

@app.get("/presentation/{presentation_id}")
async def view_presentation(presentation_id: str):
    html_path = os.path.join(OUTPUTS_DIR, f"{presentation_id}.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Presentation not found.")
    return FileResponse(html_path)

@app.get("/download/pdf/{presentation_id}")
async def download_pdf(presentation_id: str):
    html_path = os.path.join(OUTPUTS_DIR, f"{presentation_id}.html")
    pdf_path = os.path.join(OUTPUTS_DIR, f"{presentation_id}.pdf")

    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Presentation not found.")

    browser = await launch(headless=True, args=['--no-sandbox'])
    page = await browser.newPage()
    
    await page.goto(f'file://{html_path}', {'waitUntil': 'networkidle0'})
    
    await page.pdf({'path': pdf_path, 'format': 'A4', 'printBackground': True, 'width': '1920px', 'height': '1080px'})
    
    await browser.close()
    
    return FileResponse(pdf_path, media_type='application/pdf', filename=f"{presentation_id}.pdf")

# --- Add UI Helper Files ---
@app.on_event("startup")
async def startup_event():
    # Create a simple upload form for the root
    upload_form_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Markdown to Reveal.js</title>
        <style>
            body { font-family: sans-serif; margin: 40px; }
            textarea { width: 100%; height: 200px; }
        </style>
    </head>
    <body>
        <h1>Markdown to Reveal.js Converter</h1>
        <form action="/generate/" method="post" enctype="multipart/form-data">
            <label for="title">Presentation Title:</label><br>
            <input type="text" id="title" name="title" value="My Awesome Presentation"><br><br>
            
            <label for="markdown_text">Enter Markdown Text:</label><br>
            <textarea id="markdown_text" name="markdown_text"></textarea><br><br>
            
            <p><strong>OR</strong></p>
            
            <label for="markdown_file">Upload Markdown File:</label><br>
            <input type="file" id="markdown_file" name="markdown_file"><br><br>
            
            <input type="submit" value="Generate Presentation">
        </form>
    </body>
    </html>
    """
    with open(os.path.join(TEMPLATES_DIR, "upload_form.html"), "w") as f:
        f.write(upload_form_html)

    # Create a result page
    result_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Presentation Ready</title>
        <style>body { font-family: sans-serif; margin: 40px; }</style>
    </head>
    <body>
        <h1>Your presentation '{{ title }}' is ready!</h1>
        <p><a href="/presentation/{{ presentation_id }}" target="_blank">View Presentation</a></p>
        <p><a href="/download/pdf/{{ presentation_id }}">Download as PDF</a></p>
    </body>
    </html>
    """
    with open(os.path.join(TEMPLATES_DIR, "result.html"), "w") as f:
        f.write(result_html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)