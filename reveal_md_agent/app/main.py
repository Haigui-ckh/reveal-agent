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
    theme: Optional[str] = None
    presentation_id: Optional[str] = None
    output_html_path: Optional[str] = None
    output_pdf_path: Optional[str] = None

# --- LangGraph Nodes ---
def process_markdown(state: GraphState) -> GraphState:
    """Processes and structures the input markdown content using an LLM."""
    print("---CALLING LLM TO STRUCTURE MARKDOWN---")
    
    # 输出原始内容，用于调试
    print("\n===== ORIGINAL MARKDOWN CONTENT =====")
    print(state["markdown_content"])
    print("===== END ORIGINAL CONTENT =====\n")

    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("---WARNING: OPENAI_API_KEY not found. Skipping LLM structuring.---")
        # 直接添加分隔符处理，不依赖LLM
        content = state["markdown_content"]
        
        # 检查内容是否已经包含分隔符
        if "---" not in content and "--" not in content:
            # 如果没有分隔符，根据标题自动添加
            lines = content.split('\n')
            processed_lines = []
            
            for line in lines:
                # 对于主标题（# 开头），添加水平分隔符
                if line.strip().startswith('# '):
                    if processed_lines and not processed_lines[-1].strip() == '---':
                        processed_lines.append('---')
                    processed_lines.append(line)
                # 对于二级标题（## 开头），添加垂直分隔符
                elif line.strip().startswith('## '):
                    if processed_lines and not processed_lines[-1].strip() == '--':
                        processed_lines.append('--')
                    processed_lines.append(line)
                else:
                    processed_lines.append(line)
            
            # 确保开头有分隔符
            if processed_lines and not processed_lines[0].strip() == '---':
                processed_lines.insert(0, '---')
                
            structured_markdown = '\n'.join(processed_lines)
        else:
            # 已有分隔符，保持原样
            structured_markdown = content
    else:
        try:
            llm = ChatOpenAI(model="deepseek-chat", temperature=0, base_url="https://api.deepseek.com")
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert in creating presentations. Your task is to structure raw markdown content for reveal.js slides.

IMPORTANT FORMATTING RULES:
1. Insert '---' on its own line to create a new horizontal slide (main topic)
2. Insert '--' on its own line to create a new vertical slide (sub-point of the current main topic)
3. Use headings (# for main topics, ## for subtopics) to structure content
4. Each slide should have a clear heading and concise content
5. Group related content into vertical slides under the same main topic
6. Ensure proper spacing - each separator must be on its own line with no extra spaces
7. DO NOT add any explanatory text or comments - output ONLY the formatted markdown

Analyze the content's logical structure (headings, lists, paragraphs) to decide where to place the separators.
Your output must be ONLY the modified markdown content with proper slide separators."""),
                ("user", "Please structure the following markdown content for a reveal.js presentation:\n\n{markdown_input}")
            ])
            parser = StrOutputParser()
            chain = prompt_template | llm | parser
            structured_markdown = chain.invoke({"markdown_input": state["markdown_content"]})
            
            # 输出模型生成的内容，用于调试
            print("\n===== MODEL GENERATED CONTENT =====")
            print(structured_markdown)
            print("===== END MODEL GENERATED CONTENT =====\n")
            
            # 确保分隔符格式正确
            structured_markdown = structured_markdown.replace("\n\n---\n\n", "\n---\n")
            structured_markdown = structured_markdown.replace("\n\n--\n\n", "\n--\n")
            
            # 确保每个分隔符都是单独一行
            lines = structured_markdown.split('\n')
            for i in range(len(lines)):
                if lines[i].strip() == '---' or lines[i].strip() == '--':
                    lines[i] = lines[i].strip()
            structured_markdown = '\n'.join(lines)
            
            print("---LLM PROCESSING COMPLETED SUCCESSFULLY---")
        except Exception as e:
            print(f"---ERROR during LLM call: {e}. Falling back to original content.---")
            # 出错时使用基于规则的处理方法
            content = state["markdown_content"]
            
            # 检查内容是否已经包含分隔符
            if "---" not in content and "--" not in content:
                # 如果没有分隔符，根据标题自动添加
                lines = content.split('\n')
                processed_lines = []
                
                for line in lines:
                    # 对于主标题（# 开头），添加水平分隔符
                    if line.strip().startswith('# '):
                        if processed_lines and not processed_lines[-1].strip() == '---':
                            processed_lines.append('---')
                        processed_lines.append(line)
                    # 对于二级标题（## 开头），添加垂直分隔符
                    elif line.strip().startswith('## '):
                        if processed_lines and not processed_lines[-1].strip() == '--':
                            processed_lines.append('--')
                        processed_lines.append(line)
                    else:
                        processed_lines.append(line)
                
                # 确保开头有分隔符
                if processed_lines and not processed_lines[0].strip() == '---':
                    processed_lines.insert(0, '---')
                    
                structured_markdown = '\n'.join(processed_lines)
            else:
                # 已有分隔符，保持原样
                structured_markdown = content
    
    # 输出最终处理后的内容，用于调试
    print("\n===== FINAL PROCESSED CONTENT =====")
    print(structured_markdown)
    print("===== END FINAL PROCESSED CONTENT =====\n")
    
    print("---MARKDOWN PROCESSING COMPLETED SUCCESSFULLY---")
    
    # 更新状态
    state["markdown_content"] = structured_markdown
    return state

def generate_html(state: GraphState) -> GraphState:
    """Generates the reveal.js HTML file from markdown."""
    print("---GENERATING HTML---")
    # 生成唯一ID
    presentation_id = str(uuid.uuid4())
    template = templates.get_template("index.html")
    html_content = template.render(
        title=state.get("title", "My Presentation"),
        content=state["markdown_content"],
        theme=state.get("theme", "black")  # 使用用户选择的主题，默认为黑色
    )
    
    output_path = os.path.join(OUTPUTS_DIR, f"{presentation_id}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # 更新状态
    state["presentation_id"] = presentation_id
    state["output_html_path"] = output_path
    return state

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
    markdown_file: Optional[UploadFile] = File(None),
    theme: str = Form("black")
):
    markdown_content = ""
    if markdown_file and markdown_file.filename:
        content = await markdown_file.read()
        markdown_content = content.decode("utf-8")
    elif markdown_text:
        markdown_content = markdown_text

    if not markdown_content.strip():
        raise HTTPException(status_code=400, detail="Please provide markdown text or a file with content.")

    initial_state = {"markdown_content": markdown_content, "title": title, "theme": theme}
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
    # Create a simple upload form with theme selection
    upload_form_html = """

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Markdown 幻灯片生成器</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2980b9;
            --background-color: #f8f9fa;
            --card-color: #ffffff;
            --text-color: #333333;
            --border-radius: 8px;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            --transition: all 0.3s ease;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 0;
            margin: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            width: 90%;
            max-width: 800px;
            padding: 2rem;
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
        }
        
        h1 {
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 600;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            font-size: 1rem;
            transition: var(--transition);
        }
        
        textarea {
            width: 100%;
            height: 200px;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            font-size: 1rem;
            resize: vertical;
            transition: var(--transition);
        }
        
        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }
        
        select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            font-size: 1rem;
            background-color: white;
            transition: var(--transition);
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23333' d='M6 8.825L1.175 4 2.05 3.125 6 7.075 9.95 3.125 10.825 4z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 0.75rem center;
            background-size: 12px;
            cursor: pointer;
        }
        
        .divider {
            display: flex;
            align-items: center;
            margin: 1.5rem 0;
            color: #777;
        }
        
        .divider::before, .divider::after {
            content: "";
            flex: 1;
            height: 1px;
            background-color: #ddd;
        }
        
        .divider span {
            padding: 0 1rem;
            font-weight: 600;
        }
        
        .file-upload {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .file-upload-label {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 0.75rem;
            background-color: #f1f1f1;
            border: 1px dashed #ccc;
            border-radius: var(--border-radius);
            cursor: pointer;
            transition: var(--transition);
        }
        
        .file-upload-label:hover {
            background-color: #e9e9e9;
        }
        
        .file-upload input[type="file"] {
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .btn {
            display: inline-block;
            background-color: var(--primary-color);
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: var(--border-radius);
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition);
            width: 100%;
        }
        
        .btn:hover {
            background-color: var(--secondary-color);
            transform: translateY(-2px);
        }
        
        .footer {
            margin-top: 1rem;
            text-align: center;
            font-size: 0.9rem;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Markdown 幻灯片生成器</h1>
        <form action="/generate/" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="title">演示标题</label>
                <input type="text" id="title" name="title" value="我的精彩演示" placeholder="输入您的演示标题">
            </div>
            
            <div class="form-group">
                <label for="markdown_text">输入 Markdown 内容</label>
                <textarea id="markdown_text" name="markdown_text" placeholder="在此输入您的 Markdown 内容..."></textarea>
            </div>
            
            <div class="divider">
                <span>或者</span>
            </div>
            
            <div class="form-group">
                <div class="file-upload">
                    <label for="markdown_file" class="file-upload-label">
                        选择 Markdown 文件上传
                    </label>
                    <input type="file" id="markdown_file" name="markdown_file" accept=".md,.markdown,.txt">
                </div>
            </div>
            
            <div class="form-group">
                <label for="theme">选择演示主题</label>
                <select id="theme" name="theme" class="form-control">
                    <option value="black" selected>黑色 (默认)</option>
                    <option value="white">白色</option>
                    <option value="league">League</option>
                    <option value="beige">米色</option>
                    <option value="sky">天空蓝</option>
                    <option value="night">夜晚</option>
                    <option value="serif">衬线字体</option>
                    <option value="simple">简约</option>
                    <option value="solarized">Solarized</option>
                    <option value="blood">血色</option>
                    <option value="moon">月亮</option>
                </select>
            </div>
            
            <button type="submit" class="btn">生成演示幻灯片</button>
        </form>
        
        <div class="footer">
            基于 Reveal.js 构建的 Markdown 幻灯片生成工具
        </div>
    </div>
    
    <script>
        // 显示所选文件名
        document.getElementById('markdown_file').addEventListener('change', function(e) {
            const fileName = e.target.files[0] ? e.target.files[0].name : '选择 Markdown 文件上传';
            document.querySelector('.file-upload-label').textContent = fileName;
        });
    </script>
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