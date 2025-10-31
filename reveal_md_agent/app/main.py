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
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

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
    html_content: Optional[str] = None

# --- LangGraph Nodes ---
def process_markdown(state: GraphState) -> GraphState:
    """Processes input markdown; with API key, request LLM to return full HTML."""
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
            # 读取外部 Prompt 模板并注入 index.html 内容
            prompt_path = os.path.join(PROMPTS_DIR, "markdown_to_reveal_prompt.txt")
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read()
            except FileNotFoundError:
                prompt_text = (
                    "You are an expert at generating full Reveal.js HTML presentations with animation effects. "
                    "Use the provided base HTML and resource paths, return ONLY HTML."
                )

            try:
                with open(os.path.join(TEMPLATES_DIR, "index.html"), "r", encoding="utf-8") as tf:
                    index_html = tf.read()
            except Exception:
                index_html = ""
            print(f"---index_html content preview: {index_html[:1000]}... (truncated) ...---")
            print(f"---prompt_text content preview: {prompt_text}... (truncated) ...---")
            # 不在系统消息中直接替换 HTML，以免触发 {} 为格式化变量的误解析
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("user", "Title: {title}\nTheme: {theme}\nBasePath: {base_path}\n\nMarkdown Input:\n{markdown_input}")
            ])
            parser = StrOutputParser()
            chain = prompt_template | llm | parser
            full_html = chain.invoke({
                "markdown_input": state["markdown_content"],
                "title": state.get("title", "My Presentation"),
                "theme": state.get("theme", "black"),
                "base_path": "/static/reveal.js",
                "index_html": index_html,
            })
            
            # 输出模型生成的内容，用于调试
            print("\n===== MODEL GENERATED FULL HTML =====")
            print(full_html[:1000])
            print("... (truncated) ...")
            print("===== END MODEL GENERATED FULL HTML =====\n")

            print("---LLM FULL HTML GENERATION COMPLETED SUCCESSFULLY---")
            state["html_content"] = full_html
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
    
    # 输出最终处理后的内容（若为LLM HTML则略）
    if state.get("html_content"):
        print("---HTML mode active, skipping markdown debug output---")
    else:
        print("\n===== FINAL PROCESSED MARKDOWN =====")
        print(structured_markdown)
        print("===== END FINAL PROCESSED MARKDOWN =====\n")
        print("---MARKDOWN PROCESSING COMPLETED SUCCESSFULLY---")
        state["markdown_content"] = structured_markdown
    return state

def generate_html(state: GraphState) -> GraphState:
    """Generates the reveal.js HTML file from markdown."""
    print("---GENERATING HTML---")
    # 生成唯一ID
    presentation_id = str(uuid.uuid4())
    # 如果已有LLM生成的完整HTML，直接使用
    if state.get("html_content"):
        html_content = state["html_content"]
    else:
        template = templates.get_template("index.html")
        html_content = template.render(
            title=state.get("title", "My Presentation"),
            content=state["markdown_content"],
            theme=state.get("theme", "black"),  # 使用用户选择的主题，默认为黑色
            base_path="/static/reveal.js"
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
    # 不再写入模板文件，确保仅从 templates 目录读取
    # 可以在此处添加健康检查或目录创建逻辑（若需要）
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)