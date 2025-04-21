# enhanced_app.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import subprocess
import tempfile
import os
import re
import shutil
from typing import List, Optional, Dict, Any
import requests
import json
import difflib
import asyncio
import io
import aiohttp

app = FastAPI(title="Formal Verification Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a templates directory
os.makedirs("templates", exist_ok=True)

# Create a static directory
os.makedirs("static", exist_ok=True)

# Set up template and static file servers
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Temporary directory to store uploaded files
TEMP_DIR = "temp_verification_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Global variable to store the last verification result
last_verification_result = None

# Global variable to store the default AI prompt template
default_ai_prompt_template = """You are a formal verification expert. Analyze the following verification results and suggest fixes:

{property_section}

Status: {status}
Command: {command_used}
{output_section}

{source_files_section}

Please analyze the verification failure and suggest specific code fixes to resolve the issues.
Focus on addressing the root cause of the verification failure shown in the counterexample or error output.
Provide clear explanations for your suggested changes.
"""

def extract_properties(file_paths):
    """Extract all VERIFY_PROPERTY definitions from files"""
    properties = set()
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Regex to capture alphanumeric property identifiers
                matches = re.findall(r'#ifdef\s+(VERIFY_PROPERTY_\w+)', content)
                for match in matches:
                    # Extract the identifier after the last underscore
                    prop_id = match.split('_')[-1]
                    properties.add(prop_id)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return sorted(list(properties))

def extract_property_content(file_paths):
    """Extract the content of each VERIFY_PROPERTY block from files"""
    property_contents = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find all property blocks using regex
                # This pattern matches from #ifdef VERIFY_PROPERTY_X to the next #endif
                pattern = r'(#ifdef\s+(VERIFY_PROPERTY_\w+))(.*?)(#endif)'
                matches = re.findall(pattern, content, re.DOTALL)
                
                for match in matches:
                    property_name = match[1]  # The property name (VERIFY_PROPERTY_X)
                    # Include the #ifdef and #endif in the content for better context
                    property_content = match[0] + match[2] + match[3]
                    property_contents[property_name] = {
                        'content': property_content,
                        'file': os.path.basename(file_path)
                    }
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return property_contents

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ai_fix", response_class=HTMLResponse)
async def read_ai_fix(request: Request):
    """Serve the AI fix page"""
    return templates.TemplateResponse("ai_fix.html", {"request": request})

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...)
):
    """Handle file uploads and extract properties"""
    # Clean up old files
    for f in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, f)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    saved_files = []
    
    # Save uploaded files
    for file in files:
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        saved_files.append(file_path)
    
    # Extract properties and their content
    properties = extract_properties(saved_files)
    property_contents = extract_property_content(saved_files)
    
    return {
        "message": "Files uploaded successfully", 
        "files": [os.path.basename(f) for f in saved_files],
        "properties": properties,
        "property_contents": property_contents
    }

@app.get("/get_property_content")
async def get_property_content(property_name: str):
    """Return the content of a specific property"""
    try:
        file_paths = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if os.path.isfile(os.path.join(TEMP_DIR, f))]
        property_contents = extract_property_content(file_paths)
        
        if property_name in property_contents:
            return property_contents[property_name]
        else:
            return {"error": f"Property {property_name} not found"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify")
async def verify_properties(request: Request):
    """Run verification with custom command"""
    global last_verification_result  # Declare the global variable
    
    # Parse the request body
    data = await request.json()
    custom_command = data.get("command", "")
    mem_limit = data.get("memlimit", "10000")
    timeout_val = data.get("timeout", "300")
    
    # Get only C files for direct processing
    c_files = [f for f in os.listdir(TEMP_DIR) if f.endswith('.c')]
    if not c_files:
        return {"status": "ERROR", "message": "No .c files found. At least one C source file is required."}
    
    try:
        # Base command with only C files and include directory for header files
        file_paths = [os.path.join(TEMP_DIR, f) for f in c_files]
        cmd = ["esbmc"] + file_paths + ["-I", TEMP_DIR]
        
        # Add memory limit and timeout
        cmd.extend(["--memlimit", str(mem_limit), "--timeout", str(timeout_val)])
        
        # Add additional command options
        if custom_command:
            cmd.extend(custom_command.split())
        
        # Run the ESBMC command
        command_str = " ".join(cmd)
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_val)  # Convert to int to ensure it's a number
        )
        
        # Capture both stdout and stderr
        output = result.stdout + result.stderr
        
        # Determine verification status
        if "VERIFICATION SUCCESSFUL" in output:
            status = "SUCCESSFUL"
        elif "VERIFICATION FAILED" in output:
            status = "FAILED"
            # Extract counterexample
            counterexample = ""
            if "[Counterexample]" in output:
                ce_start = output.find("[Counterexample]")
                counterexample = output[ce_start:]
                
                # Create a separate verification process output without the counterexample
                verification_process = output[:ce_start].strip()
                
                verification_result = {
                    "status": status,
                    "output": verification_process,
                    "counterexample": counterexample,
                    "command_used": command_str
                }
                # Store the last verification result
                last_verification_result = verification_result
                return verification_result
        else:
            status = "UNKNOWN"
        
        verification_result = {
            "status": status, 
            "output": output, 
            "command_used": command_str
        }
        # Store the last verification result
        last_verification_result = verification_result
        return verification_result
    
    except subprocess.TimeoutExpired:
        timeout_result = {
            "status": "TIMEOUT", 
            "message": f"Verification timed out after {timeout_val} seconds",
            "command_used": command_str if 'command_str' in locals() else "Command not executed due to timeout"
        }
        last_verification_result = timeout_result
        return timeout_result
    except Exception as e:
        error_result = {
            "status": "ERROR", 
            "message": f"An error occurred during verification: {str(e)}", 
            "command_used": command_str if 'command_str' in locals() else "Command not executed due to error"
        }
        last_verification_result = error_result
        return error_result

@app.get("/get_last_verification")
async def get_last_verification():
    """Return the last verification results"""
    global last_verification_result  # Declare the global variable

    if last_verification_result is None:
        return {"status": "UNKNOWN", "message": "No verification has been run yet"}

    return last_verification_result

@app.get("/get_source_files")
async def get_source_files():
    """Return a list of source files in the temp directory"""
    try:
        files = [f for f in os.listdir(TEMP_DIR) if f.endswith(('.c', '.h'))]
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_source_content")
async def get_source_content(filename: str):
    """Return the content of a source file"""
    try:
        file_path = os.path.join(TEMP_DIR, filename)
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {"content": content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_default_prompt_template")
async def get_default_prompt_template():
    """Return the default AI prompt template"""
    global default_ai_prompt_template
    return {"template": default_ai_prompt_template}

@app.post("/update_prompt_template")
async def update_prompt_template(request: Dict[str, Any]):
    """Update the default AI prompt template"""
    global default_ai_prompt_template
    new_template = request.get("template", "")
    
    if not new_template:
        return {"status": "ERROR", "message": "Template cannot be empty"}
    
    default_ai_prompt_template = new_template
    return {"status": "SUCCESS", "message": "Prompt template updated successfully"}

@app.post("/generate_ai_fix")
async def generate_ai_fix(request: Request):
    """Generate AI fix suggestions for verification failures"""
    global default_ai_prompt_template
    
    data = await request.json()
    ai_model = data.get("ai_model", "")
    api_key = data.get("api_key", "")
    selected_files = data.get("selected_files", [])
    custom_prompt = data.get("custom_prompt", "")
    property_name = data.get("property_name", "")
    specific_model = data.get("specific_model", "")
    system_specification = data.get("system_specification", "")

    if not ai_model or not api_key:
        return {"error": "AI model and API key are required"}

    if not selected_files:
        return {"error": "No files selected for analysis"}

    try:
        # Get the last verification result
        verification_results = last_verification_result
        
        if not verification_results:
            return {"error": "No verification results available"}

        # Get property content if a property name is provided
        property_content = ""
        if property_name:
            file_paths = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if os.path.isfile(os.path.join(TEMP_DIR, f))]
            property_contents = extract_property_content(file_paths)
            if property_name in property_contents:
                property_info = property_contents[property_name]
                property_content = property_info['content']

        # Get source files content
        source_files_content = ""
        for filename in selected_files:
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                source_files_content += f"\nFile: {filename}\n```c\n{content}\n```\n"

        # Prepare the prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            # Use the template and replace placeholders
            prompt = default_ai_prompt_template
            
            # Replace placeholders
            if property_content:
                prompt = prompt.replace("{property_section}", f"Property being verified:\n```c\n{property_content}\n```")
            else:
                prompt = prompt.replace("{property_section}", "No specific property information available.")
            
            prompt = prompt.replace("{status}", verification_results.get("status", "UNKNOWN"))
            prompt = prompt.replace("{command_used}", verification_results.get("command_used", "No command information available."))
            
            # Add output or counterexample
            if "counterexample" in verification_results:
                prompt = prompt.replace("{output_section}", f"Counterexample:\n{verification_results['counterexample']}")
            elif "output" in verification_results:
                prompt = prompt.replace("{output_section}", f"Verification Output:\n{verification_results['output']}")
            else:
                prompt = prompt.replace("{output_section}", "No verification output available.")
            
            prompt = prompt.replace("{source_files_section}", f"Source Files:\n{source_files_content}")
            
            # Add system specification if provided
            if system_specification:
                prompt += f"\n\nAdditional System Specification:\n{system_specification}"

        # Call the appropriate AI model
        if ai_model == "chatgpt":
            response = call_chatgpt_api(prompt, api_key, specific_model or "gpt-4o")
        elif ai_model == "claude":
            response = call_claude_api(prompt, api_key, specific_model or "claude-3-7-sonnet-20250219")
        elif ai_model == "deepseek":
            response = call_deepseek_api(prompt, api_key, specific_model or "deepseek-chat")
        else:
            return {"error": f"Unsupported AI model: {ai_model}"}

        return {
            "response": response,
            "prompt_used": prompt  # Include the prompt that was used
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_last_verification_result")
async def get_last_verification_result():
    """Return the last verification results"""
    global last_verification_result  # Declare the global variable

    if last_verification_result is None:
        return {
            "verification_results": {
                "status": "UNKNOWN", 
                "message": "No verification has been run yet",
                "command_used": "No command used",
                "output": "No output available"
            }
        }

    # Extract property ID from verification command
    property_name = ""
    if "command_used" in last_verification_result:
        command_used = last_verification_result["command_used"]
        property_match = re.search(r'-D(VERIFY_PROPERTY_\w+)', command_used)
        if property_match:
            property_name = property_match.group(1)
    
    # Get property content if a property name is provided
    property_content = ""
    property_file = ""
    if property_name:
        file_paths = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if os.path.isfile(os.path.join(TEMP_DIR, f))]
        property_contents = extract_property_content(file_paths)
        if property_name in property_contents:
            property_info = property_contents[property_name]
            property_content = property_info['content']
            property_file = property_info['file']

    return {
        "verification_results": last_verification_result,
        "property_content": property_content,
        "property_file": property_file,
        "property_name": property_name
    }

@app.post("/apply_fix")
async def apply_fix(request: Dict[str, Any]):
    """Apply AI suggested fixes to source files"""
    filename = request.get("filename", "")
    fixed_content = request.get("fixed_content", "")
    property_id = request.get("property_id", "")
    
    if not filename or not fixed_content:
        return {"status": "ERROR", "message": "Filename and fixed content are required"}
    
    try:
        # Create a new filename with property suffix
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_fixed_{property_id}{ext}"
        new_file_path = os.path.join(TEMP_DIR, new_filename)
        
        # Write the fixed content to the new file
        with open(new_file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return {
            "status": "SUCCESS", 
            "message": f"Fix applied and saved as {new_filename}",
            "new_filename": new_filename
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

async def call_chatgpt_api_streaming(prompt, api_key, model="gpt-4o"):
    """Call OpenAI API with streaming response"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a formal verification expert who can analyze code and suggest fixes."},
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error: {response.status} - {error_text}")
            
            # Initialize response text
            response_text = ""
            
            # Process the streaming response
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                content = delta["content"]
                                response_text += content
                                yield {"type": "token", "content": content, "full_response": response_text}
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("data: [DONE]"):
                    break
            
            yield {"type": "done", "full_response": response_text}

async def call_claude_api_streaming(prompt, api_key, model="claude-3-7-sonnet-20250219"):
    """Call Anthropic Claude API with streaming response"""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Claude API error: {response.status} - {error_text}")
            
            # Initialize response text
            response_text = ""
            
            # Process the streaming response
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line and line.startswith("data: ") and not line.startswith("data: [DONE]"):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        if "type" in data and data["type"] == "content_block_delta":
                            if "delta" in data and "text" in data["delta"]:
                                content = data["delta"]["text"]
                                response_text += content
                                yield {"type": "token", "content": content, "full_response": response_text}
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("data: [DONE]"):
                    break
            
            yield {"type": "done", "full_response": response_text}

def call_chatgpt_api(prompt, api_key, model="gpt-4o"):
    """Call the OpenAI API with the given prompt"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def call_claude_api(prompt, api_key, model="claude-3-7-sonnet-20250219"):
    """Call the Anthropic Claude API with the given prompt"""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"

def call_deepseek_api(prompt, api_key, model="deepseek-chat"):
    """Call the DeepSeek API with the given prompt"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=model,  # Allow specifying different DeepSeek models
            messages=messages
        )
        
        # Handle both standard and reasoner models
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            # This is a reasoner model, include the reasoning in the response
            return f"Reasoning:\n{response.choices[0].message.reasoning_content}\n\nResponse:\n{response.choices[0].message.content}"
        else:
            # Standard model
            return response.choices[0].message.content
    except Exception as e:
        return f"Error calling DeepSeek API: {str(e)}"

@app.post("/generate_fixed_code")
async def generate_fixed_code(request: Dict[str, Any]):
    """Generate fixed code based on AI suggestions and user edits"""
    filename = request.get("filename", "")
    original_content = request.get("original_content", "")
    ai_suggestions = request.get("ai_suggestions", "")
    user_edits = request.get("user_edits", "")
    system_specification = request.get("system_specification", "")  # 获取系统规范
    ai_model = request.get("ai_model", "")
    api_key = request.get("api_key", "")
    
    if not filename or not original_content:
        return {"error": "Filename and original content are required"}
    
    if not ai_model or not api_key:
        return {"error": "AI model and API key are required"}
    
    try:
        # Prepare the prompt for the AI
        prompt = f"""You are a formal verification expert. I need you to fix the following code based on:
1. The original code
2. AI suggestions for fixing the code
3. User edits to those suggestions
4. System specification: {system_specification}  # Include system specification

Please generate a complete, fixed version of the file that incorporates all the necessary changes.

Original code:
```c
{original_content}
```

AI suggestions:
{ai_suggestions}

User edits to the suggestions:
{user_edits}

Please provide ONLY the complete fixed code without any explanations or markdown formatting. The output should be valid C code that can be directly saved to a file."""
        
        # Call the appropriate AI model API
        if ai_model == "chatgpt":
            fixed_code = call_chatgpt_api(prompt, api_key)
        elif ai_model == "claude":
            fixed_code = call_claude_api(prompt, api_key)
        elif ai_model == "deepseek":
            fixed_code = call_deepseek_api(prompt, api_key)
        else:
            return {"error": f"Unsupported AI model: {ai_model}"}
        
        # Clean up the response to extract just the code
        # Remove markdown code blocks if present
        if "```c" in fixed_code and "```" in fixed_code:
            start = fixed_code.find("```c") + 4
            end = fixed_code.rfind("```")
            fixed_code = fixed_code[start:end].strip()
        elif "```" in fixed_code:
            start = fixed_code.find("```") + 3
            end = fixed_code.rfind("```")
            fixed_code = fixed_code[start:end].strip()
        
        return {"fixed_code": fixed_code}
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify_fixed_code")
async def verify_fixed_code(request: Dict[str, Any]):
    """Run verification on the fixed code"""
    global last_verification_result  # Declare the global variable
    
    filename = request.get("filename", "")
    fixed_filename = request.get("fixed_filename", "")
    command = request.get("command", "")
    mem_limit = request.get("memlimit", "10000")
    timeout_val = request.get("timeout", "300")
    
    if not fixed_filename:
        return {"status": "ERROR", "message": "Fixed filename is required"}
    
    try:
        # Get all C files except the original file that was fixed
        c_files = [f for f in os.listdir(TEMP_DIR) if f.endswith('.c') and f != os.path.basename(filename)]
        
        # Add the fixed file to the list (only once)
        if fixed_filename not in c_files:
            c_files.append(fixed_filename)
        
        # Base command with C files and include directory for header files
        file_paths = [os.path.join(TEMP_DIR, f) for f in c_files]
        
        # Remove any duplicates in file paths
        file_paths = list(dict.fromkeys(file_paths))
        
        cmd = ["esbmc"] + file_paths + ["-I", TEMP_DIR]
        
        # Add memory limit and timeout
        cmd.extend(["--memlimit", str(mem_limit), "--timeout", str(timeout_val)])
        
        # Add the original command arguments (like --k-induction -DVERIFY_PROPERTY_1)
        # But filter out any memlimit or timeout flags to avoid duplication
        if command:
            filtered_args = []
            cmd_parts = command.split()
            i = 0
            while i < len(cmd_parts):
                if cmd_parts[i] in ["--memlimit", "--timeout"] and i + 1 < len(cmd_parts):
                    # Skip this flag and its value
                    i += 2
                else:
                    filtered_args.append(cmd_parts[i])
                    i += 1
            
            cmd.extend(filtered_args)
        
        # Run the ESBMC command
        command_str = " ".join(cmd)
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_val)  # Convert to int to ensure it's a number
        )
        
        # Capture both stdout and stderr
        output = result.stdout + result.stderr
        
        # Determine verification status
        if "VERIFICATION SUCCESSFUL" in output:
            status = "SUCCESSFUL"
        elif "VERIFICATION FAILED" in output:
            status = "FAILED"
            # Extract counterexample
            counterexample = ""
            if "[Counterexample]" in output:
                ce_start = output.find("[Counterexample]")
                counterexample = output[ce_start:]
                
                # Create a separate verification process output without the counterexample
                verification_process = output[:ce_start].strip()
                
                verification_result = {
                    "status": status,
                    "output": verification_process,
                    "counterexample": counterexample,
                    "command_used": command_str
                }
                # Store the last verification result
                last_verification_result = verification_result
                return verification_result
        else:
            status = "UNKNOWN"
        
        verification_result = {
            "status": status, 
            "output": output, 
            "command_used": command_str
        }
        # Store the last verification result
        last_verification_result = verification_result
        return verification_result
    except subprocess.TimeoutExpired:
        timeout_result = {
            "status": "TIMEOUT", 
            "message": f"Verification timed out after {timeout_val} seconds",
            "command_used": command_str if 'command_str' in locals() else "Command not executed due to timeout"
        }
        last_verification_result = timeout_result
        return timeout_result
    except Exception as e:
        error_result = {
            "status": "ERROR", 
            "message": str(e), 
            "command_used": command_str if 'command_str' in locals() else "Command not executed due to error"
        }
        last_verification_result = error_result
        return error_result

@app.post("/compare_code")
async def compare_code(request: Request):
    """Compare two pieces of code and return the differences"""
    data = await request.json()  # 获取 JSON 数据
    original_code = data.get("original_code", "")
    fixed_code = data.get("fixed_code", "")
    
    if not original_code or not fixed_code:
        return {"error": "Both original_code and fixed_code are required."}
    
    # 使用 difflib 生成差异
    diff = difflib.unified_diff(original_code.splitlines(), fixed_code.splitlines(), lineterm='', fromfile='original', tofile='fixed')

    # Format the diff with HTML for better display
    diff_html = ""
    for line in diff:
        if line.startswith('+'):
            diff_html += f'<div class="diff-line diff-added">{line}</div>'
        elif line.startswith('-'):
            diff_html += f'<div class="diff-line diff-removed">{line}</div>'
        elif line.startswith('@@'):
            diff_html += f'<div class="diff-line diff-info">{line}</div>'
        else:
            diff_html += f'<div class="diff-line">{line}</div>'

    return {"diff": "\n".join(diff), "diff_html": diff_html}

@app.post("/analyze_and_fix")
async def analyze_and_fix(request: Request):
    """Analyze the code and suggest fixes in one step with streaming response"""
    # Parse the request body
    data = await request.json()
    
    ai_model = data.get("ai_model", "")
    api_key = data.get("api_key", "")
    selected_files = data.get("selected_files", [])
    custom_prompt = data.get("custom_prompt", "")
    system_specification = data.get("system_specification", "")
    specific_model = data.get("specific_model", "")

    print(f"Received analyze_and_fix request: model={ai_model}, specific_model={specific_model}, files={selected_files}")

    if not ai_model or not api_key:
        return JSONResponse(content={"error": "AI model and API key are required"}, status_code=400)

    if not selected_files:
        return JSONResponse(content={"error": "No files selected for analysis"}, status_code=400)

    async def generate_response():
        try:
            # Send initial status
            print("Sending initial status")
            yield json.dumps({"status": "preparing", "message": "Preparing verification results..."}) + "\n"
            
            # Get verification results
            verification_results = await get_last_verification_result()
            
            if "error" in verification_results:
                print(f"Error in verification results: {verification_results['error']}")
                yield json.dumps({"status": "error", "message": verification_results["error"]}) + "\n"
                return
            
            print("Loading source files")
            yield json.dumps({"status": "loading_files", "message": "Loading source files..."}) + "\n"
            
            # Get source files content
            source_files_content = ""
            for filename in selected_files:
                file_path = os.path.join(TEMP_DIR, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    source_files_content += f"\nFile: {filename}\n```c\n{content}\n```\n"
            
            print("Preparing prompt")
            yield json.dumps({"status": "preparing_prompt", "message": "Preparing AI prompt..."}) + "\n"
            
            # Prepare the prompt
            if custom_prompt:
                prompt = custom_prompt
            else:
                # Use the template and replace placeholders
                prompt = default_ai_prompt_template
                
                # Get property content if available
                property_content = verification_results.get("property_content", "")
                
                # Replace placeholders
                if property_content:
                    prompt = prompt.replace("{property_section}", f"Property being verified:\n```c\n{property_content}\n```")
                else:
                    prompt = prompt.replace("{property_section}", "No specific property information available.")
                
                vr = verification_results.get("verification_results", {})
                prompt = prompt.replace("{status}", vr.get("status", "UNKNOWN"))
                prompt = prompt.replace("{command_used}", vr.get("command_used", "No command information available."))
                
                # Add output or counterexample
                if "counterexample" in vr:
                    prompt = prompt.replace("{output_section}", f"Counterexample:\n{vr['counterexample']}")
                elif "output" in vr:
                    prompt = prompt.replace("{output_section}", f"Verification Output:\n{vr['output']}")
                else:
                    prompt = prompt.replace("{output_section}", "No verification output available.")
                
                prompt = prompt.replace("{source_files_section}", f"Source Files:\n{source_files_content}")
                
                # Add system specification if provided
                if system_specification:
                    prompt += f"\n\nAdditional System Specification:\n{system_specification}"
            
            print(f"Calling {ai_model} API")
            yield json.dumps({"status": "calling_ai", "message": f"Calling {ai_model} API..."}) + "\n"
            
            # Call the appropriate AI model with streaming
            if ai_model == "chatgpt":
                model_name = specific_model or "gpt-4o"
                print(f"Using OpenAI model: {model_name}")
                yield json.dumps({"status": "model_selected", "message": f"Using OpenAI model: {model_name}"}) + "\n"
                
                # Stream the response
                async for chunk in call_chatgpt_api_streaming(prompt, api_key, model_name):
                    if chunk["type"] == "token":
                        yield json.dumps({
                            "status": "generating", 
                            "token": chunk["content"],
                            "current_response": chunk["full_response"]
                        }) + "\n"
                    elif chunk["type"] == "done":
                        yield json.dumps({
                            "status": "completed", 
                            "response": chunk["full_response"], 
                            "prompt_used": prompt
                        }) + "\n"
                
            elif ai_model == "claude":
                model_name = specific_model or "claude-3-7-sonnet-20250219"
                print(f"Using Claude model: {model_name}")
                yield json.dumps({"status": "model_selected", "message": f"Using Claude model: {model_name}"}) + "\n"
                
                # Stream the response
                async for chunk in call_claude_api_streaming(prompt, api_key, model_name):
                    if chunk["type"] == "token":
                        yield json.dumps({
                            "status": "generating", 
                            "token": chunk["content"],
                            "current_response": chunk["full_response"]
                        }) + "\n"
                    elif chunk["type"] == "done":
                        yield json.dumps({
                            "status": "completed", 
                            "response": chunk["full_response"], 
                            "prompt_used": prompt
                        }) + "\n"
                
            elif ai_model == "deepseek":
                model_name = specific_model or "deepseek-chat"
                print(f"Using DeepSeek model: {model_name}")
                yield json.dumps({"status": "model_selected", "message": f"Using DeepSeek model: {model_name}"}) + "\n"
                
                # Stream the response
                async for chunk in call_deepseek_api_streaming(prompt, api_key, model_name):
                    print(f"Received chunk: {chunk['type']}")
                    if chunk["type"] == "token":
                        yield json.dumps({
                            "status": "generating", 
                            "token": chunk["content"],
                            "current_response": chunk["full_response"]
                        }) + "\n"
                    elif chunk["type"] == "done":
                        print(f"Done with response: {len(chunk['full_response'])} chars")
                        yield json.dumps({
                            "status": "completed", 
                            "response": chunk["full_response"], 
                            "prompt_used": prompt
                        }) + "\n"
            else:
                yield json.dumps({"status": "error", "message": f"Unsupported AI model: {ai_model}"}) + "\n"
                return
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
    
    return StreamingResponse(generate_response(), media_type="application/x-ndjson")

async def call_deepseek_api_streaming(prompt, api_key, model="deepseek-chat"):
    """Call DeepSeek API with streaming response"""
    print(f"Calling DeepSeek API with model {model}")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a formal verification expert who can analyze code and suggest fixes."},
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"DeepSeek API error: {response.status} - {error_text}")
                raise Exception(f"DeepSeek API error: {response.status} - {error_text}")
            
            # Initialize response text
            response_text = ""
            
            # Process the streaming response
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if not line:
                    continue
                    
                print(f"DeepSeek raw line: {line}")
                
                if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                    try:
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    response_text += content
                                    print(f"DeepSeek token: {content}")
                                    yield {"type": "token", "content": content, "full_response": response_text}
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} for line: {line}")
                elif line.startswith("data: [DONE]"):
                    print("DeepSeek stream complete")
                    break
            
            print(f"DeepSeek final response length: {len(response_text)}")
            yield {"type": "done", "full_response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
