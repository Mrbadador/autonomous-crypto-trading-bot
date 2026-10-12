"""
SOVEREIGN SWARM — TWO-WAY MOBILE AGENT GATEWAY & REMOTE TASK RUNNER (V1.0)
Equipped with @security-auditor and @concurrency-expert

Features:
- Autonomous Agentic Execution Loop via Gemini / Groq LLMs
- Safe Asynchronous Tool Execution (Commands, Database, Files, Audits)
- Real-Time WebSocket Streaming (Thoughts, Tool Execution, Output, Diffs)
- Strict HMAC / Password Authentication & Rate Limiting
- Full Mobile-First Responsive Chat Interface (iOS / Android PWA)
"""

import os
import sys
import time
import json
import hmac
import re
import secrets
import sqlite3
import asyncio
import logging
import traceback
import subprocess
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC

from fastapi import FastAPI, Request, Response, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

APP_PASSWORD = os.getenv("APP_PASSWORD", "SovereignSwarm2026!")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8766"))
SESSION_SECRET = os.getenv("SESSION_SECRET_KEY", secrets.token_hex(32))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] AGENT_GATEWAY: %(message)s")
log = logging.getLogger("agent_gateway")

app = FastAPI(title="Sovereign Swarm Remote Agent Gateway", docs_url=None, redoc_url=None)

# -------------------------------------------------------------------
# AUTHENTICATION & SECURITY (@security-auditor)
# -------------------------------------------------------------------

VALID_SESSIONS: Dict[str, float] = {}
LOGIN_ATTEMPTS: Dict[str, List[float]] = {}
SESSION_TTL_HOURS = 72

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    attempts = [t for t in LOGIN_ATTEMPTS.get(ip, []) if now - t < 900]
    LOGIN_ATTEMPTS[ip] = attempts
    return len(attempts) < 10

def record_failed(ip: str):
    now = time.time()
    if ip not in LOGIN_ATTEMPTS: LOGIN_ATTEMPTS[ip] = []
    LOGIN_ATTEMPTS[ip].append(now)

def create_token() -> str:
    tok = secrets.token_urlsafe(32)
    VALID_SESSIONS[tok] = time.time() + (SESSION_TTL_HOURS * 3600)
    return tok

def verify_token(tok: Optional[str]) -> bool:
    if not tok or tok not in VALID_SESSIONS: return False
    if time.time() > VALID_SESSIONS[tok]:
        del VALID_SESSIONS[tok]
        return False
    return True

# -------------------------------------------------------------------
# ASYNC TOOL EXECUTION SUITE (@concurrency-expert)
# -------------------------------------------------------------------

async def tool_run_command(cmd: str, timeout: int = 45) -> str:
    """Executes a shell command asynchronously in the workspace directory."""
    log.info(f"[TOOL:run_command] Executing: {cmd}")
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode('utf-8', errors='replace')
        if not output.strip():
            output = f"(Command completed with exit code {proc.returncode} and no output)"
        return output[:4000]  # Cap output size
    except asyncio.TimeoutError:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error running command: {e}"

async def tool_read_file(filepath: str, max_lines: int = 150) -> str:
    """Reads a file relative to the workspace."""
    try:
        full_path = os.path.abspath(os.path.join(BASE_DIR, filepath))
        if not os.path.exists(full_path):
            return f"Error: File '{filepath}' does not exist."
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        content = "".join([f"{i+1:4d} | {l}" for i, l in enumerate(lines[:max_lines])])
        if len(lines) > max_lines:
            content += f"\n... (truncated {len(lines) - max_lines} remaining lines)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"

async def tool_write_file(filepath: str, content: str) -> str:
    """Writes or edits a file relative to the workspace."""
    try:
        full_path = os.path.abspath(os.path.join(BASE_DIR, filepath))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {filepath}."
    except Exception as e:
        return f"Error writing file: {e}"

async def tool_query_database(query: str, db_name: str = "trading_data.db") -> str:
    """Executes a SQL query against a local SQLite database."""
    try:
        db_path = os.path.join(BASE_DIR, db_name)
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                if not rows:
                    return "Query returned 0 rows."
                cols = rows[0].keys()
                header = " | ".join(cols)
                divider = " | ".join(["---"] * len(cols))
                data_rows = []
                for r in rows[:25]:
                    data_rows.append(" | ".join([str(r[c]) for c in cols]))
                return f"{header}\n{divider}\n" + "\n".join(data_rows)
            else:
                conn.commit()
                return f"Statement executed successfully. Rows affected: {cursor.rowcount}"
    except Exception as e:
        return f"Database error: {e}"

async def tool_list_directory(subpath: str = "") -> str:
    """Lists files in the workspace directory."""
    try:
        target_dir = os.path.abspath(os.path.join(BASE_DIR, subpath))
        entries = os.listdir(target_dir)
        res = []
        for e in sorted(entries):
            p = os.path.join(target_dir, e)
            if os.path.isdir(p):
                res.append(f"📁 {e}/")
            else:
                sz = os.path.getsize(p)
                res.append(f"📄 {e} ({sz:,} bytes)")
        return "\n".join(res[:50])
    except Exception as e:
        return f"Error listing directory: {e}"

# -------------------------------------------------------------------
# AGENTIC REASONING PIPELINE
# -------------------------------------------------------------------

SYSTEM_PROMPT = """You are Antigravity Sovereign Agent, an expert AI engineer and algorithmic trading commander operating directly on the user's host machine.
You have full capability to inspect files, execute terminal commands, query the SQLite trading databases, and audit the trading bot.

When you need to perform an action, output a tool call block in EXACTLY this JSON format:
```tool_call
{
  "tool": "run_command" | "read_file" | "write_file" | "query_database" | "list_directory",
  "args": { ... }
}
```

Tool Specifications:
1. `run_command`: {"cmd": "command to run in powershell/bash"} (e.g. `python check_activity.py`, `git status`)
2. `read_file`: {"filepath": "relative path to file", "max_lines": 100}
3. `write_file`: {"filepath": "relative path", "content": "content"}
4. `query_database`: {"query": "SELECT ...", "db_name": "trading_data.db"}
5. `list_directory`: {"subpath": ""}

Guidelines:
- Reason step-by-step before calling tools.
- When the user asks a question, look up live data first (query database or run a verification command) before answering.
- Keep final responses clear, actionable, and formatted in clean markdown.
- When done, provide your final answer without any tool_call block.
"""

async def run_agent_loop(user_prompt: str, ws: WebSocket) -> str:
    """Runs a multi-turn agentic tool loop streaming updates to WebSocket."""
    from ai_engine import query_cloud_llm

    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    max_steps = 6
    step = 0
    final_response = ""

    while step < max_steps:
        step += 1
        await ws.send_json({"type": "status", "content": f"Reasoning step {step}/{max_steps}..."})
        
        # Build conversational history prompt for LLM
        prompt_text = ""
        for m in messages:
            prompt_text += f"\n\n--- {m['role'].upper()} ---\n{m['content']}"

        try:
            raw_output = await query_cloud_llm(prompt_text, SYSTEM_PROMPT)
        except Exception as e:
            await ws.send_json({"type": "error", "content": f"LLM Inference error: {e}"})
            return f"Error communicating with AI Brain: {e}"

        # Parse tool calls
        tool_call_match = re.search(r"```tool_call\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        
        if tool_call_match:
            thought_text = raw_output[:tool_call_match.start()].strip()
            if thought_text:
                await ws.send_json({"type": "thought", "content": thought_text})
                
            try:
                tool_data = json.loads(tool_call_match.group(1))
                tool_name = tool_data.get("tool")
                tool_args = tool_data.get("args", {})
                
                await ws.send_json({"type": "tool_start", "tool": tool_name, "args": tool_args})
                
                # Execute tool
                tool_result = ""
                if tool_name == "run_command":
                    cmd = tool_args.get("cmd", "")
                    tool_result = await tool_run_command(cmd)
                elif tool_name == "read_file":
                    fp = tool_args.get("filepath", "")
                    ml = tool_args.get("max_lines", 100)
                    tool_result = await tool_read_file(fp, ml)
                elif tool_name == "write_file":
                    fp = tool_args.get("filepath", "")
                    cnt = tool_args.get("content", "")
                    tool_result = await tool_write_file(fp, cnt)
                elif tool_name == "query_database":
                    q = tool_args.get("query", "")
                    db = tool_args.get("db_name", "trading_data.db")
                    tool_result = await tool_query_database(q, db)
                elif tool_name == "list_directory":
                    sp = tool_args.get("subpath", "")
                    tool_result = await tool_list_directory(sp)
                else:
                    tool_result = f"Error: Unknown tool '{tool_name}'"
                
                await ws.send_json({"type": "tool_output", "tool": tool_name, "output": tool_result})
                
                # Append to history
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": f"Tool '{tool_name}' output:\n```\n{tool_result}\n```\nContinue reasoning or provide the final answer."})
                
            except Exception as pe:
                await ws.send_json({"type": "error", "content": f"Failed executing tool call: {pe}"})
                messages.append({"role": "user", "content": f"Tool execution failed: {pe}. Provide response with available info."})
        else:
            # Final response reached
            final_response = raw_output
            await ws.send_json({"type": "response", "content": final_response})
            break

    return final_response

# -------------------------------------------------------------------
# REST & WEBSOCKET ENDPOINTS
# -------------------------------------------------------------------

class LoginReq(BaseModel):
    password: str

@app.post("/api/auth/login")
async def api_login(req: LoginReq, request: Request, response: Response):
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Too many attempts. Locked for 15 minutes.")
    
    if hmac.compare_digest(req.password.strip(), APP_PASSWORD.strip()):
        token = create_token()
        response.set_cookie(key="agent_session", value=token, max_age=SESSION_TTL_HOURS * 3600, httponly=True, samesite="lax")
        return {"status": "ok", "token": token}
    else:
        record_failed(ip)
        raise HTTPException(status_code=401, detail="Invalid App Password")

@app.websocket("/ws/agent")
async def ws_agent_chat(websocket: WebSocket):
    await websocket.accept()
    token = websocket.query_params.get("token")
    if not verify_token(token):
        await websocket.send_json({"type": "error", "content": "Authentication failed. Token invalid."})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.send_json({"type": "system", "content": "🤖 Sovereign Swarm Remote Agent connected and ready."})

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                prompt = msg.get("prompt", "").strip()
                if not prompt: continue
                
                await websocket.send_json({"type": "user_echo", "content": prompt})
                await run_agent_loop(prompt, websocket)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON message received."})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error: {e}")

# -------------------------------------------------------------------
# MOBILE CHAT UI (PWA)
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_agent_chat_ui():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="theme-color" content="#090d16">
  <title>Sovereign Agent Mobile Gateway</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/lucide@latest/dist/umd/lucide.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; -webkit-tap-highlight-color: transparent; }
    .font-mono { font-family: 'JetBrains Mono', monospace; }
    .glass-card { background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .pulse-cyan { box-shadow: 0 0 20px rgba(6, 182, 212, 0.35); }
  </style>
</head>
<body class="bg-[#090d16] text-slate-100 min-h-screen flex flex-col justify-between select-none">

  <!-- AUTH MODAL -->
  <div id="auth-modal" class="fixed inset-0 z-50 bg-[#090d16]/95 backdrop-blur-xl flex flex-col items-center justify-center p-6">
    <div class="w-full max-w-sm glass-card p-8 rounded-3xl text-center shadow-2xl border border-cyan-500/20">
      <div class="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-tr from-cyan-500 to-indigo-600 p-0.5 shadow-lg pulse-cyan">
        <div class="w-full h-full bg-[#090d16] rounded-2xl flex items-center justify-center">
          <i data-lucide="bot" class="w-10 h-10 text-cyan-400"></i>
        </div>
      </div>
      <h2 class="text-2xl font-bold tracking-tight text-white mb-1">Remote Agent</h2>
      <p class="text-xs text-slate-400 mb-6">Two-Way Command Gateway</p>
      
      <div class="space-y-4">
        <input type="password" id="password-input" placeholder="App Password" 
               class="w-full bg-slate-900 border border-slate-700 rounded-2xl px-5 py-4 text-center text-lg font-mono text-white focus:outline-none focus:border-cyan-400" />
        <button onclick="handleLogin()" id="login-btn"
                class="w-full bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 text-white font-semibold py-4 rounded-2xl shadow-lg transition active:scale-95 flex items-center justify-center gap-2">
          <span>Connect Agent</span>
          <i data-lucide="arrow-right" class="w-5 h-5"></i>
        </button>
        <div id="auth-error" class="text-xs text-rose-400 hidden pt-2 font-medium"></div>
      </div>
    </div>
  </div>

  <!-- CHAT CONTAINER -->
  <div id="app-container" class="hidden flex-1 flex flex-col h-screen max-w-md mx-auto w-full">
    
    <!-- HEADER -->
    <header class="sticky top-0 z-40 bg-[#090d16]/90 backdrop-blur-md px-5 py-3 border-b border-white/5 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-3 h-3 rounded-full bg-cyan-400 animate-pulse"></div>
        <div>
          <div class="text-xs font-bold tracking-wider text-slate-400 uppercase">Sovereign Swarm</div>
          <div class="text-sm font-semibold text-white flex items-center gap-1.5">
            <span>Remote Commander</span>
            <span class="text-[10px] bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded-full font-mono">Agentic</span>
          </div>
        </div>
      </div>
      <button onclick="handleLogout()" class="p-2 rounded-xl bg-slate-800 text-slate-300 hover:text-rose-400 active:scale-95 transition">
        <i data-lucide="log-out" class="w-4 h-4"></i>
      </button>
    </header>

    <!-- QUICK ACTION CHIPS -->
    <div class="px-4 py-2 flex gap-2 overflow-x-auto no-scrollbar border-b border-white/5 bg-slate-900/40">
      <button onclick="sendQuickPrompt('What are my active positions on Bitget and current account balance?')" class="whitespace-nowrap bg-slate-800/80 hover:bg-slate-700 text-[11px] text-slate-300 px-3 py-1.5 rounded-full border border-white/5 active:scale-95">📊 Active Positions</button>
      <button onclick="sendQuickPrompt('Run a quick health check on all 18 bot subprocesses')" class="whitespace-nowrap bg-slate-800/80 hover:bg-slate-700 text-[11px] text-slate-300 px-3 py-1.5 rounded-full border border-white/5 active:scale-95">⚡ Swarm Health</button>
      <button onclick="sendQuickPrompt('Audit recent 24H closed trades and calculate net PnL and win rate')" class="whitespace-nowrap bg-slate-800/80 hover:bg-slate-700 text-[11px] text-slate-300 px-3 py-1.5 rounded-full border border-white/5 active:scale-95">📈 24H Performance</button>
      <button onclick="sendQuickPrompt('Emergency Halt: verify if RISK_HALT is active in system_state')" class="whitespace-nowrap bg-rose-900/40 hover:bg-rose-800/60 text-[11px] text-rose-300 px-3 py-1.5 rounded-full border border-rose-500/20 active:scale-95">🛑 Check Risk Gate</button>
    </div>

    <!-- MESSAGES FEED -->
    <main id="chat-feed" class="flex-1 overflow-y-auto p-4 space-y-4">
      <div class="text-center py-6 text-xs text-slate-500">
        <i data-lucide="sparkles" class="w-5 h-5 mx-auto mb-2 text-cyan-400"></i>
        <span>Ask anything. The agent can run commands, audit code, and query databases.</span>
      </div>
    </main>

    <!-- INPUT BAR -->
    <footer class="p-3 bg-[#090d16]/95 border-t border-white/5 backdrop-blur-md">
      <form id="chat-form" onsubmit="handleSend(event)" class="flex items-center gap-2">
        <input type="text" id="user-input" placeholder="Prompt your remote agent..." 
               class="flex-1 bg-slate-900 border border-slate-700/80 rounded-2xl px-4 py-3.5 text-sm text-white focus:outline-none focus:border-cyan-400 transition" />
        <button type="submit" id="send-btn"
                class="bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 text-white p-3.5 rounded-2xl active:scale-95 transition flex items-center justify-center">
          <i data-lucide="send" class="w-5 h-5"></i>
        </button>
      </form>
    </footer>

  </div>

  <script>
    let token = localStorage.getItem('agent_token') || '';
    let ws = null;

    lucide.createIcons();

    if (token) {
      document.getElementById('auth-modal').classList.add('hidden');
      document.getElementById('app-container').classList.remove('hidden');
      initWs();
    }

    async function handleLogin() {
      const pwd = document.getElementById('password-input').value;
      const btn = document.getElementById('login-btn');
      const err = document.getElementById('auth-error');
      
      btn.disabled = true;
      err.classList.add('hidden');

      try {
        const res = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: pwd })
        });
        const data = await res.json();
        if (res.ok && data.token) {
          token = data.token;
          localStorage.setItem('agent_token', token);
          document.getElementById('auth-modal').classList.add('hidden');
          document.getElementById('app-container').classList.remove('hidden');
          initWs();
        } else {
          err.innerText = data.detail || 'Authentication failed';
          err.classList.remove('hidden');
        }
      } catch (e) {
        err.innerText = 'Connection failed.';
        err.classList.remove('hidden');
      } finally {
        btn.disabled = false;
      }
    }

    function handleLogout() {
      localStorage.removeItem('agent_token');
      token = '';
      if (ws) ws.close();
      location.reload();
    }

    function initWs() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${proto}//${window.location.host}/ws/agent?token=${token}`;
      ws = new WebSocket(wsUrl);

      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        renderMessage(msg);
      };

      ws.onclose = () => {
        setTimeout(initWs, 3000);
      };
    }

    function appendBubble(html, isUser = false) {
      const feed = document.getElementById('chat-feed');
      const div = document.createElement('div');
      div.className = isUser ? "flex justify-end" : "flex justify-start";
      div.innerHTML = `
        <div class="max-w-[90%] ${isUser ? 'bg-indigo-600 text-white rounded-3xl rounded-br-sm p-4 text-sm' : 'glass-card text-slate-200 rounded-3xl rounded-bl-sm p-4 text-xs space-y-2'} shadow-lg">
          ${html}
        </div>
      `;
      feed.appendChild(div);
      feed.scrollTop = feed.scrollHeight;
      lucide.createIcons();
    }

    let activeAgentContainer = null;

    function renderMessage(msg) {
      const feed = document.getElementById('chat-feed');
      
      if (msg.type === 'user_echo') {
        appendBubble(`<div>${escapeHtml(msg.content)}</div>`, true);
        activeAgentContainer = null;
      } else if (msg.type === 'status') {
        // Status indicator
        let statusEl = document.getElementById('agent-status-badge');
        if (!statusEl) {
          const div = document.createElement('div');
          div.id = 'agent-status-badge';
          div.className = 'text-[11px] font-mono text-cyan-400 flex items-center gap-1.5 py-1';
          div.innerHTML = `<i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i><span>${escapeHtml(msg.content)}</span>`;
          feed.appendChild(div);
        } else {
          statusEl.querySelector('span').innerText = msg.content;
        }
        feed.scrollTop = feed.scrollHeight;
        lucide.createIcons();
      } else if (msg.type === 'thought') {
        clearStatus();
        const html = `
          <details class="cursor-pointer">
            <summary class="text-[11px] font-mono text-slate-400 flex items-center gap-1">
              <i data-lucide="brain" class="w-3.5 h-3.5 text-indigo-400"></i>
              <span>Agent Reasoning</span>
            </summary>
            <div class="text-[11px] font-mono text-slate-300 pt-2 border-t border-white/5 whitespace-pre-wrap">${escapeHtml(msg.content)}</div>
          </details>
        `;
        appendBubble(html, false);
      } else if (msg.type === 'tool_start') {
        clearStatus();
        const argsStr = JSON.stringify(msg.args, null, 2);
        const html = `
          <div class="font-mono text-[11px] bg-slate-900/90 p-3 rounded-2xl border border-cyan-500/20 space-y-1">
            <div class="flex items-center gap-1.5 text-cyan-400 font-bold">
              <i data-lucide="terminal" class="w-3.5 h-3.5"></i>
              <span>Tool: ${escapeHtml(msg.tool)}</span>
            </div>
            <div class="text-slate-400 text-[10px] overflow-x-auto">${escapeHtml(argsStr)}</div>
          </div>
        `;
        appendBubble(html, false);
      } else if (msg.type === 'tool_output') {
        clearStatus();
        const html = `
          <div class="font-mono text-[11px] bg-black/80 p-3 rounded-2xl border border-white/10 space-y-1 text-slate-300">
            <div class="text-[10px] text-emerald-400 font-bold flex items-center gap-1">
              <i data-lucide="check-circle" class="w-3 h-3"></i>
              <span>Output (${escapeHtml(msg.tool)})</span>
            </div>
            <pre class="overflow-x-auto text-[10px] text-slate-300 max-h-40 overflow-y-auto leading-tight">${escapeHtml(msg.output)}</pre>
          </div>
        `;
        appendBubble(html, false);
      } else if (msg.type === 'response') {
        clearStatus();
        const html = `
          <div class="prose prose-invert max-w-none text-xs leading-relaxed">
            ${marked.parse(msg.content)}
          </div>
        `;
        appendBubble(html, false);
      } else if (msg.type === 'error') {
        clearStatus();
        appendBubble(`<div class="text-rose-400 font-mono text-xs flex items-center gap-1.5"><i data-lucide="alert-triangle" class="w-4 h-4"></i><span>${escapeHtml(msg.content)}</span></div>`, false);
      }
    }

    function clearStatus() {
      const statusEl = document.getElementById('agent-status-badge');
      if (statusEl) statusEl.remove();
    }

    function handleSend(e) {
      e.preventDefault();
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text || !ws) return;
      
      ws.send(JSON.stringify({ prompt: text }));
      input.value = '';
    }

    function sendQuickPrompt(text) {
      if (!ws) return;
      ws.send(JSON.stringify({ prompt: text }));
    }

    function escapeHtml(text) {
      if (!text) return '';
      return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("agent_mobile_bridge:app", host="0.0.0.0", port=AGENT_PORT, reload=False)
