import streamlit as st
import base64
from openai import OpenAI
from typing import Optional, Dict, Union, List
import re
from datetime import datetime
import json
import os
import time

# Streamlit page config MUST be the first st command
st.set_page_config(
    page_title="Question Solver & Classifier",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HARDCODED API KEY - Replace with your actual API key
API_KEY = "sk-proj-ayJpqqcZIdIGoLIjI1HobyLZyAGj_A"  # REPLACE THIS WITH YOUR ACTUAL API KEY

# Custom CSS for better styling - FIXED TEXT COLOR ISSUE
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .answer-container {
        background-color: #f0f2f6 !important;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #000000 !important;  /* Force black text */
    }
    /* Force all text elements in answer container to be black */
    .answer-container * {
        color: #000000 !important;
    }
    .answer-container h1, .answer-container h2, .answer-container h3, 
    .answer-container h4, .answer-container h5, .answer-container h6 {
        color: #000000 !important;
        font-weight: bold;
    }
    .answer-container p, .answer-container span, .answer-container div {
        color: #000000 !important;
    }
    .answer-container li {
        color: #000000 !important;
    }
    .answer-container strong, .answer-container em {
        color: #000000 !important;
    }
    .answer-container code {
        background-color: #e8e8e8 !important;
        color: #000000 !important;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: monospace;
    }
    .answer-container pre {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
        margin: 15px 0;
    }
    .answer-container pre code {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    .answer-container blockquote {
        border-left: 4px solid #666 !important;
        margin: 15px 0;
        padding-left: 15px;
        color: #333333 !important;
    }
    .classification-box {
        background-color: #e0e6ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #1f1f1f;
    }
    .difficulty-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .level-1 { background-color: #90EE90; color: #2d5016; }
    .level-2 { background-color: #FFD700; color: #4B4B00; }
    .level-3 { background-color: #FFA500; color: #663300; }
    .level-4 { background-color: #FF6347; color: #660000; }
    .level-5 { background-color: #8B008B; color: #FFFFFF; }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
        color: #000000;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
        color: #000000;
    }
    /* Markdown styling within answer container */
    .answer-container h1 { font-size: 28px; margin-top: 20px; margin-bottom: 10px; }
    .answer-container h2 { font-size: 24px; margin-top: 18px; margin-bottom: 8px; }
    .answer-container h3 { font-size: 20px; margin-top: 16px; margin-bottom: 6px; }
    .answer-container h4 { font-size: 18px; margin-top: 14px; margin-bottom: 4px; }
    .answer-container p { margin: 10px 0; line-height: 1.6; }
    .answer-container ul, .answer-container ol { 
        margin: 10px 0; 
        padding-left: 30px;
        color: #000000 !important;
    }
    .answer-container li { margin: 5px 0; }
    /* Math display styling */
    .math-block {
        background-color: #f9f9f9 !important;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
        text-align: center;
        font-family: 'Times New Roman', serif;
        font-size: 18px;
        color: #000000 !important;
        border: 1px solid #ddd;
    }
    .math-inline {
        font-family: 'Times New Roman', serif;
        font-style: italic;
        padding: 0 3px;
        color: #000000 !important;
    }
    /* Model info box */
    .model-info-box {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #4169e1;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced handler class with conversation history
class GPT4OVisionHandler:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
    
    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def format_math_display(self, text: str) -> str:
        """Convert LaTeX math to readable format with proper styling"""
        # Handle display math first
        def replace_display_math(match):
            math_content = match.group(1)
            cleaned = self.clean_math_expression(math_content)
            return f'\n<div class="math-block">{cleaned}</div>\n'
        
        # Display math patterns
        text = re.sub(r'\$\$(.*?)\$\$', replace_display_math, text, flags=re.DOTALL)
        text = re.sub(r'\\\[(.*?)\\\]', replace_display_math, text, flags=re.DOTALL)
        
        # Handle inline math
        def replace_inline_math(match):
            math_content = match.group(1)
            cleaned = self.clean_math_expression(math_content)
            return f'<span class="math-inline">{cleaned}</span>'
        
        text = re.sub(r'\$([^\$]+)\$', replace_inline_math, text)
        text = re.sub(r'\\\(([^\)]+)\\\)', replace_inline_math, text)
        
        return text
    
    def clean_math_expression(self, expr: str) -> str:
        """Clean individual math expressions"""
        replacements = {
            r'\\frac{([^}]+)}{([^}]+)}': r'(\1)/(\2)',
            r'\\sqrt{([^}]+)}': r'√(\1)',
            r'\\sqrt\[([^\]]+)\]{([^}]+)}': r'\1√(\2)',
            r'\^{([^}]+)}': r'^(\1)',
            r'\_\{([^}]+)\}': r'_(\1)',
            r'\\cdot': '·',
            r'\\times': '×',
            r'\\div': '÷',
            r'\\pm': '±',
            r'\\mp': '∓',
            r'\\leq': '≤',
            r'\\geq': '≥',
            r'\\neq': '≠',
            r'\\approx': '≈',
            r'\\equiv': '≡',
            r'\\propto': '∝',
            r'\\infty': '∞',
            r'\\sum': 'Σ',
            r'\\prod': '∏',
            r'\\int': '∫',
            r'\\alpha': 'α',
            r'\\beta': 'β',
            r'\\gamma': 'γ',
            r'\\delta': 'δ',
            r'\\epsilon': 'ε',
            r'\\theta': 'θ',
            r'\\lambda': 'λ',
            r'\\mu': 'μ',
            r'\\pi': 'π',
            r'\\sigma': 'σ',
            r'\\phi': 'φ',
            r'\\omega': 'ω',
            r'\\Delta': 'Δ',
            r'\\Sigma': 'Σ',
            r'\\rightarrow': '→',
            r'\\leftarrow': '←',
            r'\\Rightarrow': '⇒',
            r'\\Leftarrow': '⇐',
            r'\\iff': '⇔',
            r'\\in': '∈',
            r'\\notin': '∉',
            r'\\subset': '⊂',
            r'\\cup': '∪',
            r'\\cap': '∩',
            r'\\text{([^}]+)}': r'\1',
            r'\\mathrm{([^}]+)}': r'\1',
            r'\\mathbf{([^}]+)}': r'**\1**',
            r'\\left\(': '(',
            r'\\right\)': ')',
            r'\\left\[': '[',
            r'\\right\]': ']',
            r'\\left\{': '{',
            r'\\right\}': '}',
            r'\\left\|': '||',
            r'\\right\|': '||',
        }
        
        cleaned = expr
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Handle boxed content specially
        cleaned = re.sub(r'\\boxed{([^}]+)}', r'[ANSWER: \1]', cleaned)
        
        # Remove any remaining backslashes before letters
        cleaned = re.sub(r'\\([a-zA-Z]+)', r'\1', cleaned)
        
        # Clean up spacing
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def format_for_display(self, text: str) -> str:
        """Enhanced formatting for better display"""
        # First handle math expressions
        text = self.format_math_display(text)
        
        # Handle markdown headings - convert to proper HTML
        text = re.sub(r'^#{6}\s+(.+)$', r'<h6>\1</h6>', text, flags=re.MULTILINE)
        text = re.sub(r'^#{5}\s+(.+)$', r'<h5>\1</h5>', text, flags=re.MULTILINE)
        text = re.sub(r'^#{4}\s+(.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^#{3}\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^#{2}\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^#{1}\s+(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        
        # Handle bold and italic
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'\_\_(.+?)\_\_', r'<strong>\1</strong>', text)
        text = re.sub(r'\_(.+?)\_', r'<em>\1</em>', text)
        
        # Handle code blocks
        text = re.sub(r'```(\w+)?\n(.*?)```', lambda m: f'<pre><code class="{m.group(1) or ""}">{m.group(2)}</code></pre>', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        # Handle lists
        # Unordered lists
        lines = text.split('\n')
        in_list = False
        new_lines = []
        
        for line in lines:
            if re.match(r'^\s*[-*+]\s+', line):
                if not in_list:
                    new_lines.append('<ul>')
                    in_list = True
                content = re.sub(r'^\s*[-*+]\s+', '', line)
                new_lines.append(f'<li>{content}</li>')
            else:
                if in_list and line.strip() == '':
                    continue
                elif in_list and not re.match(r'^\s*[-*+]\s+', line):
                    new_lines.append('</ul>')
                    in_list = False
                new_lines.append(line)
        
        if in_list:
            new_lines.append('</ul>')
        
        text = '\n'.join(new_lines)
        
        # Handle numbered lists
        text = re.sub(r'^(\d+)\.\s+(.+)$', r'<ol><li>\2</li></ol>', text, flags=re.MULTILINE)
        # Merge consecutive ol tags
        text = re.sub(r'</ol>\n<ol>', '\n', text)
        
        # Handle blockquotes
        text = re.sub(r'^>\s+(.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
        
        # Handle horizontal rules
        text = re.sub(r'^---+$', '<hr>', text, flags=re.MULTILINE)
        
        # Handle line breaks
        text = re.sub(r'\n\n', '</p><p>', text)
        text = f'<p>{text}</p>'
        
        # Clean up empty paragraphs
        text = re.sub(r'<p>\s*</p>', '', text)
        text = re.sub(r'<p>(<h[1-6]>)', r'\1', text)
        text = re.sub(r'(</h[1-6]>)</p>', r'\1', text)
        text = re.sub(r'<p>(<ul>|<ol>|<pre>|<blockquote>|<hr>)', r'\1', text)
        text = re.sub(r'(</ul>|</ol>|</pre>|</blockquote>)</p>', r'\1', text)
        
        return text
    
    def add_to_history(self, role: str, content: str, image_path: Optional[str] = None):
        """Add a message to conversation history"""
        if image_path:
            base64_image = self.encode_image(image_path)
            message_content = [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        else:
            message_content = content
        
        self.conversation_history.append({"role": role, "content": message_content})
    
    def get_conversation_messages(self, system_message: Optional[str] = None) -> List[Dict]:
        """Get properly formatted conversation messages for API call"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        return messages
    
    def answer_question(self, question: str, image_path: Optional[str] = None,
                       image_url: Optional[str] = None, max_completion_tokens: int = 1000,
                       use_history: bool = True, model_override: Optional[str] = None) -> str:
        
        # Enhanced system message for better formatting
        system_message = """You are a helpful AI assistant that solves mathematical and scientific problems uptill the end and give final solutions. 
        Provide clear, step-by-step solutions. Use proper markdown formatting:
        - Use # for main headings, ## for subheadings, ### for sub-subheadings
        - Use **bold** for emphasis and *italics* for terms
        - Use $inline math$ for inline equations and $$display math$$ for display equations
        - Use - or * for bullet points
        - Use 1. 2. 3. for numbered lists
        - Use > for important notes or quotes
        - Use ``` for code blocks
        - Structure your response clearly with proper sections"""
        
        # Add current question to history
        self.add_to_history("user", question, image_path)
        
        # Get messages with history if enabled
        if use_history:
            messages = self.get_conversation_messages(system_message)
        else:
            # Just use current question
            messages = [{"role": "system", "content": system_message}]
            content = [{"type": "text", "text": question}]
            
            if image_path:
                base64_image = self.encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            elif image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            messages.append({"role": "user", "content": content})
        
        try:
            # Use model override if provided, otherwise use instance model
            model_to_use = model_override if model_override else self.model
            
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_completion_tokens=max_completion_tokens
            )
            
            # FIXED: Extract the actual content from the response
            answer = response.choices[0].message.content
            
            # Check if the response was cut off due to length
            if response.choices[0].finish_reason == 'length':
                answer += "\n\n[Note: Response was truncated due to length limit. Consider increasing max_completion_tokens for longer responses.]"
            
            # Check if answer is empty
            if not answer or answer.strip() == "":
                answer = "I apologize, but I couldn't generate a response. This might be due to an issue with the model or API. Please try again or check your API configuration."
            
            # Add assistant's response to history
            if use_history:
                self.add_to_history("assistant", answer)
            
            return answer
        except Exception as e:
            error_msg = f"Error calling OpenAI API: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg
    
    def classify_question(self, question: str, image_path: Optional[str] = None, 
                         image_url: Optional[str] = None) -> Dict[str, Union[int, str]]:
        classification_prompt = f"""Analyze this question and provide:
1. Difficulty level (1-5)
2. Subject category

**Difficulty Levels:**
1 - Basic Arithmetic
2 - Elementary Problem Solving
3 - Moderate Conceptual Thinking
4 - Good level of calculations and reasoning and thinking, But not too intense.
5 - Advanced Reasoning / Multi-Step Logic / Intense Multi-Concept Reasoning

**Subject Categories:**
- general: General knowledge
- maths: Mathematics
- physics: Physics
- biology: Biology
- chemistry: Chemistry
- prover: Mathematical proofs

Question: {question}

Output format:
Level: [1-5]
Subject: [category]"""

        messages = [
            {"role": "system", "content": "You are an expert question classifier."}
        ]
        
        content = [{"type": "text", "text": classification_prompt}]
        
        if image_path:
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        elif image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        messages.append({"role": "user", "content": content})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=100
            )
            
            result_text = response.choices[0].message.content
            level_match = re.search(r'Level:\s*(\d)', result_text)
            subject_match = re.search(r'Subject:\s*(\w+)', result_text)
            
            level = int(level_match.group(1)) if level_match else 0
            subject = subject_match.group(1).lower() if subject_match else "unknown"
            
            return {
                "level": level,
                "subject": subject,
                "raw_response": result_text
            }
        except Exception as e:
            return {
                "level": 0,
                "subject": "error",
                "raw_response": f"Error: {str(e)}"
            }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# Function to display formatted answer
def display_formatted_answer(answer: str, container):
    """Display answer with proper formatting"""
    handler = GPT4OVisionHandler(API_KEY)
    formatted_answer = handler.format_for_display(answer)
    
    with container:
        st.markdown(
            f'<div class="answer-container">{formatted_answer}</div>',
            unsafe_allow_html=True
        )

# Function to determine which model to use based on settings
def get_model_for_request(model_type: str, classification_level: int = 0) -> str:
    """
    Determine which model to use based on settings
    Model 1: Always use gpt-4o-mini
    Model 2: Use gpt-4o-mini for levels 1-4, gpt-4o for level 5
    """
    if model_type == "Model 1: Ours":
        return "gpt-4o-mini"
    elif model_type == "Model 2: Ours+API":
        if classification_level == 5:
            return "gpt-4o"  # Use gpt-4o for level 5 questions
        else:
            return "gpt-4o-mini"  # Use gpt-4o-mini for levels 1-4
    else:
        return "gpt-4o-mini"  # Default

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'qa_handler' not in st.session_state:
    st.session_state.qa_handler = GPT4OVisionHandler(API_KEY, model="gpt-4o-mini")

if 'classifier_handler' not in st.session_state:
    st.session_state.classifier_handler = GPT4OVisionHandler(API_KEY, model="gpt-4o-mini")

if 'model_type' not in st.session_state:
    st.session_state.model_type = "Model 1: Ours"

# Title and description
st.title("🧮 Intelligent Question Solver & Classifier")
st.markdown("Upload an image or type a question to get step-by-step solutions with difficulty classification")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    model_type = st.radio(
        "Choose Model Type:",
        ["Model 1: Ours", "Model 2: Ours+API"],
        index=0 if st.session_state.model_type == "Model 1: Ours" else 1,
        help="Model 1: Always uses gpt-4o-mini\nModel 2: Uses gpt-4o-mini for levels 1-4, gpt-4o for level 5 questions"
    )
    st.session_state.model_type = model_type
    
    # Display model info
    if model_type == "Model 1: Ours":
        st.info("📌 Model 1. ")
    else:
        st.info("📌 Model 2.")

    with st.expander("Advanced Settings"):
        max_completion_tokens = st.slider("Max Response Length", 100, 8000, 2000, 50)
        use_conversation_history = st.checkbox("Use Conversation History", value=True,
                                             help="Include previous Q&As in context")
    
    show_history = st.checkbox("Show History", value=True)
    show_conversation = st.checkbox("Show Conversation Context", value=False)
    
    # Clear conversation button
    if st.button("🔄 Clear Conversation History"):
        st.session_state.qa_handler.clear_history()
        st.info("Conversation history cleared!")
    
    st.markdown("---")
    st.markdown("### 📊 Session Statistics")
    
    if st.session_state.history:
        total_questions = len(st.session_state.history)
        st.metric("Total Questions", total_questions)
        
        levels = [h['level'] for h in st.session_state.history if h.get('level', 0) > 0]
        if levels:
            avg_difficulty = sum(levels) / len(levels)
            st.metric("Average Difficulty", f"{avg_difficulty:.1f}/5")
        
        subjects = [h['subject'] for h in st.session_state.history if h.get('subject')]
        if subjects:
            st.markdown("**Subject Distribution:**")
            subject_counts = {}
            for s in subjects:
                subject_counts[s] = subject_counts.get(s, 0) + 1
            for subject, count in sorted(subject_counts.items()):
                st.write(f"- {subject.title()}: {count}")
        
        # Model usage stats
        if any(h.get('model_used') for h in st.session_state.history):
            st.markdown("**Model Usage:**")
            model_counts = {}
            for h in st.session_state.history:
                model = h.get('model_used', 'Unknown')
                model_counts[model] = model_counts.get(model, 0) + 1
            for model, count in model_counts.items():
                st.write(f"- {model}: {count}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Question")
    
    # Show conversation context if enabled
    if show_conversation and st.session_state.qa_handler.conversation_history:
        st.markdown("**Current Conversation:**")
        for msg in st.session_state.qa_handler.conversation_history:
            role = msg['role']
            content = msg['content']
            if isinstance(content, list):
                content = content[0]['text'] if content and 'text' in content[0] else "Image"
            
            if role == "user":
                st.markdown(f'<div class="chat-message user-message">👤 {content}</div>', 
                          unsafe_allow_html=True)
            else:
                # Format assistant messages
                formatted_content = st.session_state.qa_handler.format_for_display(content)
                st.markdown(f'<div class="chat-message assistant-message">🤖 {formatted_content}</div>', 
                          unsafe_allow_html=True)
        st.markdown("---")
    
    question = st.text_area("Enter your question:", height=100,
                           placeholder="Type your question here or upload an image with a question...")
    
    uploaded_file = st.file_uploader("Upload an image (optional)", 
                                   type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                                   help="Upload an image containing a question or problem")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        solve_classify = st.button("🎯 Solve & Classify", type="primary", use_container_width=True)
    
    with col_btn2:
        solve_only = st.button("💡 Solve Only", use_container_width=True)
    
    with col_btn3:
        classify_only = st.button("📊 Classify Only", use_container_width=True)

# Process the request
if solve_classify or solve_only or classify_only:
    if not question and not uploaded_file:
        st.error("Please enter a question or upload an image!")
    else:
        image_path = None
        if uploaded_file:
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing..."):
            try:
                if solve_classify:
                    # First classify the question
                    classification = st.session_state.classifier_handler.classify_question(
                        question if question else "Classify the problem shown in this image",
                        image_path=image_path
                    )
                    
                    # Determine which model to use based on classification and settings
                    model_to_use = get_model_for_request(st.session_state.model_type, classification['level'])
                    
                    # Display model info
                    st.markdown(f'<div class="model-info-box">🤖 Using model: <strong>{model_to_use}</strong></div>', 
                              unsafe_allow_html=True)
                    
                    # Use Q&A handler for solving with the selected model
                    answer = st.session_state.qa_handler.answer_question(
                        question if question else "Solve the problem shown in this image",
                        image_path=image_path,
                        max_completion_tokens=max_completion_tokens,
                        use_history=use_conversation_history,
                        model_override=model_to_use
                    )
                    
                    # Display classification
                    with col2:
                        st.header("📊 Classification")
                        
                        level = classification['level']
                        difficulty_descriptions = {
                            1: "Basic Arithmetic",
                            2: "Elementary Problem Solving",
                            3: "Moderate Conceptual Thinking",
                            4: "Good level of calculations",
                            5: "Intense Multi-Concept Reasoning"
                        }
                        
                        st.markdown(f"""
                        <div class="classification-box">
                            <h4>Difficulty Level</h4>
                            <div class="difficulty-badge level-{level}">
                                Level {level}: {difficulty_descriptions.get(level, 'Unknown')}
                            </div>
                            <h4 style="margin-top: 15px;">Subject</h4>
                            <div style="font-size: 18px; font-weight: bold; color: #1f4788;">
                                {classification['subject'].title()}
                            </div>
                            <h4 style="margin-top: 15px;">Model Used</h4>
                            <div style="font-size: 16px; color: #4169e1;">
                                {model_to_use}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display answer with proper formatting
                    st.header("💡 Solution")
                    answer_container = st.container()
                    display_formatted_answer(answer, answer_container)
                    
                    # Add to history
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'question': question if question else "Image question",
                        'answer': answer,
                        'level': classification['level'],
                        'subject': classification['subject'],
                        'model_used': model_to_use
                    })
                    
                elif solve_only:
                    # For solve only, use default model based on settings (no classification)
                    model_to_use = get_model_for_request(st.session_state.model_type, 0)
                    
                    # Display model info
                    st.markdown(f'<div class="model-info-box">🤖 Using model: <strong>{model_to_use}</strong></div>', 
                              unsafe_allow_html=True)
                    
                    answer = st.session_state.qa_handler.answer_question(
                        question if question else "Solve the problem shown in this image",
                        image_path=image_path,
                        max_completion_tokens=max_completion_tokens,
                        use_history=use_conversation_history,
                        model_override=model_to_use
                    )
                    
                    st.header("💡 Solution")
                    answer_container = st.container()
                    display_formatted_answer(answer, answer_container)
                    
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'question': question if question else "Image question",
                        'answer': answer,
                        'level': 0,
                        'subject': 'unknown',
                        'model_used': model_to_use
                    })
                    
                elif classify_only:
                    classification = st.session_state.classifier_handler.classify_question(
                        question if question else "Classify the problem shown in this image",
                        image_path=image_path
                    )
                    
                    with col2:
                        st.header("📊 Classification")
                        
                        level = classification['level']
                        difficulty_descriptions = {
                            1: "Basic Arithmetic",
                            2: "Elementary Problem Solving",
                            3: "Moderate Conceptual Thinking",
                            4: "Good level of calculations",
                            5: "Intense Multi-Concept Reasoning"
                        }
                        
                        # Show which model would be used
                        model_would_use = get_model_for_request(st.session_state.model_type, level)
                        
                        st.markdown(f"""
                        <div class="classification-box">
                            <h4>Difficulty Level</h4>
                            <div class="difficulty-badge level-{level}">
                                Level {level}: {difficulty_descriptions.get(level, 'Unknown')}
                            </div>
                            <h4 style="margin-top: 15px;">Subject</h4>
                            <div style="font-size: 18px; font-weight: bold; color: #1f4788;">
                                {classification['subject'].title()}
                            </div>
                            <h4 style="margin-top: 15px;">Model Would Use</h4>
                            <div style="font-size: 16px; color: #4169e1;">
                                {model_would_use}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if "api" in str(e).lower():
                    st.info("Please check that the API key is correctly configured in the code.")
        
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

# History section
if show_history and st.session_state.history:
    st.markdown("---")
    st.header("📜 History")
    
    for idx, item in enumerate(reversed(st.session_state.history[-10:])):
        with st.expander(f"{item['timestamp']} - {item['question'][:50]}..."):
            col_h1, col_h2 = st.columns([3, 1])
            
            with col_h1:
                st.markdown("**Question:**")
                st.write(item["question"])
                st.markdown("**Answer:**")
                # Display formatted answer in history
                formatted = st.session_state.qa_handler.format_for_display(item["answer"])
                st.markdown(
                    f'<div class="answer-container">{formatted}</div>',
                    unsafe_allow_html=True
                )
            
            with col_h2:
                if item['level'] > 0:
                    st.markdown("**Classification:**")
                    st.write(f"Level: {item['level']}")
                    st.write(f"Subject: {item['subject'].title()}")
                if item.get('model_used'):
                    st.write(f"Model: {item['model_used']}")

if st.session_state.history:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
st.markdown("Powered by Advanced AI Technology")