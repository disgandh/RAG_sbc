import requests
import json
import os
from pathlib import Path
import gradio as gr
from datetime import datetime
import numpy as np
from typing import List, Dict, Any

# Langchain imports for RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class OllamaHMRRAG:
    def __init__(self, model_name="llama3.2:3b", ollama_host="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.vector_db = None
        self.embeddings = None
        
    def check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                if self.model_name not in available_models:
                    print(f"Model {self.model_name} not found. Available models: {available_models}")
                    print(f"To download the model, run: ollama pull {self.model_name}")
                    return False
                    
                print(f"✓ Connected to Ollama. Using model: {self.model_name}")
                return True
            else:
                print("✗ Cannot connect to Ollama. Make sure it's running.")
                return False
                
        except requests.exceptions.ConnectionError:
            print("✗ Ollama not running. Start it with: ollama serve")
            return False

    def generate_with_ollama(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate response using Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            "options": {
                "num_predict": 500,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()['response']
            
        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama: {str(e)}"

    def create_training_examples(self) -> List[Dict]:
        """Create training examples for fine-tuning context"""
        return [
            {
                "intent": "Add P-Asserted-Identity using From header",
                "sip_msg": "INVITE sip:bob@oracle.com SIP/2.0\nFrom: <sip:alice@telco.com>",
                "hmr": """header-rules
  name                                    addPAI
  header-name                             p-asserted-identity
  action                                  add
  comparison-type                         pattern-rule
  msg-type                                request
  methods                                 INVITE
  new-value                               "<sip:"+$From.$From_er.$0+"@telco.com>\""""
            },
            {
                "intent": "Remove Diversion Header",
                "sip_msg": "INVITE sip:bob@oracle.com SIP/2.0\nDiversion: <sip:olduser@domain.com>",
                "hmr": """header-rules
  name                                    removeDiversion
  header-name                             diversion
  action                                  remove
  comparison-type                         pattern-rule
  msg-type                                request
  methods                                 INVITE"""
            },
            {
                "intent": "Replace domain in From header to newdomain.com",
                "sip_msg": "From: <sip:user@olddomain.com>",
                "hmr": """header-rules
  name                                    replaceFromDomain
  header-name                             from
  action                                  replace
  comparison-type                         pattern-rule
  msg-type                                request
  methods                                 INVITE
  match-value                             "olddomain.com"
  new-value                               "newdomain.com\""""
            },
            {
                "intent": "Add Contact header with custom domain",
                "sip_msg": "REGISTER sip:example.com SIP/2.0\nFrom: <sip:user@example.com>",
                "hmr": """header-rules
  name                                    addContact
  header-name                             contact
  action                                  add
  comparison-type                         pattern-rule
  msg-type                                request
  methods                                 REGISTER
  new-value                               "<sip:user@newdomain.com>\""""
            },
            {
                "intent": "Modify Content-Type for SDP manipulation",
                "sip_msg": "INVITE sip:bob@oracle.com SIP/2.0\nContent-Type: application/sdp",
                "hmr": """header-rules
  name                                    modify_sdp
  header-name                             Content-Type
  action                                  manipulate
  comparison-type                         case-sensitive
  msg-type                                request
  methods                                 INVITE
  element-rules
    name                                  add_fmtp
    parameter-name                        application/sdp
    type                                  mime
    action                                find-replace-all
    match-val-type                        any
    comparison-type                       pattern-rule
    match-value                           m=audio.*
    new-value                             $0+$CRLF+"a=fmtp:18 annexb=no\""""
            }
        ]

    def advanced_chunking(self, documents: List[Document]) -> List[Document]:
        """Advanced chunking strategy for HMR documents"""
        hmr_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=[
                "\nheader-rules\n",
                "\nelement-rules\n", 
                "\nsip-manipulation\n",
                "\nmime-sdp-rules\n",
                "\nmime-isup-rules\n",
                "\nmime-header-rules\n",
                "\nmime-rules\n",
                "\nisup-param-rules\n",
                "\nsdp-session-rules\n",
                "\nsdp-line-rules\n",
                "\nsdp-media-rules\n",
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )
        
        return hmr_splitter.split_documents(documents)

    def extract_hmr_metadata(self, chunk_text: str) -> Dict[str, str]:
        """Extract metadata from HMR chunks"""
        metadata = {}
        
        # Rule type detection
        rule_types = {
            "header-rules": "header-rule",
            "element-rules": "element-rule", 
            "sip-manipulation": "sip-manipulation",
            "mime-sdp-rules": "mime-sdp-rule",
            "mime-isup-rules": "mime-isup-rule",
            "mime-header-rules": "mime-header-rule",
            "mime-rules": "mime-rule",
            "isup-param-rules": "isup-param-rule",
            "sdp-session-rules": "sdp-session-rule",
            "sdp-line-rules": "sdp-line-rule",
            "sdp-media-rules": "sdp-media-rule"
        }
        
        for rule_pattern, rule_type in rule_types.items():
            if rule_pattern in chunk_text:
                metadata["rule_type"] = rule_type
                break
        
        # Action detection
        actions = ["add", "delete", "find-replace-all", "manipulate", "store", 
                  "log", "none", "monitor", "reject", "sip-manip", "remove",
                  "delete-element", "delete-header", "replace"]
        
        for action in actions:
            if f"action {' ' * 10}{action}" in chunk_text or f"action{' ' * 20}{action}" in chunk_text:
                metadata["action"] = action
                break
        
        # Header detection
        headers = ["from", "to", "contact", "p-asserted-identity", "diversion", 
                  "content-type", "via", "route", "record-route"]
        for header in headers:
            if f"header-name {' ' * 10}{header}" in chunk_text.lower():
                metadata["header"] = header
                break
        
        return metadata

    def setup_vector_database(self, data_dir: str = "data/text/") -> FAISS:
        """Setup vector database from HMR documents"""
        print("🔍 Setting up vector database...")
        
        # Load documents
        documents = []
        data_path = Path(data_dir)
        
        if data_path.exists():
            for file_path in data_path.glob("*.txt"):
                try:
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"✓ Loaded {file_path.name}")
                except Exception as e:
                    print(f"✗ Error loading {file_path}: {e}")
        else:
            print(f"Directory {data_dir} not found. Creating sample documents...")
        
        # Add training examples as documents
        training_examples = self.create_training_examples()
        for example in training_examples:
            doc_content = f"Intent: {example['intent']}\nSIP Message: {example['sip_msg']}\nHMR:\n{example['hmr']}"
            documents.append(Document(page_content=doc_content, metadata={"source": "training_example"}))
        
        if not documents:
            print("No documents found. Creating minimal sample data...")
            sample_docs = self.create_sample_documents()
            documents.extend(sample_docs)
        
        # Advanced chunking
        chunks = self.advanced_chunking(documents)
        
        # Add metadata
        for chunk in chunks:
            hmr_metadata = self.extract_hmr_metadata(chunk.page_content)
            chunk.metadata.update(hmr_metadata)
        
        # Create embeddings
        print("🔗 Creating embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector database
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        
        print(f"✅ Vector database created with {len(chunks)} chunks")
        return self.vector_db

    def create_sample_documents(self) -> List[Document]:
        """Create sample documents for testing"""
        sample_data = [
            """header-rules
  name                                    addFromDomain  
  header-name                             from
  action                                  add
  comparison-type                         pattern-rule
  msg-type                                request
  methods                                 INVITE
  new-value                               "domain.com\"""",
            
            """header-rules
  name                                    removePAI
  header-name                             p-asserted-identity  
  action                                  remove
  comparison-type                         case-sensitive
  msg-type                                request
  methods                                 INVITE""",
            
            """element-rules
  name                                    modifySDPCodec
  type                                    mime
  action                                  find-replace-all
  match-val-type                          any
  comparison-type                         pattern-rule
  match-value                             "m=audio.*"
  new-value                               "m=audio 5004 RTP/AVP 0 18\""""
        ]
        
        return [Document(page_content=content, metadata={"source": "sample"}) for content in sample_data]

    def extract_hmr_keywords(self, query: str) -> List[str]:
        """Extract HMR-specific keywords"""
        hmr_keywords = [
            "header-rules", "element-rules", "sip-manipulation",
            "add", "remove", "replace", "manipulate", "store", "delete",
            "from", "to", "contact", "p-asserted-identity", "diversion", "via",
            "invite", "register", "bye", "cancel", "ack", "options",
            "content-type", "route", "record-route"
        ]
        
        found_keywords = []
        query_lower = query.lower()
        for keyword in hmr_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        return found_keywords

    def hybrid_retrieval(self, query: str, k: int = 8) -> List[Document]:
        """Enhanced retrieval combining semantic and keyword search"""
        if not self.vector_db:
            raise ValueError("Vector database not initialized. Call setup_vector_database() first.")
        
        # Semantic search
        semantic_docs = self.vector_db.similarity_search(query, k=k//2)
        
        # Keyword-enhanced search
        keywords = self.extract_hmr_keywords(query)
        if keywords:
            keyword_query = " ".join(keywords)
            keyword_docs = self.vector_db.similarity_search(keyword_query, k=k//2)
        else:
            keyword_docs = []
        
        # Combine and remove duplicates
        all_docs = semantic_docs + keyword_docs
        unique_docs = self.remove_duplicates(all_docs)
        
        # Re-rank documents
        reranked_docs = self.rerank_documents(query, unique_docs)
        
        return reranked_docs[:k]

    def remove_duplicates(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents"""
        seen = set()
        unique_docs = []
        
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs

    def rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents based on HMR-specific criteria"""
        if not docs:
            return docs
            
        scores = []
        query_lower = query.lower()
        
        for doc in docs:
            score = 1.0  # Base score
            content = doc.page_content.lower()
            
            # Boost complete HMR blocks
            if "header-rules" in content and "name" in content:
                score += 3.0
            elif "element-rules" in content and "name" in content:
                score += 2.5
            
            # Boost similar actions
            actions = ["add", "remove", "replace", "modify", "manipulate", "delete"]
            for action in actions:
                if action in query_lower and action in content:
                    score += 2.0
                    break
            
            # Boost similar headers
            headers = ["from", "to", "contact", "p-asserted-identity", "diversion", "via", "content-type"]
            for header in headers:
                if header in query_lower and header in content:
                    score += 1.5
                    break
            
            # Boost SIP methods
            methods = ["invite", "register", "bye", "cancel", "options"]
            for method in methods:
                if method in query_lower and method in content:
                    score += 1.0
                    break
            
            scores.append(score)
        
        # Sort by score (descending)
        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs]

    def create_hmr_prompt(self, intent: str, sip_msg: str, context_docs: List[Document]) -> str:
        """Create optimized prompt for HMR generation"""
        
        # Prepare context from retrieved documents
        context_examples = []
        for i, doc in enumerate(context_docs[:3]):  # Use top 3 most relevant
            context_examples.append(f"Example {i+1}:\n{doc.page_content}\n")
        
        context_str = "\n".join(context_examples) if context_examples else "No specific examples found."
        
        # Get training examples for few-shot learning
        training_examples = self.create_training_examples()
        few_shot_examples = []
        
        # Select most relevant training examples
        for example in training_examples[:2]:  # Use 2 best examples
            few_shot_examples.append(f"""
Intent: {example['intent']}
SIP Message: {example['sip_msg']}
Generated HMR:
{example['hmr']}
""")
        
        few_shot_str = "\n---\n".join(few_shot_examples)
        
        prompt = f"""You are an Oracle Session Border Controller (SBC) expert specializing in Header Manipulation Rules (HMR).

Your task is to generate syntactically correct Oracle SBC HMR configuration based on the provided intent and SIP message.

CONTEXT FROM KNOWLEDGE BASE:
{context_str}

EXAMPLES OF CORRECT HMR GENERATION:
{few_shot_str}

REQUIREMENTS:
1. Generate valid Oracle SBC CLI format
2. Use appropriate rule types (header-rules, element-rules, etc.)
3. Include all required parameters (name, action, comparison-type, msg-type, etc.)
4. Follow Oracle SBC best practices
5. Handle the specific SIP scenario correctly

CURRENT TASK:
Intent: {intent}
SIP Message: {sip_msg if sip_msg else "Not provided"}

ANALYSIS:
1. Identify the SIP headers/elements that need modification
2. Determine the appropriate HMR action (add/remove/replace/manipulate)
3. Consider message type and method restrictions
4. Apply proper pattern matching and value substitution

Generate the Oracle SBC HMR configuration:"""

        return prompt

    def generate_hmr(self, intent: str, sip_msg: str = "") -> str:
        """Generate HMR using Ollama with RAG"""
        if not self.check_ollama_connection():
            return "Error: Cannot connect to Ollama. Please ensure it's running and the model is available."
        
        try:
            # Retrieve relevant context
            query = f"{intent} {sip_msg}".strip()
            relevant_docs = self.hybrid_retrieval(query, k=5)
            
            # Create optimized prompt
            prompt = self.create_hmr_prompt(intent, sip_msg, relevant_docs)
            
            # Generate with Ollama
            response = self.generate_with_ollama(prompt, temperature=0.1)
            
            # Clean up the response
            cleaned_response = self.clean_hmr_response(response)
            
            return cleaned_response
            
        except Exception as e:
            return f"Error generating HMR: {str(e)}"

    def clean_hmr_response(self, response: str) -> str:
        """Clean and format the HMR response"""
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        # If the response contains the prompt echo, remove it
        if "Generate the Oracle SBC HMR configuration:" in response:
            response = response.split("Generate the Oracle SBC HMR configuration:")[-1].strip()
        
        # Remove any explanatory text after the HMR block
        lines = response.split('\n')
        hmr_lines = []
        in_hmr_block = False
        
        for line in lines:
            # Start of HMR block
            if any(rule_type in line for rule_type in ['header-rules', 'element-rules', 'sip-manipulation']):
                in_hmr_block = True
                hmr_lines.append(line)
            # Continue HMR block
            elif in_hmr_block and (line.strip().startswith(' ') or line.strip() == '' or 
                                 any(param in line for param in ['name', 'action', 'header-name', 'comparison-type', 'msg-type', 'methods', 'match-value', 'new-value'])):
                hmr_lines.append(line)
            # End of HMR block
            elif in_hmr_block and line.strip() and not line.strip().startswith(' '):
                break
            # Before HMR block
            elif not in_hmr_block:
                continue
        
        return '\n'.join(hmr_lines) if hmr_lines else response

    def evaluate_hmr_quality(self, generated_hmr: str, intent: str = "") -> Dict[str, float]:
        """Evaluate generated HMR quality"""
        metrics = {
            "syntax_score": self.check_syntax_correctness(generated_hmr),
            "completeness_score": self.check_completeness(generated_hmr),
            "oracle_compliance": self.check_oracle_format(generated_hmr),
            "intent_alignment": self.check_intent_alignment(generated_hmr, intent)
        }
        
        # Overall score
        metrics["overall_score"] = sum(metrics.values()) / len(metrics)
        
        return metrics

    def check_syntax_correctness(self, hmr: str) -> float:
        """Check if HMR follows Oracle SBC syntax"""
        required_elements = ["name", "action", "comparison-type", "msg-type"]
        optional_elements = ["header-name", "methods", "match-value", "new-value"]
        
        score = 0
        hmr_lower = hmr.lower()
        
        # Check required elements
        for element in required_elements:
            if element in hmr_lower:
                score += 0.2
        
        # Check for proper indentation (Oracle SBC uses spaces)
        lines = hmr.split('\n')
        proper_indentation = sum(1 for line in lines[1:] if line.startswith('  ') or line.strip() == '')
        if proper_indentation > 0:
            score += 0.1
        
        # Check for proper rule block start
        if any(rule_type in hmr_lower for rule_type in ['header-rules', 'element-rules']):
            score += 0.1
        
        return min(score, 1.0)

    def check_completeness(self, hmr: str) -> float:
        """Check completeness of HMR configuration"""
        score = 0
        hmr_lower = hmr.lower()
        
        # Check for rule name
        if "name" in hmr_lower:
            score += 0.3
        
        # Check for action
        actions = ["add", "remove", "replace", "manipulate", "delete", "store"]
        if any(action in hmr_lower for action in actions):
            score += 0.3
        
        # Check for target (header-name or similar)
        if any(target in hmr_lower for target in ["header-name", "parameter-name"]):
            score += 0.2
        
        # Check for message context
        if "msg-type" in hmr_lower:
            score += 0.2
        
        return min(score, 1.0)

    def check_oracle_format(self, hmr: str) -> float:
        """Check Oracle SBC format compliance"""
        score = 0
        
        # Check for proper rule block declaration
        if any(rule_type in hmr for rule_type in ['header-rules', 'element-rules', 'sip-manipulation']):
            score += 0.4
        
        # Check for proper parameter spacing (Oracle uses specific spacing)
        lines = hmr.split('\n')
        proper_format_lines = 0
        for line in lines[1:]:  # Skip first line (rule declaration)
            if line.strip() and ('  ' in line[:20] or line.strip().startswith('name') or line.strip().startswith('action')):
                proper_format_lines += 1
        
        if proper_format_lines > 0:
            score += 0.4
        
        # Check for reasonable rule name (no special characters, reasonable length)
        if "name" in hmr.lower():
            score += 0.2
        
        return min(score, 1.0)

    def check_intent_alignment(self, hmr: str, intent: str) -> float:
        """Check if HMR aligns with the stated intent"""
        if not intent:
            return 0.5  # Neutral score if no intent provided
        
        score = 0
        intent_lower = intent.lower()
        hmr_lower = hmr.lower()
        
        # Check action alignment
        if "add" in intent_lower and "add" in hmr_lower:
            score += 0.3
        elif "remove" in intent_lower and ("remove" in hmr_lower or "delete" in hmr_lower):
            score += 0.3
        elif "replace" in intent_lower and "replace" in hmr_lower:
            score += 0.3
        elif "modify" in intent_lower and ("manipulate" in hmr_lower or "find-replace" in hmr_lower):
            score += 0.3
        
        # Check header alignment
        headers = ["from", "to", "contact", "p-asserted-identity", "diversion", "via", "content-type"]
        for header in headers:
            if header in intent_lower and header in hmr_lower:
                score += 0.4
                break
        
        # Check method alignment
        methods = ["invite", "register", "bye", "cancel", "options","message","notify","prack","subscribe","publish","refer","update"]
        for method in methods:
            if method in intent_lower and method in hmr_lower:
                score += 0.3
                break
        
        return min(score, 1.0)

    def create_gradio_interface(self):
        """Create enhanced Gradio interface"""
        def generate_and_evaluate(intent, sip_msg):
            if not intent.strip():
                return "Please provide an intent.", "", ""
            
            try:
                # Generate HMR
                hmr = self.generate_hmr(intent, sip_msg)
                
                # Evaluate quality
                metrics = self.evaluate_hmr_quality(hmr, intent)
                
                # Format metrics
                metrics_str = f"""📊 Quality Metrics:
• Overall Score: {metrics['overall_score']:.2f}/1.0
• Syntax Score: {metrics['syntax_score']:.2f}/1.0
• Completeness: {metrics['completeness_score']:.2f}/1.0  
• Oracle Compliance: {metrics['oracle_compliance']:.2f}/1.0
• Intent Alignment: {metrics['intent_alignment']:.2f}/1.0

💡 Score Guide:
• 0.8+ : Excellent
• 0.6-0.8 : Good  
• 0.4-0.6 : Fair
• <0.4 : Needs improvement"""
                
                # Add recommendations
                recommendations = []
                if metrics['syntax_score'] < 0.6:
                    recommendations.append("• Check parameter names and indentation")
                if metrics['completeness_score'] < 0.6:
                    recommendations.append("• Verify all required parameters are present")
                if metrics['oracle_compliance'] < 0.6:
                    recommendations.append("• Review Oracle SBC format requirements")
                if metrics['intent_alignment'] < 0.6:
                    recommendations.append("• Ensure HMR matches the stated intent")
                
                if recommendations:
                    recommendations_str = "\n🔧 Recommendations:\n" + "\n".join(recommendations)
                else:
                    recommendations_str = "\n✅ HMR looks good!"
                
                return hmr, metrics_str, recommendations_str
                
            except Exception as e:
                return f"Error: {str(e)}", "Error in evaluation", ""
                '''
        
        '''def load_example(example_name):
            examples = {
                "Add P-Asserted-Identity": (
                    "Add P-Asserted-Identity header using From header value",
                    "INVITE sip:bob@oracle.com SIP/2.0\nFrom: <sip:alice@telco.com>"
                ),
                "Remove Diversion": (
                    "Remove Diversion header from INVITE requests", 
                    "INVITE sip:bob@oracle.com SIP/2.0\nDiversion: <sip:olduser@domain.com>"
                ),
                "Replace Domain": (
                    "Replace domain in From header to newdomain.com",
                    "INVITE sip:user@example.com SIP/2.0\nFrom: <sip:user@olddomain.com>"
                ),
                "Add Contact": (
                    "Add Contact header for REGISTER requests",
                    "REGISTER sip:example.com SIP/2.0\nFrom: <sip:user@example.com>"
                )
            }
            
            if example_name in examples:
                return examples[example_name][0], examples[example_name][1]
            return "", ""
        '''
        
        with gr.Blocks(title="Oracle SBC HMR Generator") as interface:
            gr.Markdown("""
            # 🔧 Oracle SBC Header Manipulation Rule Generator
            
            Generate Oracle Session Border Controller Header Manipulation Rules using AI and RAG technology.
            Powered by Ollama and optimized for Oracle SBC configurations.
            """, elem_classes=["main-header"])
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("Input Configuration")
                    
                    intent_input = gr.Textbox(
                        label="Intent",
                        placeholder="Describe what you want the HMR to do...\nExample: Add P-Asserted-Identity header using From header",
                        lines=3,
                        info="Clearly describe the desired HMR functionality"
                    )
                    
                    sip_msg_input = gr.Textbox(
                        label="SIP Message (Optional)",
                        placeholder="Provide relevant SIP message content...\nExample: INVITE sip:bob@oracle.com SIP/2.0\nFrom: <sip:alice@telco.com>",
                        lines=6,
                        info="Include relevant SIP headers and message content"
                    )
                    
                    with gr.Row():
                        generate_btn = gr.Button("🚀 Generate HMR", variant="primary", size="lg")
                        
                    
                    '''gr.Markdown("### 📚 Quick Examples")
                    example_dropdown = gr.Dropdown(
                        choices=["Add P-Asserted-Identity", "Remove Diversion", "Replace Domain", "Add Contact"],
                        label="Load Example",
                        info="Select a pre-defined example to get started"
                    )'''
                    
                    '''# Model Status
                    with gr.Accordion("🔧 System Status", open=False):
                        status_text = gr.Textbox(
                            value="Click 'Check Status' to verify Ollama connection",
                            label="Ollama Status",
                            interactive=False
                        )
                        check_status_btn = gr.Button("Check Status")'''

                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Generated HMR Configuration")
                    
                    hmr_output = gr.Code(
                        label="Generated HMR",
                        lines=20,
                        show_label=True
                    )
                    
                    with gr.Row():
                        copy_btn = gr.Button("📋 Copy to Clipboard", size="sm")
                        save_btn = gr.Button("💾 Save HMR", size="sm")
                    
                    '''metrics_output = gr.Textbox(
                        label="📊 Quality Assessment",
                        lines=12,
                        elem_classes=["metrics-output"],
                        interactive=False
                    )
                    
                    recommendations_output = gr.Textbox(
                        label="💡 Recommendations",
                        lines=5,
                        interactive=False
                    )'''

            # Event handlers
            generate_btn.click(
                generate_and_evaluate,
                inputs=[intent_input, sip_msg_input],
                outputs=[hmr_output, metrics_output, recommendations_output]
            )
            
            example_dropdown.change(
                load_example,
                inputs=[example_dropdown],
                outputs=[intent_input, sip_msg_input]
            )
            
            '''def check_system_status():
                if self.check_ollama_connection():
                    return "✅ Ollama connected successfully! Model ready for use."
                else:
                    return "❌ Cannot connect to Ollama. Please ensure it's running and model is available."
            
            check_status_btn.click(
                check_system_status,
                outputs=[status_text]
            )
            
            # Add footer information
            gr.Markdown("""
            ---
            ### 📖 How to Use:
            1. **Describe your intent** - What should the HMR accomplish?
            2. **Provide SIP context** - Include relevant SIP message parts (optional but helpful)
            3. **Generate HMR** - Click the generate button to create the configuration
            4. **Review quality** - Check the metrics and recommendations
            5. **Copy & Deploy** - Use the generated HMR in your Oracle SBC
            
            ### 🔧 Prerequisites:
            - Ollama must be running (`ollama serve`)
            - Llama 3.2 model must be available (`ollama pull llama3.2:3b`)
            
            ### 💡 Tips:
            - Be specific in your intent description
            - Include relevant SIP headers in the message field
            - Use the quality metrics to validate the generated HMR
            """)
        '''
        return interface

    def save_hmr_to_file(self, hmr_content: str, filename: str = None) -> str:
        """Save generated HMR to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_hmr_{timestamp}.txt"
        
        try:
            output_dir = Path("generated_hmrs")
            output_dir.mkdir(exist_ok=True)
            
            file_path = output_dir / filename
            file_path.write_text(hmr_content)
            
            return f"HMR saved to: {file_path}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

    def fine_tune_with_ollama(self, training_data_path: str = None):
        """
        Fine-tune Ollama model with custom HMR data
        Note: This requires creating a Modelfile and using ollama create
        """
        print("🔧 Setting up Ollama fine-tuning...")
        
        # Create training examples in Ollama format
        training_examples = self.create_training_examples()
        
        # Generate Modelfile content
        modelfile_content = f"""FROM {self.model_name}

# Set parameters for HMR generation
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# System message for HMR generation
SYSTEM \"\"\"You are an Oracle Session Border Controller (SBC) expert specializing in Header Manipulation Rules (HMR). 
Generate syntactically correct Oracle SBC CLI format configurations based on user intents and SIP message context.
Always follow Oracle SBC best practices and proper parameter formatting.\"\"\"

"""
        
        # Add training examples as few-shot prompts
        for i, example in enumerate(training_examples):
            modelfile_content += f"""
# Example {i+1}
TEMPLATE \"\"\"### Instruction:
Generate Oracle SBC Header Manipulation Rule based on the intent and SIP message

### Input:
Intent: {example['intent']}
SIP Message: {example['sip_msg']}

### Response:
{example['hmr']}
\"\"\"

"""
        
        # Save Modelfile
        modelfile_path = Path("Modelfile_HMR")
        modelfile_path.write_text(modelfile_content)
        
        print(f"✅ Modelfile created: {modelfile_path}")
        print("To create the custom model, run:")
        print(f"ollama create hmr-specialist -f {modelfile_path}")
        
        return str(modelfile_path)

    def benchmark_model_performance(self) -> Dict[str, Any]:
        """Benchmark the model performance on test cases"""
        test_cases = [
            {
                "intent": "Add P-Asserted-Identity from From header",
                "sip_msg": "INVITE sip:bob@oracle.com SIP/2.0\nFrom: <sip:alice@telco.com>",
                "expected_elements": ["p-asserted-identity", "add", "from"]
            },
            {
                "intent": "Remove Diversion header",
                "sip_msg": "INVITE sip:bob@oracle.com SIP/2.0\nDiversion: <sip:olduser@domain.com>",
                "expected_elements": ["diversion", "remove"]
            },
            {
                "intent": "Replace domain in Contact header",
                "sip_msg": "REGISTER sip:example.com SIP/2.0\nContact: <sip:user@olddomain.com>",
                "expected_elements": ["contact", "replace", "domain"]
            }
        ]
        
        results = []
        total_score = 0
        
        print("🧪 Running performance benchmark...")
        
        for i, test_case in enumerate(test_cases):
            print(f"Test {i+1}/{len(test_cases)}: {test_case['intent']}")
            
            # Generate HMR
            generated_hmr = self.generate_hmr(test_case['intent'], test_case['sip_msg'])
            
            # Evaluate
            metrics = self.evaluate_hmr_quality(generated_hmr, test_case['intent'])
            
            # Check for expected elements
            expected_found = sum(1 for element in test_case['expected_elements'] 
                               if element.lower() in generated_hmr.lower())
            expected_score = expected_found / len(test_case['expected_elements'])
            
            test_result = {
                "test_case": test_case['intent'],
                "generated_hmr": generated_hmr,
                "metrics": metrics,
                "expected_score": expected_score,
                "overall_score": (metrics['overall_score'] + expected_score) / 2
            }
            
            results.append(test_result)
            total_score += test_result['overall_score']
        
        benchmark_summary = {
            "individual_results": results,
            "average_score": total_score / len(test_cases),
            "total_tests": len(test_cases),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"📊 Benchmark completed. Average score: {benchmark_summary['average_score']:.2f}")
        
        return benchmark_summary


def main():
    """Main function to run the HMR RAG system"""
    print("🚀 Initializing Oracle SBC HMR Generator with Ollama...")
    
    # Initialize the system
    hmr_system = OllamaHMRRAG(model_name="llama3.2:3b")
    
    # Check Ollama connection
    if not hmr_system.check_ollama_connection():
        print("❌ Cannot proceed without Ollama connection.")
        print("Please ensure:")
        print("1. Ollama is running: ollama serve")
        print("2. Model is available: ollama pull llama3.2:3b")
        return
    
    # Setup vector database
    print("\n📚 Setting up knowledge base...")
    try:
        hmr_system.setup_vector_database("data/text/")
        print("✅ Knowledge base ready!")
    except Exception as e:
        print(f"⚠️  Warning: Knowledge base setup failed: {e}")
        print("Continuing with basic examples...")
        hmr_system.setup_vector_database()
    
    # Test generation
    print("\n🧪 Testing HMR generation...")
    test_intent = "Add P-Asserted-Identity header using From header"
    test_sip = "INVITE sip:bob@oracle.com SIP/2.0\nFrom: <sip:alice@telco.com>"
    
    try:
        result = hmr_system.generate_hmr(test_intent, test_sip)
        print("Generated HMR:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
        # Evaluate quality
        metrics = hmr_system.evaluate_hmr_quality(result, test_intent)
        print(f"\nQuality Score: {metrics['overall_score']:.2f}/1.0")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Launch Gradio interface
    print("\n🌐 Launching web interface...")
    try:
        interface = hmr_system.create_gradio_interface()
        interface.launch()
    except Exception as e:
        print(f"❌ Failed to launch interface: {e}")


# Additional utility functions
def create_modelfile_for_hmr():
    """Create a specialized Modelfile for HMR generation"""
    hmr_system = OllamaHMRRAG()
    modelfile_path = hmr_system.fine_tune_with_ollama()
    print(f"Modelfile created at: {modelfile_path}")


def run_benchmark():
    """Run performance benchmark"""
    hmr_system = OllamaHMRRAG()
    hmr_system.setup_vector_database()
    results = hmr_system.benchmark_model_performance()
    
    # Save results
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Benchmark results saved to benchmark_results.json")


if __name__ == "__main__":
    # You can run different modes:
    # main()                    # Full system with GUI
    # create_modelfile_for_hmr() # Create Ollama Modelfile
    # run_benchmark()           # Run performance tests
    
    main()