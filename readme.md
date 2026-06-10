⚡ Setup & Installation
1. Prerequisites
Ensure you have Ollama installed on your system.

Download Ollama from ollama.com

Pull and host the Gemma model locally via terminal:

Bash
ollama run gemma
2. Environment Configuration
Clone this repository and create a clean virtual environment:

Bash
git clone [https://github.com/yourusername/privashield-rag.git](https://github.com/yourusername/privashield-rag.git)
cd privashield-rag

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\\Scripts\\activate

# Install required dependencies
pip install -r requirements.txt
3. Populating Data
Create a data/ directory in the project root if it doesn't already exist.

Drop 1 or more PDF files (e.g., sample agreements, compliance documentation, financial statements) into the data/ folder.

🛠️ Running the Application
You can execute the entire pipeline through the graphical dashboard:

Bash
streamlit run src/app.py
Workflow Steps:
When the UI opens, click the 🔄 Initialize/Refresh Vector DB button in the left sidebar. This fires up the ingestion pipeline, splits your PDFs into 1000-character chunks with a 200-character semantic overlap, builds the local vector embeddings, and initializes ChromaDB.

Type your compliance or risk evaluation query into the central prompt block.

Review the Analysis & Answer along with the Retrieved Source Context dropdown window to evaluate document source transparency.