# ⚖️ Themis – Legal Discovery AI Assistant

## Overview

Themis is an AI-powered legal discovery assistant designed to help professionals at a law firm navigate research, analyze documents, and gain actionable insights.  
Built with **Streamlit**, Themis provides a conversational interface for easy interaction.

## Features

- 💬 Chat interface with **typing animation** for a human-like assistant experience.  
- 🧭 Sidebar configuration panel for selecting assistant mode and clearing chat history.  
- 📊 Maintains **chat session history** within the browser session.  
- 🎨 Customizable **UI with chat bubbles, timestamps, and avatars**.  
- 🏛️ Professional layout tailored for legal teams.

## Installation

###  Step 1. Clone the repository:

```
git clone https://github.com/baong28/themis.git

cd themis
```

###  Step 2: Get a virtual environment
```
python -m venv "venv_name"

venv_name\Scripts\activate
```

###  Step 3: Install required libraries 
```
pip install -r requirements.txt
```

###  Step 4: Run the App
```
streamlit run app.py
```

## III. Project Structure
```
├── requirements.txt  # List of all required libraries
├── index.py          # Script to train the model 
├── app.py            # Script to run the Streamlit app to display model results                    
└── ask.py            # Ask Function
```

