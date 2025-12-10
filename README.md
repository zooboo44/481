# Symptom Checker Chatbot - Web Interface

A web-based medical symptom checker chatbot built with Flask and scikit-learn.

# Method 1: Docker (Recommended)
## Setup

### Prerequisites
- Windows, MacOS, Linux
- Docker
    - [Windows Installation]
    - [MacOS Installation]
    - [Linux Installation]

### Running  
1. cd into the directory
1. run `docker compose up -d`
    - This builds the docker container and runs it on port `5051`
1. Open `http://127.0.0.1:5050` in a browser

# Method 2: Build from source (Fail-safe)

## Prerequisites
- Windows, MacOS, Linux
- Python 3
- Pip
    - Ability to install packages

### Environment Setup
1. Install dependencies

    ```
    pip install -r requirements.txt
    ```  

### Running Program

1. Run `app.py`

    ```
    python3 app.py
    ```

1. Open `127.0.0.1:5050` in a browser
