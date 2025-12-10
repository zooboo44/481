481 Project: MedSense AI
Ayman Sadek
Christian Ziobro
Peter Wongprasert
Yordy Raya
Members: Ayman Sadek, Christian Ziobro, Peter Wongprasert, Yordy Raya

Method 1: Docker (Recommended)
Prerequisites:
- Windows, MacOS, Linux
- Docker
    - Windows Installation https://docs.docker.com/desktop/setup/install/windows-install/
    - MacOS Installation https://docs.docker.com/desktop/setup/install/mac-install/
    - Linux Installation https://docs.docker.com/engine/install/

Running:
1. cd into the directory
2. run `docker compose up -d`
    - This builds the docker container and runs it on port `5051`
3. Open `http://127.0.0.1:5050` in a browser



Method 2: Build from source (Fail-safe)
Prerequisites:
- Windows, MacOS, Linux
- Python 3
- Pip
    - Ability to install packages

Environment Setup:
1. Install dependencies
    pip install -r requirements.txt

Running Program:

1. Run `app.py`
    python3 app.py

2. Open `http://127.0.0.1:5050` in a browser
