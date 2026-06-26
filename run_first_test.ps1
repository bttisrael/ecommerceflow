cd "C:\Users\israb\Documents\commerceflow_ai"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --reload
