Face recognition CCTV project

Overview
- Flask backend serving a small web app with 3 pages: login, camera selection, and live stream.
- Face detection with MTCNN (from facenet-pytorch).
- Face embeddings from FaceNet (InceptionResnetV1 from facenet-pytorch).
- A simple classifier (KNN) trained on saved embeddings.
- Snapshots saved to a dataset folder with a SQLite metadata DB.

Quick start (Windows PowerShell)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
python app.py
```

3. Open http://127.0.0.1:5000 in the browser.

Notes and common issues
- Large PyTorch wheels: if pip fails to install torch, visit https://pytorch.org for the correct command for your system (CUDA vs CPU).
- If camera is busy or not accessible, close other apps using the webcam.
- If you see "CUDA" errors and you don't want GPU, install CPU-only torch.

Next steps
- Extend the login system (current scaffold uses a simple hardcoded user).
- Add persistent user accounts and permissions.
- Add background retraining or dataset management UI.
