# PLY Processing Backend

This is a FastAPI backend for processing PLY files and markers.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python app.py
```

The server will run on http://localhost:5000.

## API Documentation

FastAPI automatically generates interactive API documentation. After starting the server, you can access:

- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

## API Endpoints

### POST /process_markers
Process markers and return a modified PLY file.

**Request:**
- Form data with:
  - `markers`: JSON string of marker data
  - `ply_file`: (Optional) PLY file to process

**Response:**
- PLY file as attachment

### GET /available_ply_files
Get a list of available PLY files.

**Response:**
- JSON array of filenames

### GET /download_ply/{filename}
Download a specific PLY file.

**Response:**
- PLY file as attachment 