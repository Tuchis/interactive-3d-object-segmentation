# Interactive 3D Point Cloud Viewer with Python Backend

This application allows users to view and annotate 3D point cloud models (PLY files) with markers, and process these markers using a Python backend.

## Features

- Import and view PLY files in both mesh and point cloud modes
- Add positive and negative markers to the 3D model
- Remove markers with a "Remove Last Marker" button
- Process markers on a Python backend server
- Browse and download PLY files from the server

## Project Structure

```
├── src/                  # Frontend React application
│   ├── components/       # React components
│   ├── utils/            # Utility functions
│   │   ├── apiService.ts # API service for backend communication
│   │   └── constants.ts  # Frontend constants and configuration
│   └── types/            # TypeScript type definitions
├── backend/              # Python FastAPI backend
│   ├── app.py            # Main FastAPI application
│   ├── constants.py      # Backend constants and configuration
│   ├── requirements.txt  # Python dependencies
│   └── ply_files/        # Directory for storing PLY files
└── public/               # Static assets
```

## Setup and Running

### Frontend (React)

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:5173.

### Backend (Python FastAPI)

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the server:
```bash
python app.py
```

The backend will be available at http://localhost:5000.

## Usage

1. Start both the frontend and backend servers.
2. In the frontend, you can:
   - Import a PLY file from your computer
   - Browse and download PLY files from the server
   - Add markers by clicking on the 3D model
   - Toggle between mesh and point cloud views
   - Send markers to the backend for processing
   - Download the processed PLY file

## API Documentation

FastAPI automatically generates interactive API documentation. After starting the backend server, you can access:

- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

## API Endpoints

The backend provides the following API endpoints:

- `POST /process_markers`: Process markers and return a modified PLY file
- `GET /available_ply_files`: Get a list of available PLY files
- `GET /download_ply/{filename}`: Download a specific PLY file

## Development

### Configuration

Both the frontend and backend use constants files for configuration:

- **Frontend**: `src/utils/constants.ts` contains API endpoints, UI settings, and other constants.
- **Backend**: `backend/constants.py` contains server settings, CORS configuration, and API paths.

To change the backend port or other settings, modify the appropriate constants file.

### Adding PLY Files to the Server

To add PLY files to the server, place them in the `backend/ply_files` directory.

### Customizing the Backend Processing

The backend processing logic is in the `process_markers` function in `backend/app.py`. You can customize this function to implement your specific processing requirements.
