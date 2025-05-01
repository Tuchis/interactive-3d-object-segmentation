import React, { useState, useRef, useEffect } from 'react'
import { Box, Button, Container, CssBaseline, Grid, Paper, ThemeProvider, createTheme, Stack, Snackbar, Alert, Tooltip } from '@mui/material'
import PointCloudViewer from './components/PointCloudViewer/PointCloudViewer'
import VideoViewer from './components/VideoViewer/VideoViewer'
import SettingsPanel from './components/SettingsPanel/SettingsPanel'
import { MarkerType, PointCloudMarker, VideoMarker, VideoFrameMarkers, ExportData, ProcessingSettings } from './types'
import { exportToJson, downloadFile, processFolderImport } from './utils/fileUtils'
import { sendMarkersToBackend } from './utils/apiService'
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import SendIcon from '@mui/icons-material/Send';
import FolderIcon from '@mui/icons-material/Folder';
import './App.css'

// Create a theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
})

// Default processing settings
const defaultSettings: ProcessingSettings = {
  negativeBackground: false,
  segmentationThreshold: 0.5,
  useSAM2: false,
  propagateVideo: false,
  reversePropagation: false,
  frameInterval: 5
};

function App() {
  // State for marker type selection
  const [currentMarkerType, setCurrentMarkerType] = useState<MarkerType>('+')
  
  // State for pointcloud markers
  const [pointCloudMarkers, setPointCloudMarkers] = useState<PointCloudMarker[]>([])
  
  // State for video markers (organized by frame)
  const [videoMarkers, setVideoMarkers] = useState<VideoFrameMarkers>({})
  
  // State for processing settings
  const [settings, setSettings] = useState<ProcessingSettings>(defaultSettings);
  
  // State for server operations
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [showSnackbar, setShowSnackbar] = useState<boolean>(false);
  const [snackbarMessage, setSnackbarMessage] = useState<string>('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error' | 'info' | 'warning'>('info');
  
  // Create refs for the components
  const pointCloudViewerRef = useRef<any>(null);
  const videoViewerRef = useRef<any>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  
  // Handle adding a pointcloud marker
  const addPointCloudMarker = (marker: PointCloudMarker) => {
    setPointCloudMarkers([...pointCloudMarkers, marker])
  }
  
  // Handle removing a pointcloud marker
  const removePointCloudMarker = (id: string) => {
    setPointCloudMarkers(pointCloudMarkers.filter(marker => marker.id !== id))
  }
  
  // Handle adding a video marker
  const addVideoMarker = (marker: VideoMarker) => {
    setVideoMarkers(prev => {
      const frameMarkers = prev[marker.frameNumber] || []
      return {
        ...prev,
        [marker.frameNumber]: [...frameMarkers, marker]
      }
    })
  }
  
  // Handle removing a video marker
  const removeVideoMarker = (id: string) => {
    setVideoMarkers(prev => {
      // Create a new object to avoid mutating the previous state
      const newVideoMarkers = { ...prev }
      
      // Find the frame that contains the marker with the given id
      for (const frameNumber in newVideoMarkers) {
        const frameMarkers = newVideoMarkers[frameNumber]
        const markerIndex = frameMarkers.findIndex(marker => marker.id === id)
        
        if (markerIndex !== -1) {
          // Remove the marker from the frame
          const newFrameMarkers = [...frameMarkers]
          newFrameMarkers.splice(markerIndex, 1)
          
          // Update the frame markers
          newVideoMarkers[frameNumber] = newFrameMarkers
          
          // If the frame is now empty, remove it
          if (newFrameMarkers.length === 0) {
            delete newVideoMarkers[frameNumber]
          }
          
          break
        }
      }
      
      return newVideoMarkers
    })
  }
  
  // Handle settings change
  const handleSettingsChange = (newSettings: ProcessingSettings) => {
    setSettings(newSettings);
  };
  
  // Handle folder import
  const handleFolderImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) {
      return;
    }
    
    setIsProcessing(true);
    setSnackbarMessage('Processing folder import...');
    setSnackbarSeverity('info');
    setShowSnackbar(true);
    
    try {
      const { plyFile, videoFile, otherFiles } = processFolderImport(e.target.files);
      
      // Load PLY file if available
      if (plyFile && pointCloudViewerRef.current && pointCloudViewerRef.current.loadFile) {
        pointCloudViewerRef.current.loadFile(plyFile);
      }
      
      // Load video file if available
      if (videoFile && videoViewerRef.current && videoViewerRef.current.loadFile) {
        videoViewerRef.current.loadFile(videoFile);
      }
      
      // Show success message
      const message = [];
      if (plyFile) message.push(`PLY: ${plyFile.name}`);
      if (videoFile) message.push(`Video: ${videoFile.name}`);
      if (otherFiles.length > 0) message.push(`${otherFiles.length} other files ignored`);
      
      if (message.length > 0) {
        setSnackbarMessage(`Imported: ${message.join(', ')}`);
        setSnackbarSeverity('success');
      } else {
        setSnackbarMessage('No compatible files found in the folder');
        setSnackbarSeverity('warning');
      }
      
      setShowSnackbar(true);
    } catch (error) {
      console.error('Error importing folder:', error);
      setSnackbarMessage('Error importing folder');
      setSnackbarSeverity('error');
      setShowSnackbar(true);
    } finally {
      setIsProcessing(false);
      
      // Reset the input value so the same folder can be selected again
      if (folderInputRef.current) {
        folderInputRef.current.value = '';
      }
    }
  };
  
  // Handle folder import button click
  const handleFolderImportClick = () => {
    if (folderInputRef.current) {
      folderInputRef.current.click();
    }
  };
  
  // Handle export function
  const handleExport = () => {
    const exportData: ExportData = {
      pointCloudMarkers,
      videoFrameMarkers: videoMarkers
    }
    
    const jsonData = exportToJson(exportData)
    downloadFile(jsonData, 'marker_data.json')
  }
  
  // Handle process on server
  const handleProcessOnServer = async () => {
    if (pointCloudMarkers.length === 0 && Object.keys(videoMarkers).length === 0) {
      setSnackbarMessage('No markers to process');
      setSnackbarSeverity('warning');
      setShowSnackbar(true);
      return;
    }
    
    try {
      setIsProcessing(true);
      setSnackbarMessage('Processing markers on the server...');
      setSnackbarSeverity('info');
      setShowSnackbar(true);
      
      // Get the current PLY and video files from the components
      let plyFile: File | undefined;
      let videoFile: File | undefined;
      
      if (pointCloudViewerRef.current && pointCloudViewerRef.current.getCurrentFile) {
        plyFile = pointCloudViewerRef.current.getCurrentFile();
      }
      
      if (videoViewerRef.current && videoViewerRef.current.getCurrentFile) {
        videoFile = videoViewerRef.current.getCurrentFile();
      }
      
      // Send both 3D and 2D markers to the backend along with settings
      const processedPly = await sendMarkersToBackend(
        pointCloudMarkers,
        videoMarkers,
        settings,
        plyFile,
        videoFile
      );
      
      // Update the PointCloudViewer with the processed PLY file
      if (pointCloudViewerRef.current && pointCloudViewerRef.current.loadProcessedFile) {
        await pointCloudViewerRef.current.loadProcessedFile(processedPly);
      }
      
      setSnackbarMessage('Successfully processed markers on the server');
      setSnackbarSeverity('success');
      setShowSnackbar(true);
    } catch (error) {
      console.error('Error processing markers:', error);
      setSnackbarMessage('Error processing markers on the server');
      setSnackbarSeverity('error');
      setShowSnackbar(true);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Handle server files
  const handleServerFiles = async () => {
    try {
      // Check if the PointCloudViewer component is available
      if (pointCloudViewerRef.current && pointCloudViewerRef.current.fetchAvailablePlyFiles) {
        // Call the fetchAvailablePlyFiles method directly
        await pointCloudViewerRef.current.fetchAvailablePlyFiles();
      } else {
        throw new Error('PointCloudViewer component not available');
      }
    } catch (error) {
      console.error('Error accessing server files:', error);
      setSnackbarMessage('Error accessing server files. Please try again.');
      setSnackbarSeverity('error');
      setShowSnackbar(true);
    }
  };
  
  // Make videoViewerRef available to other components
  useEffect(() => {
    const handleVideoFileLoad = (event: CustomEvent) => {
      if (event.detail && event.detail.file && videoViewerRef.current && videoViewerRef.current.loadFile) {
        videoViewerRef.current.loadFile(event.detail.file);
        setSnackbarMessage(`Video file loaded from server`);
        setSnackbarSeverity('success');
        setShowSnackbar(true);
      }
    };
    
    // Add event listener for custom video file loading event
    window.addEventListener('loadVideoFile', handleVideoFileLoad as EventListener);
    
    return () => {
      // Remove event listener when component unmounts
      window.removeEventListener('loadVideoFile', handleVideoFileLoad as EventListener);
    };
  }, [videoViewerRef]);
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth={false} sx={{ height: '100vh', pt: 2, pb: 2 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          {/* Header with buttons */}
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
            {/* Left side - Import button */}
            <Tooltip title="Import a folder containing PLY and video files">
              <Button
                variant="contained"
                color="primary"
                onClick={handleFolderImportClick}
                startIcon={<FolderIcon />}
                disabled={isProcessing}
              >
                Import Folder
              </Button>
            </Tooltip>
            
            {/* Hidden file input for folder import */}
            <input
              type="file"
              ref={folderInputRef}
              onChange={handleFolderImport}
              style={{ display: 'none' }}
              // @ts-ignore - These attributes are valid in modern browsers but not in the TypeScript definitions
              webkitdirectory=""
              directory=""
              multiple
            />
            
            {/* Right side - Action buttons */}
            <Stack direction="row" spacing={2}>
              <Button
                variant="outlined"
                color="primary"
                onClick={handleServerFiles}
                startIcon={<CloudDownloadIcon />}
                disabled={isProcessing}
              >
                Server Files
              </Button>
              
              <Button
                variant="contained"
                color="primary"
                onClick={handleProcessOnServer}
                startIcon={<SendIcon />}
                disabled={(pointCloudMarkers.length === 0 && Object.keys(videoMarkers).length === 0) || isProcessing}
              >
                Process on Server
              </Button>
              
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleExport}
              >
                Export Data
              </Button>
            </Stack>
          </Box>
          
          {/* Settings Panel */}
          <SettingsPanel 
            settings={settings} 
            onSettingsChange={handleSettingsChange} 
          />
          
          {/* Main content */}
          <Grid container spacing={2} sx={{ flexGrow: 1 }}>
            {/* Left panel - Pointcloud */}
            <Grid item xs={12} md={6} sx={{ height: '100%' }}>
              <Paper sx={{ height: '100%', overflow: 'hidden' }}>
                <PointCloudViewer 
                  markers={pointCloudMarkers}
                  addMarker={addPointCloudMarker}
                  removeMarker={removePointCloudMarker}
                  currentMarkerType={currentMarkerType}
                  setCurrentMarkerType={setCurrentMarkerType}
                  ref={pointCloudViewerRef}
                  hideServerButtons={true}
                />
              </Paper>
            </Grid>
            
            {/* Right panel - Video */}
            <Grid item xs={12} md={6} sx={{ height: '100%' }}>
              <Paper sx={{ height: '100%', overflow: 'hidden' }}>
                <VideoViewer 
                  markers={videoMarkers}
                  addMarker={addVideoMarker}
                  removeMarker={removeVideoMarker}
                  currentMarkerType={currentMarkerType}
                  setCurrentMarkerType={setCurrentMarkerType}
                  ref={videoViewerRef}
                />
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Container>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={showSnackbar}
        autoHideDuration={3000}
        onClose={() => setShowSnackbar(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setShowSnackbar(false)} 
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  )
}

export default App
