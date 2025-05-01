import { useEffect, useState, useRef, useMemo, forwardRef, useImperativeHandle } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { MarkerType, PointCloudMarker } from '../../types';
import { Box, Button, Typography, CircularProgress, Stack, Switch, FormControlLabel, Snackbar, Paper, Menu, MenuItem, Dialog, DialogTitle, DialogContent, DialogActions, List, ListItem, ListItemText } from '@mui/material';
import UndoIcon from '@mui/icons-material/Undo';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import SendIcon from '@mui/icons-material/Send';
import RefreshIcon from '@mui/icons-material/Refresh';
import FolderSpecialIcon from '@mui/icons-material/FolderSpecial';
import { sendMarkersToBackend, getAvailablePlyFiles, downloadPlyFile } from '../../utils/apiService';
import { FILE_TYPES, MARKER_TYPES, UI_CONFIG, SPECIFIC_FILES } from '../../utils/constants';

// Point Cloud component that renders the .ply file
const PointCloud = ({ 
  url, 
  onPointClick, 
  markers, 
  currentMarkerType,
  showMesh,
  onModelInfo
}: { 
  url: string | null, 
  onPointClick: (position: [number, number, number]) => void,
  markers: PointCloudMarker[],
  currentMarkerType: MarkerType,
  showMesh: boolean,
  onModelInfo: (hasColors: boolean, clickFeedback: string | null) => void
}) => {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [meshGeometry, setMeshGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clickFeedback, setClickFeedback] = useState<string | null>(null);
  const [hasColors, setHasColors] = useState(false);
  const pointsRef = useRef<THREE.Points>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera, raycaster, gl, scene } = useThree();
  
  // Configure raycaster for better point detection
  useEffect(() => {
    if (raycaster) {
      // Increase the precision for point detection
      raycaster.params.Points.threshold = 0.02;
    }
  }, [raycaster]);
  
  // Process and set up materials for the mesh
  const meshMaterial = useMemo(() => {
    if (!meshGeometry) return null;
    
    // Check if the geometry has color attributes
    const hasVertexColors = meshGeometry.attributes.color !== undefined;
    setHasColors(hasVertexColors);
    
    if (hasVertexColors) {
      return new THREE.MeshPhongMaterial({
        vertexColors: true,
        flatShading: true,
        side: THREE.DoubleSide,
        shininess: 150
      });
    } else {
      // Default material if no vertex colors are available
      return new THREE.MeshPhongMaterial({
        color: 0xAAAAAA,
        flatShading: true,
        side: THREE.DoubleSide,
        shininess: 150
      });
    }
  }, [meshGeometry]);
  
  // Create a point material with vertex colors if available
  const pointMaterial = useMemo(() => {
    if (!geometry) return null;
    
    const hasVertexColors = geometry.attributes.color !== undefined;
    setHasColors(hasVertexColors);
    
    if (hasVertexColors) {
      return new THREE.PointsMaterial({
        vertexColors: true,
        size: 0.02,
        sizeAttenuation: true
      });
    } else {
      return new THREE.PointsMaterial({
        color: 0xFFFFFF,
        size: 0.02,
        sizeAttenuation: true
      });
    }
  }, [geometry]);
  
  useEffect(() => {
    if (!url) return;
    
    setLoading(true);
    setError(null);
    
    // Clear previous geometries when loading a new file
    setGeometry(null);
    setMeshGeometry(null);
    
    console.log("Loading PLY file from URL:", url);
    
    const loader = new PLYLoader();
    
    loader.load(
      url,
      (loadedGeometry) => {
        // Make a copy for mesh version
        const meshGeo = loadedGeometry.clone();
        
        // Compute normals for the mesh
        loadedGeometry.computeVertexNormals();
        meshGeo.computeVertexNormals();
        
        // Set both geometries
        setGeometry(loadedGeometry);
        setMeshGeometry(meshGeo);
        
        setLoading(false);
        console.log("PLY file loaded successfully");
      },
      (progress) => {
        // Optional: handle progress
        const percentComplete = Math.round((progress.loaded / progress.total) * 100);
        console.log(`Loading PLY: ${percentComplete}% complete`);
      },
      (error) => {
        console.error('Error loading PLY file:', error);
        setError('Failed to load PLY file');
        setLoading(false);
      }
    );
    
    // Cleanup function
    return () => {
      // Cancel any pending loads if component unmounts or URL changes
      loader.manager.onLoad = () => {};
      loader.manager.onProgress = () => {};
      loader.manager.onError = () => {};
    };
  }, [url]); // Only re-run when URL changes
  
  // Pass model info up to parent component
  useEffect(() => {
    onModelInfo(hasColors, clickFeedback);
  }, [hasColors, clickFeedback, onModelInfo]);
  
  // Handle click for both mesh and points
  const handleClick = (event: React.MouseEvent) => {
    // Prevent the event from propagating up to parent elements
    event.stopPropagation();
    
    if ((!geometry && !meshGeometry) || (!pointsRef.current && !meshRef.current)) {
      setClickFeedback("No 3D geometry available");
      return;
    }
    
    // Convert mouse coordinates to normalized device coordinates (-1 to +1)
    const canvas = gl.domElement;
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Update the picking ray with the camera and mouse position
    const mouse = new THREE.Vector2(x, y);
    raycaster.setFromCamera(mouse, camera);
    
    // Calculate objects intersecting the picking ray
    const intersectables = [];
    if (showMesh && meshRef.current) {
      intersectables.push(meshRef.current);
    } else if (pointsRef.current) {
      intersectables.push(pointsRef.current);
    }
    
    const intersects = raycaster.intersectObjects(intersectables, false);
    
    if (intersects.length > 0) {
      const intersection = intersects[0];
      const feedback = `Clicked at point (${intersection.point.x.toFixed(2)}, ${intersection.point.y.toFixed(2)}, ${intersection.point.z.toFixed(2)})`;
      setClickFeedback(feedback);
      
      // Call onPointClick with the 3D position
      const position: [number, number, number] = [
        intersection.point.x,
        intersection.point.y,
        intersection.point.z
      ];
      onPointClick(position);
    } else {
      setClickFeedback("No point detected at this position");
      console.log("Click missed all objects - try clicking directly on a visible part");
    }
  };
  
  if (loading) {
    return (
      <Html center>
        <CircularProgress />
      </Html>
    );
  }
  
  if (error) {
    return (
      <Html center>
        <Typography color="error">{error}</Typography>
      </Html>
    );
  }
  
  if (!geometry && !meshGeometry) {
    return (
      <Html center>
        <Typography>No 3D model loaded. Please import a .ply file.</Typography>
      </Html>
    );
  }
  
  return (
    <>
      {/* Render point cloud when showMesh is false */}
      {!showMesh && geometry && (
        <points 
          ref={pointsRef} 
          onClick={handleClick}
        >
          <bufferGeometry attach="geometry" {...geometry} />
          {pointMaterial && <primitive attach="material" object={pointMaterial} />}
        </points>
      )}
      
      {/* Render mesh when showMesh is true */}
      {showMesh && meshGeometry && (
        <mesh
          ref={meshRef}
          onClick={handleClick}
        >
          <bufferGeometry attach="geometry" {...meshGeometry} />
          {meshMaterial && <primitive attach="material" object={meshMaterial} />}
        </mesh>
      )}
      
      {/* Render markers */}
      {markers.map((marker) => (
        <mesh key={marker.id} position={marker.position as any}>
          <sphereGeometry args={[0.03, 16, 16]} />
          <meshBasicMaterial color={marker.type === '+' ? 'green' : 'red'} />
        </mesh>
      ))}
    </>
  );
};

// Canvas wrapper to handle clicks
const CanvasWrapper = ({ 
  url, 
  onPointClick, 
  markers, 
  currentMarkerType,
  showMesh,
  onModelInfo
}: { 
  url: string | null, 
  onPointClick: (position: [number, number, number]) => void,
  markers: PointCloudMarker[],
  currentMarkerType: MarkerType,
  showMesh: boolean,
  onModelInfo: (hasColors: boolean, clickFeedback: string | null) => void
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  return (
    <Canvas 
      ref={canvasRef}
      camera={{ position: [0, 0, 5], fov: 60 }}
      style={{ outline: 'none' }}
    >
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <directionalLight position={[0, 0, 5]} intensity={0.5} />
      <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      <PointCloud 
        url={url} 
        onPointClick={onPointClick} 
        markers={markers}
        currentMarkerType={currentMarkerType}
        showMesh={showMesh}
        onModelInfo={onModelInfo}
      />
    </Canvas>
  );
};

// Main PointCloudViewer component
const PointCloudViewer = forwardRef(({
  markers,
  addMarker,
  removeMarker,
  currentMarkerType,
  setCurrentMarkerType,
  hideServerButtons = false
}: {
  markers: PointCloudMarker[];
  addMarker: (marker: PointCloudMarker) => void;
  removeMarker: (id: string) => void;
  currentMarkerType: MarkerType;
  setCurrentMarkerType: (type: MarkerType) => void;
  hideServerButtons?: boolean;
}, ref) => {
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [showMesh, setShowMesh] = useState(UI_CONFIG.DEFAULT_MESH_VIEW); // Default to mesh view
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Add state for the last added marker and notifications
  const [lastAddedMarker, setLastAddedMarker] = useState<PointCloudMarker | null>(null);
  const [showSnackbar, setShowSnackbar] = useState<boolean>(false);
  const [snackbarMessage, setSnackbarMessage] = useState<string>('');
  
  // Add state for model info that was previously floating
  const [hasColors, setHasColors] = useState<boolean>(false);
  const [clickFeedback, setClickFeedback] = useState<string | null>(null);
  
  // Add state for backend integration
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [availablePlyFiles, setAvailablePlyFiles] = useState<Array<{
    name: string;
    type: 'ply' | 'video';
    size: number;
  }>>([]);
  const [showFilesDialog, setShowFilesDialog] = useState<boolean>(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  
  // Expose methods to parent component via ref
  useImperativeHandle(ref, () => ({
    handleSendToBackend,
    handleMenuOpen,
    handleDownloadPly,
    fetchAvailablePlyFiles,
    getCurrentFile: () => file,
    loadProcessedFile: async (processedPly: Blob) => {
      try {
        // Clean up old URL if it exists
        if (fileUrl) {
          URL.revokeObjectURL(fileUrl);
        }
        
        // Create a new File object from the blob
        const filename = file ? `processed_${file.name}` : 'processed.ply';
        const processedFile = new File([processedPly], filename, { type: 'application/octet-stream' });
        
        // Update state with the new file
        setFile(processedFile);
        
        // Create a new object URL for the processed file
        const newUrl = URL.createObjectURL(processedPly);
        setFileUrl(newUrl);
        
        console.log('Processed PLY file loaded:', newUrl);
        return true;
      } catch (error) {
        console.error('Error loading processed file:', error);
        return false;
      }
    }
  }));
  
  // Handle model info updates from the 3D component
  const handleModelInfo = (hasColors: boolean, clickFeedback: string | null) => {
    setHasColors(hasColors);
    setClickFeedback(clickFeedback);
  };
  
  // Handle file import
  const handleFileImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      
      // Check if file is a PLY file
      if (!selectedFile.name.toLowerCase().endsWith(FILE_TYPES.PLY)) {
        alert(`Please select a valid ${FILE_TYPES.PLY} file`);
        return;
      }
      
      setFile(selectedFile);
      
      // Create object URL for the file
      const url = URL.createObjectURL(selectedFile);
      setFileUrl(url);
      console.log("Created URL for PLY file:", url);
    }
  };
  
  // Handle point click for marker placement
  const handlePointClick = (position: [number, number, number]) => {
    console.log("Adding marker at position:", position);
    const newMarker: PointCloudMarker = {
      id: `pc-${Date.now()}`,
      position,
      type: currentMarkerType
    };
    
    addMarker(newMarker);
    setLastAddedMarker(newMarker); // Track the last added marker
  };
  
  // Add function to handle removing the last added marker
  const handleRemoveLastMarker = () => {
    if (markers.length === 0) return;
    
    let markerToRemove: PointCloudMarker | null = null;
    
    // First, try to remove the last added marker if we have it
    if (lastAddedMarker && markers.some(m => m.id === lastAddedMarker.id)) {
      markerToRemove = lastAddedMarker;
    } else {
      // Otherwise, remove the last marker in the array
      markerToRemove = markers[markers.length - 1];
    }
    
    if (markerToRemove) {
      removeMarker(markerToRemove.id);
      setSnackbarMessage(`Removed marker ${markerToRemove.type} at (${markerToRemove.position.map(p => p.toFixed(2)).join(', ')})`);
      setShowSnackbar(true);
      
      // Update last added marker if needed
      if (lastAddedMarker && lastAddedMarker.id === markerToRemove.id) {
        setLastAddedMarker(null);
      }
    }
  };
  
  // Toggle between mesh and point cloud view
  const handleViewToggle = () => {
    setShowMesh(!showMesh);
  };
  
  // Handle import button click
  const handleImportClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  // Handle menu open
  const handleMenuOpen = (event?: React.MouseEvent<HTMLElement>) => {
    if (event) {
      setAnchorEl(event.currentTarget);
    } else {
      // If no event is provided, use a dummy anchor element
      // This will position the menu in the center of the screen
      setAnchorEl(document.body);
    }
  };
  
  // Handle menu close
  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  // Fetch available PLY files from the backend
  const fetchAvailablePlyFiles = async () => {
    try {
      const files = await getAvailablePlyFiles();
      setAvailablePlyFiles(files);
      setShowFilesDialog(true);
      handleMenuClose();
    } catch (error) {
      console.error('Error fetching files:', error);
      setSnackbarMessage('Error fetching files from server');
      setShowSnackbar(true);
    }
  };
  
  // Handle downloading a PLY file from the backend
  const handleDownloadPly = async (file: { name: string; type: 'ply' | 'video'; size: number; }) => {
    try {
      setIsProcessing(true);
      setSnackbarMessage(`Downloading ${file.name}...`);
      setShowSnackbar(true);
      
      const blob = await downloadPlyFile(file.name);
      
      // Clean up old URL if it exists
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
      }
      
      // Create a File object from the blob
      const newFile = new File([blob], file.name, { 
        type: file.type === 'ply' ? 'application/octet-stream' : 'video/mp4' 
      });
      
      // If it's a PLY file, load it in this component
      if (file.type === 'ply') {
        setFile(newFile);
        
        // Create object URL for the file
        const url = URL.createObjectURL(blob);
        setFileUrl(url);
        
        setShowFilesDialog(false);
        setSnackbarMessage(`Downloaded ${file.name} from server`);
        setShowSnackbar(true);
        
        console.log('Downloaded PLY file loaded:', url);
      } 
      // If it's a video file, try to send it to the parent component
      else if (file.type === 'video') {
        // Try to send the video file to the VideoViewer component via a custom event
        try {
          // Dispatch a custom event that the App component can listen for
          const event = new CustomEvent('loadVideoFile', { 
            detail: { file: newFile }
          });
          window.dispatchEvent(event);
          
          setSnackbarMessage(`Video file ${file.name} sent to Video Viewer`);
        } catch (e) {
          console.error('Error sending video to VideoViewer:', e);
          
          // If the custom event fails, just download the file
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = file.name;
          a.click();
          URL.revokeObjectURL(url);
          
          setSnackbarMessage(`Downloaded ${file.name} (video file)`);
        }
        
        setShowFilesDialog(false);
        setShowSnackbar(true);
      }
    } catch (error) {
      console.error('Error downloading file:', error);
      setSnackbarMessage('Error downloading file from server');
      setShowSnackbar(true);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Send markers to the backend for processing
  const handleSendToBackend = async () => {
    if (markers.length === 0) {
      setSnackbarMessage('No markers to send');
      setShowSnackbar(true);
      return;
    }
    
    try {
      setIsProcessing(true);
      setSnackbarMessage('Processing markers on the server...');
      setShowSnackbar(true);
      
      // Create default settings
      const defaultSettings = {
        negativeBackground: false,
        segmentationThreshold: 0.5,
        useSAM2: false,
        propagateVideo: false,
        reversePropagation: false,
        frameInterval: 5
      };
      
      // Send markers to the backend
      const processedPly = await sendMarkersToBackend(
        markers, 
        {}, // Empty video markers
        defaultSettings,
        file || undefined
      );
      
      // Clean up old URL if it exists
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
      }
      
      // Create a new File object from the blob
      const filename = file ? `processed_${file.name}` : 'processed.ply';
      const processedFile = new File([processedPly], filename, { type: 'application/octet-stream' });
      
      // Update state with the new file
      setFile(processedFile);
      
      // Create a new object URL for the processed file
      const newUrl = URL.createObjectURL(processedPly);
      setFileUrl(newUrl);
      
      // Show success message
      setSnackbarMessage('Successfully processed markers on the server');
      setShowSnackbar(true);
      
      console.log('Processed PLY file loaded:', newUrl);
    } catch (error) {
      console.error('Error processing markers:', error);
      setSnackbarMessage('Error processing markers on the server');
      setShowSnackbar(true);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Add function to load the specific PLY file
  const handleLoadSpecificPly = async () => {
    try {
      setIsProcessing(true);
      setSnackbarMessage('Loading specific PLY file...');
      setShowSnackbar(true);
      
      // Get the list of available files
      const files = await getAvailablePlyFiles();
      
      // Find the specific file
      const specificFileName = SPECIFIC_FILES.PLY_FILENAME;
      const specificFile = files.find(file => file.name === specificFileName && file.type === 'ply');
      
      if (specificFile) {
        // Download the specific file
        await handleDownloadPly(specificFile);
      } else {
        // If the specific file is not available, send markers to get it
        await handleSendToBackend();
      }
    } catch (error) {
      console.error('Error loading specific PLY file:', error);
      setSnackbarMessage('Error loading specific PLY file');
      setShowSnackbar(true);
    } finally {
      setIsProcessing(false);
    }
  };
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid #ddd' }}>
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
          <Button 
            variant="contained" 
            onClick={handleImportClick}
            startIcon={<CloudUploadIcon />}
          >
            Import PLY
          </Button>
          <input 
            type="file"
            ref={fileInputRef}
            onChange={handleFileImport}
            style={{ display: 'none' }}
            accept=".ply"
          />
          
          {!hideServerButtons && (
            <>
              <Button
                variant="outlined"
                color="primary"
                onClick={handleMenuOpen}
                startIcon={<CloudDownloadIcon />}
              >
                Server Files
              </Button>
              
              <Button
                variant="contained"
                color="secondary"
                onClick={handleLoadSpecificPly}
                startIcon={<FolderSpecialIcon />}
                sx={{ ml: 1 }}
              >
                Load Specific PLY
              </Button>
            </>
          )}
          
          <Button 
            variant="outlined" 
            color="success" 
            onClick={() => setCurrentMarkerType('+')}
            sx={{ 
              bgcolor: currentMarkerType === '+' ? 'rgba(0,255,0,0.1)' : 'transparent',
              fontWeight: currentMarkerType === '+' ? 'bold' : 'normal'
            }}
          >
            + Marker
          </Button>
          <Button 
            variant="outlined" 
            color="error" 
            onClick={() => setCurrentMarkerType('-')}
            sx={{ 
              bgcolor: currentMarkerType === '-' ? 'rgba(255,0,0,0.1)' : 'transparent',
              fontWeight: currentMarkerType === '-' ? 'bold' : 'normal'
            }}
          >
            - Marker
          </Button>
          
          <Button
            variant="outlined"
            color="primary"
            onClick={handleRemoveLastMarker}
            startIcon={<UndoIcon />}
            disabled={markers.length === 0}
          >
            Remove Last Marker
          </Button>
          
          {!hideServerButtons && (
            <Button
              variant="contained"
              color="primary"
              onClick={handleSendToBackend}
              startIcon={<SendIcon />}
              disabled={markers.length === 0 || isProcessing}
              sx={{ ml: 2 }}
            >
              Process on Server
            </Button>
          )}
          
          <FormControlLabel
            control={
              <Switch
                checked={showMesh}
                onChange={handleViewToggle}
                color="primary"
              />
            }
            label={showMesh ? "Mesh View" : "Point View"}
          />
          
          {file && (
            <Typography variant="body2" color="text.secondary" noWrap>
              Loaded: {file.name}
            </Typography>
          )}
          
          {isProcessing && <CircularProgress size={24} />}
        </Stack>
      </Box>
      
      <Box sx={{ flexGrow: 1, position: 'relative' }}>
        {/* Fixed position info panel */}
        {fileUrl && (
          <Paper 
            elevation={3} 
            sx={{ 
              position: 'absolute', 
              top: 10, 
              right: 10, 
              zIndex: 10, 
              p: 1.5,
              backgroundColor: 'rgba(0,0,0,0.7)',
              color: 'white',
              borderRadius: 1,
              maxWidth: 300
            }}
          >
            <Typography variant="body2" sx={{ 
              mb: 1, 
              fontWeight: 'bold',
              color: currentMarkerType === '+' ? 'green' : 'red'
            }}>
              Current marker: {currentMarkerType}
            </Typography>
            
            <Typography variant="body2" sx={{ mb: clickFeedback ? 1 : 0 }}>
              View: {showMesh ? 'Mesh' : 'Points'} | Colors: {hasColors ? 'Yes' : 'No'}
            </Typography>
            
            {clickFeedback && (
              <Typography variant="caption" sx={{ display: 'block', wordBreak: 'break-word' }}>
                {clickFeedback}
              </Typography>
            )}
          </Paper>
        )}
        
        <CanvasWrapper 
          url={fileUrl} 
          onPointClick={handlePointClick} 
          markers={markers}
          currentMarkerType={currentMarkerType}
          showMesh={showMesh}
          onModelInfo={handleModelInfo}
        />
      </Box>
      
      {/* Server files menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={fetchAvailablePlyFiles}>
          <ListItemText primary="View Available Files" />
        </MenuItem>
      </Menu>
      
      {/* Available files dialog */}
      <Dialog
        open={showFilesDialog}
        onClose={() => setShowFilesDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Available Files</DialogTitle>
        <DialogContent>
          {availablePlyFiles.length === 0 ? (
            <Typography>No files available on the server</Typography>
          ) : (
            <List>
              {availablePlyFiles.map((file) => (
                <ListItem 
                  key={file.name} 
                  onClick={() => handleDownloadPly(file)}
                  sx={{ cursor: 'pointer' }}
                >
                  <ListItemText 
                    primary={file.name} 
                    secondary={`${file.type.toUpperCase()} - ${(file.size / 1024).toFixed(2)} KB`} 
                  />
                </ListItem>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => fetchAvailablePlyFiles()} 
            startIcon={<RefreshIcon />}
            disabled={isProcessing}
          >
            Refresh
          </Button>
          <Button onClick={() => setShowFilesDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={showSnackbar}
        autoHideDuration={UI_CONFIG.SNACKBAR_DURATION}
        onClose={() => setShowSnackbar(false)}
        message={snackbarMessage}
      />
    </Box>
  );
});

export default PointCloudViewer; 