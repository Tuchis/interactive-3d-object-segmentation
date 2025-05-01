import React, { useState, useRef, useEffect, forwardRef, useImperativeHandle } from 'react';
import ReactPlayer from 'react-player';
import { Box, Button, Slider, Typography, Stack, CircularProgress, Alert, Link, Tooltip, Snackbar } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import UndoIcon from '@mui/icons-material/Undo';
import { MarkerType, VideoMarker, VideoFrameMarkers } from '../../types';
import { convertToCompatibleMp4, isLikelyCompatible } from '../../utils/videoConverter';

// Enhanced video compatibility checker with codec detection
const checkVideoCompatibility = (file: File): { 
  isLikelyCompatible: boolean; 
  format: string;
  suggestedAction: string;
  possibleIssue: string;
} => {
  const fileType = file.type.toLowerCase();
  const fileName = file.name.toLowerCase();
  const extension = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();
  
  // Highly compatible formats
  const highlyCompatible = ['video/mp4', '.mp4', 'video/webm', '.webm'];
  // Moderately compatible formats
  const moderatelyCompatible = ['video/quicktime', '.mov', 'video/ogg', '.ogg', '.m4v'];
  // Problematic formats
  const problematicFormats = ['.avi', '.mkv', '.flv', '.wmv', '.divx', '.vob', '.mpg', '.mpeg'];
  
  let format = extension;
  if (fileType) format = `${fileType} (${extension})`;
  
  // Special case for MP4 files - they might use unsupported codecs
  if (extension === '.mp4' || fileType.includes('mp4')) {
    return { 
      isLikelyCompatible: true, 
      format,
      suggestedAction: 'If this MP4 file doesn\'t play, it may use an unsupported codec (like HEVC/H.265) or container features.',
      possibleIssue: 'MP4 files can contain various codecs - some browsers only support H.264 codec and not newer ones like HEVC/H.265.'
    };
  }
  
  // Check if the format is likely to be compatible
  if (highlyCompatible.some(f => fileType.includes(f) || extension.includes(f))) {
    return { 
      isLikelyCompatible: true, 
      format,
      suggestedAction: '',
      possibleIssue: ''
    };
  } else if (moderatelyCompatible.some(f => fileType.includes(f) || extension.includes(f))) {
    return { 
      isLikelyCompatible: true, 
      format,
      suggestedAction: 'This format may have limited compatibility. If you experience issues, consider converting to MP4 with H.264 codec.',
      possibleIssue: 'Browser support for this format varies and may depend on the specific codec used.'
    };
  } else if (problematicFormats.some(f => extension.includes(f))) {
    return { 
      isLikelyCompatible: false, 
      format,
      suggestedAction: `${extension.toUpperCase()} format has limited browser support. We recommend converting to MP4 with H.264 codec for better compatibility.`,
      possibleIssue: 'This format is not natively supported by most browsers.'
    };
  }
  
  return { 
    isLikelyCompatible: false, 
    format,
    suggestedAction: 'This video format may not be supported by your browser. Try converting it to MP4 format with H.264 codec for better compatibility.',
    possibleIssue: 'Unknown format or codec compatibility issues.'
  };
};

// Add a function to diagnose MP4 playback issues
const diagnoseMP4Issue = (file: File, errorCode: number): string => {
  // Check file size - very large files might cause issues
  const fileSizeMB = file.size / (1024 * 1024);
  
  if (fileSizeMB > 500) {
    return `Large file (${fileSizeMB.toFixed(1)} MB) may exceed browser limits. Try a smaller file or different browser.`;
  }
  
  // Check for common MP4 issues based on error code
  if (errorCode === 4) { // MEDIA_ERR_SRC_NOT_SUPPORTED
    return "This MP4 file likely uses a codec (like HEVC/H.265) or container features not supported by your browser. Convert to MP4 with H.264 codec.";
  } else if (errorCode === 3) { // MEDIA_ERR_DECODE
    return "Browser couldn't decode this MP4. It may use an unsupported codec or be corrupted. Convert to MP4 with H.264 codec.";
  } else if (errorCode === 2) { // MEDIA_ERR_NETWORK
    return "Network error while loading the video. The file may be too large or partially downloaded.";
  }
  
  return "This MP4 file may use an unsupported codec or container features. Convert to MP4 with H.264 codec using HandBrake or FFmpeg.";
};

const VideoViewer = forwardRef(({
  markers,
  addMarker,
  removeMarker,
  currentMarkerType,
  setCurrentMarkerType
}: {
  markers: VideoFrameMarkers;
  addMarker: (marker: VideoMarker) => void;
  removeMarker: (id: string) => void;
  currentMarkerType: MarkerType;
  setCurrentMarkerType: (type: MarkerType) => void;
}, ref) => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [fps, setFps] = useState(30); // Assuming 30fps by default
  const [videoWidth, setVideoWidth] = useState(0);
  const [videoHeight, setVideoHeight] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [useNativeFallback, setUseNativeFallback] = useState(false);
  const [formatInfo, setFormatInfo] = useState<{
    format: string;
    isLikelyCompatible: boolean;
    suggestedAction: string;
  } | null>(null);
  
  // Add state for detailed diagnostics
  const [detailedDiagnostics, setDetailedDiagnostics] = useState<string>('');
  const [errorCode, setErrorCode] = useState<number | null>(null);
  
  // Add state for advanced diagnostics
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [codecInfo, setCodecInfo] = useState<string>('');
  
  // Add state for conversion
  const [isConverting, setIsConverting] = useState<boolean>(false);
  const [conversionProgress, setConversionProgress] = useState<number>(0);
  const [convertedVideoUrl, setConvertedVideoUrl] = useState<string | null>(null);
  
  // Add state for FFmpeg loading
  const [ffmpegError, setFfmpegError] = useState<string | null>(null);
  const [showSnackbar, setShowSnackbar] = useState<boolean>(false);
  const [snackbarMessage, setSnackbarMessage] = useState<string>('');
  
  // Add state to track the last added marker
  const [lastAddedMarker, setLastAddedMarker] = useState<VideoMarker | null>(null);
  
  const playerRef = useRef<ReactPlayer>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Calculate current frame based on time and fps
  useEffect(() => {
    const frameNumber = Math.floor(currentTime * fps);
    setCurrentFrame(frameNumber);
  }, [currentTime, fps]);
  
  // Handle file import
  const handleFileImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      
      // Check video compatibility
      const compatibility = checkVideoCompatibility(selectedFile);
      setFormatInfo(compatibility);
      
      // Check if file is a supported video format
      const supportedFormats = ['.mp4', '.mov', '.webm', '.ogg', '.m4v', '.avi', '.mkv'];
      const fileExtension = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
      
      if (!supportedFormats.some(format => fileExtension.includes(format))) {
        console.warn(`File extension ${fileExtension} may not be fully supported, but will try to play it.`);
      }
      
      setFile(selectedFile);
      setError(null);
      setLoaded(false);
      setUseNativeFallback(false); // Reset fallback on new file
      
      // Create object URL for the file
      try {
        // Release previous URL if exists
        if (videoUrl) {
          URL.revokeObjectURL(videoUrl);
        }
        
        const newVideoUrl = URL.createObjectURL(selectedFile);
        console.log('Created video URL:', newVideoUrl, 'for file type:', selectedFile.type);
        
        // Try to get metadata from the video first using the native video element
        try {
          // Create a temporary video element to get metadata
          const videoEl = document.createElement('video');
          
          // Create a promise to handle the metadata loading
          const metadataPromise = new Promise<void>((resolve, reject) => {
            videoEl.onloadedmetadata = () => {
              setVideoWidth(videoEl.videoWidth);
              setVideoHeight(videoEl.videoHeight);
              setDuration(videoEl.duration);
              console.log('Video metadata loaded:', videoEl.videoWidth, 'x', videoEl.videoHeight);
              resolve();
            };
            
            videoEl.onerror = () => {
              console.warn('Failed to load video metadata with native element');
              reject(new Error('Failed to load video metadata'));
            };
            
            // Add timeupdate event to ensure we get metadata even if onloadedmetadata doesn't fire
            videoEl.ontimeupdate = () => {
              if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
                setVideoWidth(videoEl.videoWidth);
                setVideoHeight(videoEl.videoHeight);
                setDuration(videoEl.duration);
                console.log('Video metadata loaded via timeupdate:', videoEl.videoWidth, 'x', videoEl.videoHeight);
                resolve();
              }
            };
          });
          
          // Set the source and wait for metadata
          videoEl.preload = 'metadata';
          videoEl.src = newVideoUrl;
          videoEl.load(); // Explicitly load to trigger metadata loading
          
          // Wait for metadata with a timeout
          const timeoutPromise = new Promise<void>((resolve, reject) => {
            setTimeout(() => {
              console.warn('Metadata loading timed out, continuing anyway');
              // Don't reject, just continue with default values
              resolve();
            }, 3000);
          });
          
          await Promise.race([metadataPromise, timeoutPromise]);
        } catch (err) {
          console.warn('Could not read video metadata:', err);
          // Continue anyway, ReactPlayer will try to handle it
        }
        
        // Now set the URL for ReactPlayer
        setVideoUrl(newVideoUrl);
      } catch (err) {
        console.error('Error creating object URL:', err);
        setError(`Failed to load video: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  };
  
  // Trigger file input click
  const handleImportClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  // Handle video time update
  const handleProgress = (state: { playedSeconds: number }) => {
    setCurrentTime(state.playedSeconds);
  };
  
  // Handle video duration loaded
  const handleDuration = (duration: number) => {
    setDuration(duration);
  };
  
  // Handle slider change
  const handleSliderChange = (_: Event, value: number | number[]) => {
    const newTime = typeof value === 'number' ? value : value[0];
    setCurrentTime(newTime);
    if (playerRef.current) {
      playerRef.current.seekTo(newTime);
    }
  };
  
  // Handle video ready
  const handleReady = () => {
    setLoaded(true);
    if (playerRef.current && videoContainerRef.current) {
      const wrapper = playerRef.current.getInternalPlayer() as HTMLVideoElement;
      if (wrapper) {
        setVideoWidth(wrapper.videoWidth);
        setVideoHeight(wrapper.videoHeight);
        console.log('Video dimensions:', wrapper.videoWidth, 'x', wrapper.videoHeight);
      }
    }
  };
  
  // Handle error with enhanced diagnostics
  const handleError = (e: any) => {
    console.error('Video error:', e);
    
    // Provide more detailed error information
    let errorMessage = 'Error loading video';
    let code = null;
    
    if (e?.target) {
      // Handle HTML5 video element errors
      const videoElement = e.target as HTMLVideoElement;
      if (videoElement.error) {
        code = videoElement.error.code;
        setErrorCode(code);
        
        switch (videoElement.error.code) {
          case MediaError.MEDIA_ERR_ABORTED:
            errorMessage = 'Video playback aborted';
            break;
          case MediaError.MEDIA_ERR_NETWORK:
            errorMessage = 'Network error occurred while loading the video';
            break;
          case MediaError.MEDIA_ERR_DECODE:
            errorMessage = 'Video decoding error - format may be unsupported';
            break;
          case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
            errorMessage = 'Video format or MIME type is not supported';
            break;
          default:
            errorMessage = `Error loading video: ${videoElement.error.message || 'Unknown error'}`;
        }
        errorMessage += ` (code: ${videoElement.error.code})`;
        
        // Generate detailed diagnostics for MP4 files
        if (file && file.name.toLowerCase().endsWith('.mp4')) {
          const diagnosis = diagnoseMP4Issue(file, videoElement.error.code);
          setDetailedDiagnostics(diagnosis);
        }
      }
    } else if (e?.message) {
      errorMessage = `Error loading video: ${e.message}`;
    }
    
    setError(errorMessage);
    
    // Try using native video element as fallback
    setUseNativeFallback(true);
  };
  
  // Handle play/pause
  const handlePlayPause = () => {
    setPlaying(!playing);
  };
  
  // Handle frame step forward
  const handleStepForward = () => {
    const newTime = Math.min(duration, currentTime + (1 / fps));
    setCurrentTime(newTime);
    if (playerRef.current) {
      playerRef.current.seekTo(newTime);
    }
  };
  
  // Handle frame step backward
  const handleStepBackward = () => {
    const newTime = Math.max(0, currentTime - (1 / fps));
    setCurrentTime(newTime);
    if (playerRef.current) {
      playerRef.current.seekTo(newTime);
    }
  };
  
  // Handle video click for marker placement
  const handleVideoClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoContainerRef.current || !videoUrl || !loaded) return;
    
    const rect = videoContainerRef.current.getBoundingClientRect();
    
    // Find actual video element inside container
    const videoElement = videoContainerRef.current.querySelector('video');
    
    // If we have a video element with computed dimensions
    if (videoElement) {
      const videoRect = videoElement.getBoundingClientRect();
      
      // Only process clicks that are within the actual video display area
      if (
        e.clientX >= videoRect.left &&
        e.clientX <= videoRect.right &&
        e.clientY >= videoRect.top &&
        e.clientY <= videoRect.bottom
      ) {
        // Calculate click position relative to the video display area, not the container
        const x = (e.clientX - videoRect.left) / videoRect.width;
        const y = (e.clientY - videoRect.top) / videoRect.height;
        
        // Create new marker
        const newMarker: VideoMarker = {
          id: `video-${Date.now()}`,
          position: [x, y],
          type: currentMarkerType,
          frameNumber: currentFrame
        };
        
        addMarker(newMarker);
        setLastAddedMarker(newMarker); // Track the last added marker
        console.log(`Added marker at position ${x.toFixed(3)}, ${y.toFixed(3)}`);
      }
    } else {
      // Fallback to container-based calculation if video element not found
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      
      // Create new marker
      const newMarker: VideoMarker = {
        id: `video-${Date.now()}`,
        position: [x, y],
        type: currentMarkerType,
        frameNumber: currentFrame
      };
      
      addMarker(newMarker);
      setLastAddedMarker(newMarker); // Track the last added marker
      console.log(`Added marker at position ${x.toFixed(3)}, ${y.toFixed(3)} (fallback method)`);
    }
  };
  
  // Preload FFmpeg when a potentially incompatible video is loaded
  useEffect(() => {
    if (file && formatInfo && !formatInfo.isLikelyCompatible) {
      // Import dynamically to avoid loading FFmpeg unnecessarily
      import('../../utils/videoConverter').then(module => {
        // Just importing the module will trigger the FFmpeg loading
        setSnackbarMessage('Preparing video conversion tools in the background...');
        setShowSnackbar(true);
      }).catch(err => {
        console.error('Failed to preload FFmpeg:', err);
      });
    }
  }, [file, formatInfo]);
  
  // Add function to handle automatic conversion
  const handleConvertVideo = async () => {
    if (!file) return;
    
    try {
      setIsConverting(true);
      setConversionProgress(0);
      setFfmpegError(null);
      
      // Convert the video using FFmpeg
      const convertedBlob = await convertToCompatibleMp4(file, (progress) => {
        setConversionProgress(progress);
      });
      
      // Create a URL for the converted video
      if (convertedVideoUrl) {
        URL.revokeObjectURL(convertedVideoUrl);
      }
      
      const newUrl = URL.createObjectURL(convertedBlob);
      setConvertedVideoUrl(newUrl);
      
      // Create a new file object for the converted video
      const convertedFile = new File(
        [convertedBlob], 
        file.name.substring(0, file.name.lastIndexOf('.')) + '_converted.mp4',
        { type: 'video/mp4' }
      );
      
      // Update state with the converted video
      setFile(convertedFile);
      setVideoUrl(newUrl);
      setError(null);
      setUseNativeFallback(false);
      setLoaded(false);
      
      console.log('Video successfully converted to compatible MP4');
      setSnackbarMessage('Video successfully converted to compatible MP4');
      setShowSnackbar(true);
    } catch (err) {
      console.error('Error converting video:', err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to convert video: ${errorMessage}`);
      setFfmpegError(errorMessage);
    } finally {
      setIsConverting(false);
    }
  };
  
  // Clean up object URLs when component unmounts
  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
      if (convertedVideoUrl) {
        URL.revokeObjectURL(convertedVideoUrl);
      }
    };
  }, [videoUrl, convertedVideoUrl]);
  
  // Get markers for current frame
  const currentFrameMarkers = markers[currentFrame] || [];
  
  // Add a function to attempt to extract codec information
  const analyzeVideoFile = async () => {
    if (!file) return;
    
    setIsAnalyzing(true);
    
    try {
      // Create a video element to test playback
      const videoEl = document.createElement('video');
      videoEl.style.display = 'none';
      document.body.appendChild(videoEl);
      
      // Set up event listeners
      const canPlayPromise = new Promise<string>((resolve) => {
        // Check what the browser reports about supported types
        const canPlayMP4 = videoEl.canPlayType('video/mp4') ? 'probably/maybe' : 'no';
        const canPlayH264 = videoEl.canPlayType('video/mp4; codecs="avc1.42E01E"') ? 'probably/maybe' : 'no';
        const canPlayHEVC = videoEl.canPlayType('video/mp4; codecs="hev1"') ? 'probably/maybe' : 'no';
        const canPlayAV1 = videoEl.canPlayType('video/mp4; codecs="av01"') ? 'probably/maybe' : 'no';
        
        const info = `Browser codec support:\n- MP4 container: ${canPlayMP4}\n- H.264/AVC: ${canPlayH264}\n- HEVC/H.265: ${canPlayHEVC}\n- AV1: ${canPlayAV1}`;
        
        // For Safari, we can try to get more info from the video element
        videoEl.onloadedmetadata = () => {
          // Try to detect if we're in Safari which might have videoTracks info
          let videoTrackInfo = '';
          try {
            // Use bracket notation to avoid TypeScript errors
            const videoTracks = (videoEl as any)['videoTracks'];
            if (videoTracks && typeof videoTracks.length === 'number') {
              videoTrackInfo = `\n\nDetected video tracks: ${videoTracks.length}`;
            }
          } catch (e) {
            // Ignore errors if videoTracks is not supported
          }
          
          resolve(info + videoTrackInfo);
        };
        
        // If we can't get metadata, just return the basic info
        videoEl.onerror = () => {
          resolve(info + "\n\nCouldn't analyze the specific file - likely using an unsupported codec.");
        };
        
        // Set a timeout in case nothing happens
        setTimeout(() => {
          resolve(info + "\n\nAnalysis timed out. This suggests the file uses an unsupported codec or container format.");
        }, 3000);
      });
      
      // Try to load the file
      if (videoUrl) {
        videoEl.src = videoUrl;
        videoEl.load();
      }
      
      // Wait for the analysis to complete
      const analysisResult = await canPlayPromise;
      setCodecInfo(analysisResult);
      
      // Clean up
      document.body.removeChild(videoEl);
    } catch (err) {
      console.error('Error analyzing video:', err);
      setCodecInfo(`Error analyzing video: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  // Add a function to handle removing the last added marker
  const handleRemoveLastMarker = () => {
    if (lastAddedMarker) {
      removeMarker(lastAddedMarker.id);
      setLastAddedMarker(null);
      
      // Show notification
      setSnackbarMessage('Last marker removed');
      setShowSnackbar(true);
    } else {
      // Find the last marker in the current frame
      const currentFrameMarkers = markers[currentFrame] || [];
      if (currentFrameMarkers.length > 0) {
        const lastMarker = currentFrameMarkers[currentFrameMarkers.length - 1];
        removeMarker(lastMarker.id);
        
        // Show notification
        setSnackbarMessage('Last marker in current frame removed');
        setShowSnackbar(true);
      } else {
        // If no markers in current frame, find the last marker in any frame
        const allFrames = Object.keys(markers).map(Number).sort((a, b) => b - a); // Sort frames in descending order
        
        for (const frame of allFrames) {
          const frameMarkers = markers[frame];
          if (frameMarkers && frameMarkers.length > 0) {
            const lastMarker = frameMarkers[frameMarkers.length - 1];
            removeMarker(lastMarker.id);
            
            // Show notification
            setSnackbarMessage(`Marker removed from frame ${frame}`);
            setShowSnackbar(true);
            break;
          }
        }
      }
    }
  };
  
  // Expose methods to parent component via ref
  useImperativeHandle(ref, () => ({
    getCurrentFile: () => file,
    loadFile: async (newFile: File) => {
      try {
        // Check video compatibility
        const compatibility = checkVideoCompatibility(newFile);
        setFormatInfo(compatibility);
        
        // Check if file is a supported video format
        const supportedFormats = ['.mp4', '.mov', '.webm', '.ogg', '.m4v', '.avi', '.mkv'];
        const fileExtension = newFile.name.substring(newFile.name.lastIndexOf('.')).toLowerCase();
        
        if (!supportedFormats.some(format => fileExtension.includes(format))) {
          console.warn(`File extension ${fileExtension} may not be fully supported, but will try to play it.`);
        }
        
        // Clean up old URL if it exists
        if (videoUrl) {
          URL.revokeObjectURL(videoUrl);
        }
        
        setFile(newFile);
        setError(null);
        setLoaded(false);
        setUseNativeFallback(false); // Reset fallback on new file
        
        // Create object URL for the file
        const url = URL.createObjectURL(newFile);
        setVideoUrl(url);
        
        console.log("Loaded video file:", newFile.name);
        return true;
      } catch (error) {
        console.error('Error loading video file:', error);
        setError('Failed to load video file');
        return false;
      }
    }
  }));
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid #ddd' }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <Button 
            variant="contained" 
            onClick={handleImportClick}
          >
            Import Video
          </Button>
          <input 
            type="file"
            ref={fileInputRef}
            onChange={handleFileImport}
            style={{ display: 'none' }}
            accept="video/*,.mp4,.mov,.webm,.ogg,.m4v"
          />
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
            disabled={Object.keys(markers).length === 0}
            sx={{ ml: 2 }}
          >
            Remove Last Marker
          </Button>
          {file && (
            <Typography variant="body2" color="text.secondary" noWrap sx={{ maxWidth: 200 }}>
              {file.name}
            </Typography>
          )}
        </Stack>
        
        {formatInfo && formatInfo.suggestedAction && (
          <Alert 
            severity={formatInfo.isLikelyCompatible ? "info" : "warning"} 
            sx={{ mt: 2 }}
          >
            <Typography variant="body2">
              {formatInfo.format}: {formatInfo.suggestedAction}
            </Typography>
            {!formatInfo.isLikelyCompatible && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                Try using a tool like <Link href="https://handbrake.fr/" target="_blank" rel="noopener">HandBrake</Link> or 
                <Link href="https://www.ffmpeg.org/" target="_blank" rel="noopener"> FFmpeg</Link> to convert your video.
              </Typography>
            )}
          </Alert>
        )}
      </Box>
      
      <Box 
        ref={containerRef} 
        sx={{ 
          flexGrow: 1, 
          display: 'flex', 
          flexDirection: 'column', 
          position: 'relative',
          p: 2,
          overflow: 'hidden'
        }}
      >
        {/* Video player container with adaptive sizing */}
        <Box 
          sx={{
            width: '100%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            flex: '1 1 auto',
            minHeight: 0,
            overflow: 'hidden'
          }}
        >
          <Box 
            ref={videoContainerRef}
            onClick={handleVideoClick}
            sx={{ 
              position: 'relative',
              height: 'auto',
              width: 'auto',
              maxHeight: '70vh',
              maxWidth: '100%',
              aspectRatio: videoWidth && videoHeight ? `${videoWidth}/${videoHeight}` : '16/9',
              margin: '0 auto',
              cursor: 'crosshair',
              backgroundColor: 'black',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden',
              ...(videoWidth && videoHeight && (videoWidth / videoHeight > 2.5 || videoHeight / videoWidth > 2.5) && {
                maxWidth: '90%',
                maxHeight: '60vh'
              })
            }}
          >
            {videoUrl ? (
              <>
                {!useNativeFallback ? (
                  <ReactPlayer
                    ref={playerRef}
                    url={videoUrl}
                    width="100%"
                    height="100%"
                    playing={playing}
                    onProgress={handleProgress}
                    onDuration={handleDuration}
                    onReady={handleReady}
                    onError={handleError}
                    progressInterval={100}
                    config={{
                      file: {
                        attributes: {
                          style: {
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            maxWidth: '100%'
                          },
                          playsInline: true,
                          controlsList: 'nodownload'
                        },
                        forceVideo: true,
                        // Adjust format handling for better compatibility
                        forceSafariHLS: false,
                        forceHLS: false,
                        forceDisableHls: false,
                        hlsOptions: {
                          enableWorker: true,
                          lowLatencyMode: false,
                          progressive: true,
                          xhrSetup: (xhr: XMLHttpRequest) => {
                            xhr.responseType = 'arraybuffer';
                          }
                        }
                      }
                    }}
                  />
                ) : (
                  <Box 
                    sx={{ 
                      width: '100%', 
                      height: '100%', 
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      maxWidth: '100%'
                    }}
                  >
                    <video 
                      src={videoUrl} 
                      controls={true}
                      playsInline
                      style={{ 
                        width: '100%', 
                        height: '100%', 
                        objectFit: 'contain',
                        maxWidth: '100%'
                      }}
                      autoPlay={false}
                      muted={false}
                      onTimeUpdate={(e) => {
                        const video = e.target as HTMLVideoElement;
                        setCurrentTime(video.currentTime);
                      }}
                      onLoadedMetadata={(e) => {
                        const video = e.target as HTMLVideoElement;
                        setVideoWidth(video.videoWidth);
                        setVideoHeight(video.videoHeight);
                        setDuration(video.duration);
                        setLoaded(true);
                        console.log('Native video loaded:', video.videoWidth, 'x', video.videoHeight);
                      }}
                      onError={(e) => {
                        console.error('Native video error:', e);
                        const videoElement = e.target as HTMLVideoElement;
                        let errorMessage = 'Failed to load video';
                        let code = null;
                        
                        if (videoElement.error) {
                          code = videoElement.error.code;
                          setErrorCode(code);
                          
                          switch (videoElement.error.code) {
                            case MediaError.MEDIA_ERR_ABORTED:
                              errorMessage = 'Video playback aborted';
                              break;
                            case MediaError.MEDIA_ERR_NETWORK:
                              errorMessage = 'Network error occurred while loading the video';
                              break;
                            case MediaError.MEDIA_ERR_DECODE:
                              errorMessage = 'Video decoding error - format may be unsupported';
                              break;
                            case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
                              errorMessage = 'Video format or MIME type is not supported';
                              break;
                            default:
                              errorMessage = `Failed to load video: ${videoElement.error.message || 'Format may be unsupported'}`;
                          }
                          errorMessage += ` (code: ${videoElement.error.code})`;
                          
                          // Generate detailed diagnostics for MP4 files
                          if (file && file.name.toLowerCase().endsWith('.mp4')) {
                            const diagnosis = diagnoseMP4Issue(file, videoElement.error.code);
                            setDetailedDiagnostics(diagnosis);
                          }
                        }
                        
                        setError(errorMessage);
                      }}
                    />
                  </Box>
                )}
                
                {error && (
                  <Box sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    right: 0,
                    padding: 2,
                    backgroundColor: 'rgba(255,0,0,0.7)', 
                    color: 'white',
                    zIndex: 20
                  }}>
                    <Typography>{error}</Typography>
                    
                    {!useNativeFallback && (
                      <Button 
                        variant="contained" 
                        size="small" 
                        onClick={() => setUseNativeFallback(true)}
                        sx={{ mt: 1, mr: 1 }}
                      >
                        Try Native Player
                      </Button>
                    )}
                    
                    {file && !isConverting && (
                      <Button
                        variant="contained"
                        size="small"
                        color="success"
                        onClick={handleConvertVideo}
                        startIcon={<AutoFixHighIcon />}
                        sx={{ mt: 1 }}
                      >
                        Convert Automatically
                      </Button>
                    )}
                    
                    {isConverting && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="body2">
                          Converting video... {conversionProgress}%
                        </Typography>
                        <Box sx={{ width: '100%', mt: 1 }}>
                          <Box
                            sx={{
                              height: 10,
                              borderRadius: 5,
                              bgcolor: 'rgba(255,255,255,0.3)',
                              position: 'relative',
                              overflow: 'hidden'
                            }}
                          >
                            <Box
                              sx={{
                                position: 'absolute',
                                left: 0,
                                top: 0,
                                height: '100%',
                                width: `${conversionProgress}%`,
                                bgcolor: 'success.main',
                                transition: 'width 0.3s ease'
                              }}
                            />
                          </Box>
                        </Box>
                        <Typography variant="caption" sx={{ display: 'block', mt: 1 }}>
                          This may take a few minutes depending on the file size and your device performance.
                        </Typography>
                      </Box>
                    )}
                    
                    {ffmpegError && (
                      <Box sx={{ mt: 2, p: 1, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                          FFmpeg Error Details:
                        </Typography>
                        <Typography variant="caption" sx={{ display: 'block', mt: 0.5, whiteSpace: 'pre-wrap' }}>
                          {ffmpegError}
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          This error may be due to:
                        </Typography>
                        <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                          <li>
                            <Typography variant="caption">
                              Cross-Origin Isolation issues - try using Chrome or Edge
                            </Typography>
                          </li>
                          <li>
                            <Typography variant="caption">
                              Network issues loading FFmpeg core files
                            </Typography>
                          </li>
                          <li>
                            <Typography variant="caption">
                              Insufficient browser permissions or memory
                            </Typography>
                          </li>
                        </ul>
                        <Typography variant="caption" sx={{ display: 'block', mt: 1 }}>
                          You can still try converting the video manually using HandBrake or FFmpeg.
                        </Typography>
                      </Box>
                    )}
                    
                    {useNativeFallback && file && !isConverting && (
                      <>
                        <Typography variant="body2" sx={{ display: 'block', mt: 1 }}>
                          {file.name.toLowerCase().endsWith('.mp4') ? 
                            <>
                              Your browser can't play this specific MP4 file.
                              <Tooltip title="MP4 is just a container format. The actual video codec inside (like H.264, HEVC/H.265, AV1, etc.) determines compatibility.">
                                <InfoIcon sx={{ fontSize: 16, ml: 1, verticalAlign: 'middle' }} />
                              </Tooltip>
                            </> : 
                            `Your browser doesn't support ${file.name.substring(file.name.lastIndexOf('.')).toUpperCase()} format.`
                          }
                        </Typography>
                        
                        {detailedDiagnostics && (
                          <Typography variant="body2" sx={{ display: 'block', mt: 1 }}>
                            <strong>Diagnosis:</strong> {detailedDiagnostics}
                          </Typography>
                        )}
                        
                        {file.name.toLowerCase().endsWith('.mp4') && (
                          <Box sx={{ mt: 1 }}>
                            <Button
                              variant="outlined"
                              size="small"
                              color="inherit"
                              onClick={analyzeVideoFile}
                              disabled={isAnalyzing}
                              startIcon={isAnalyzing ? <CircularProgress size={16} color="inherit" /> : <HelpOutlineIcon />}
                              sx={{ mb: 1 }}
                            >
                              {isAnalyzing ? 'Analyzing...' : 'Analyze MP4 File'}
                            </Button>
                            
                            {codecInfo && (
                              <Box 
                                sx={{ 
                                  mt: 1, 
                                  p: 1, 
                                  backgroundColor: 'rgba(0,0,0,0.3)', 
                                  borderRadius: 1,
                                  whiteSpace: 'pre-line'
                                }}
                              >
                                <Typography variant="caption">
                                  {codecInfo}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        )}
                        
                        <Typography variant="body2" sx={{ display: 'block', mt: 1 }}>
                          Try converting it to MP4 format with H.264 codec using:
                        </Typography>
                        <Box sx={{ mt: 1 }}>
                          <Link 
                            href="https://handbrake.fr/" 
                            target="_blank" 
                            rel="noopener" 
                            sx={{ color: 'white', textDecoration: 'underline', mr: 1 }}
                          >
                            HandBrake
                          </Link>
                          (Recommended settings: MP4 container, H.264 video codec, AAC audio)
                        </Box>
                        <Box sx={{ mt: 1 }}>
                          <Link 
                            href="https://www.ffmpeg.org/" 
                            target="_blank" 
                            rel="noopener" 
                            sx={{ color: 'white', textDecoration: 'underline', mr: 1 }}
                          >
                            FFmpeg
                          </Link>
                          (Command: ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4)
                        </Box>
                      </>
                    )}
                  </Box>
                )}

                {!loaded && !error && (
                  <Box sx={{ 
                    position: 'absolute',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    backgroundColor: 'rgba(0,0,0,0.5)',
                    zIndex: 10,
                    width: '100%',
                    height: '100%'
                  }}>
                    <CircularProgress color="inherit" sx={{ mb: 2 }} />
                    <Typography>Loading video...</Typography>
                  </Box>
                )}
                
                {/* Render markers for current frame */}
                {currentFrameMarkers.map((marker) => (
                  <Box
                    key={marker.id}
                    sx={{
                      position: 'absolute',
                      left: `${marker.position[0] * 100}%`,
                      top: `${marker.position[1] * 100}%`,
                      transform: 'translate(-50%, -50%)',
                      color: marker.type === '+' ? 'green' : 'red',
                      fontWeight: 'bold',
                      fontSize: '24px',
                      pointerEvents: 'none',
                      userSelect: 'none',
                      zIndex: 10
                    }}
                  >
                    {marker.type}
                  </Box>
                ))}
              </>
            ) : (
              <Box sx={{ 
                height: '100%', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                color: 'white'
              }}>
                <Typography>No video loaded. Please import a video file.</Typography>
              </Box>
            )}
          </Box>
        </Box>
        
        {/* Video controls */}
        <Box sx={{ mt: 2, width: '100%' }}>
          <Stack spacing={2}>
            <Stack 
              direction={{ xs: 'column', sm: 'row' }} 
              spacing={2} 
              alignItems="center" 
              justifyContent="center"
            >
              <Button 
                variant="contained" 
                onClick={handlePlayPause}
                disabled={!videoUrl}
                sx={{ minWidth: '85px' }}
              >
                {playing ? 'Pause' : 'Play'}
              </Button>
              <Button 
                variant="outlined" 
                onClick={handleStepBackward}
                disabled={!videoUrl}
              >
                Prev Frame
              </Button>
              <Button 
                variant="outlined" 
                onClick={handleStepForward}
                disabled={!videoUrl}
              >
                Next Frame
              </Button>
              <Typography>
                Frame: {currentFrame} / {Math.floor(duration * fps)}
              </Typography>
            </Stack>
            
            <Box sx={{ px: 2, width: '100%', maxWidth: '800px', margin: '0 auto' }}>
              <Slider
                value={currentTime}
                onChange={handleSliderChange}
                min={0}
                max={duration}
                step={1 / fps}
                disabled={!videoUrl}
              />
            </Box>
            
            <Typography align="center">
              Time: {currentTime.toFixed(2)} / {duration.toFixed(2)} seconds
            </Typography>
          </Stack>
        </Box>
      </Box>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={showSnackbar}
        autoHideDuration={6000}
        onClose={() => setShowSnackbar(false)}
        message={snackbarMessage}
      />
    </Box>
  );
});

export default VideoViewer; 