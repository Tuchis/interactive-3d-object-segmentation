import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile, toBlobURL } from '@ffmpeg/util';

// Create FFmpeg instance
const ffmpeg = new FFmpeg();
let ffmpegLoaded = false;
let ffmpegLoadPromise: Promise<void> | null = null;

// Load FFmpeg with proper error handling and caching
const loadFFmpeg = async () => {
  if (ffmpegLoaded) return;
  
  // If already loading, return the existing promise
  if (ffmpegLoadPromise) return ffmpegLoadPromise;
  
  ffmpegLoadPromise = (async () => {
    try {
      console.log('Loading FFmpeg...');
      
      // Try loading from local files first (most reliable)
      try {
        console.log('Trying to load FFmpeg from local files...');
        await ffmpeg.load({
          coreURL: '/ffmpeg/ffmpeg-core.js',
          wasmURL: '/ffmpeg/ffmpeg-core.wasm',
        });
        
        ffmpegLoaded = true;
        console.log('FFmpeg loaded successfully from local files');
        return;
      } catch (localError) {
        console.warn('Failed to load FFmpeg from local files, trying CDN...', localError);
      }
      
      // Try primary CDN
      const baseURL = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/umd';
      
      // Load FFmpeg core with proper MIME types
      await ffmpeg.load({
        coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
        wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
      });
      
      ffmpegLoaded = true;
      console.log('FFmpeg loaded successfully from CDN');
    } catch (error) {
      console.error('Failed to load FFmpeg from primary CDN:', error);
      
      // Try alternative CDN if the first one fails
      try {
        console.log('Trying alternative CDN...');
        const altBaseURL = 'https://unpkg.com/@ffmpeg/core@0.12.10/dist/umd';
        
        await ffmpeg.load({
          coreURL: await toBlobURL(`${altBaseURL}/ffmpeg-core.js`, 'text/javascript'),
          wasmURL: await toBlobURL(`${altBaseURL}/ffmpeg-core.wasm`, 'application/wasm'),
        });
        
        ffmpegLoaded = true;
        console.log('FFmpeg loaded successfully from alternative CDN');
      } catch (altError) {
        console.error('Failed to load FFmpeg from alternative CDN:', altError);
        throw new Error(`Failed to load FFmpeg: ${error instanceof Error ? error.message : String(error)}\n\nPlease check your internet connection and browser settings. This feature requires a modern browser with SharedArrayBuffer support.`);
      }
    }
  })();
  
  return ffmpegLoadPromise;
};

/**
 * Convert a video file to MP4 with H.264 codec
 * @param file The input video file
 * @param onProgress Progress callback (0-100)
 * @returns A Promise that resolves to the converted file as a Blob
 */
export const convertToCompatibleMp4 = async (
  file: File,
  onProgress?: (progress: number) => void
): Promise<Blob> => {
  try {
    // Load FFmpeg if not already loaded
    await loadFFmpeg();
    
    // Generate input and output filenames
    const inputFileName = 'input' + file.name.substring(file.name.lastIndexOf('.'));
    const outputFileName = 'output.mp4';
    
    // Write the input file to FFmpeg's virtual file system
    await ffmpeg.writeFile(inputFileName, await fetchFile(file));
    
    // Set up progress tracking
    if (onProgress) {
      ffmpeg.on('progress', (progressEvent: any) => {
        // The progress event structure may vary depending on the FFmpeg.wasm version
        // Make sure we handle both possible formats
        const progressValue = typeof progressEvent.progress === 'number' 
          ? progressEvent.progress 
          : (typeof progressEvent.ratio === 'number' ? progressEvent.ratio : 0);
        
        onProgress(Math.round(progressValue * 100));
      });
    }
    
    // Run FFmpeg command to convert the video
    // -i: input file
    // -c:v libx264: use H.264 codec for video
    // -preset fast: balance between encoding speed and compression
    // -crf 23: constant rate factor (quality, lower is better)
    // -c:a aac: use AAC codec for audio
    // -b:a 128k: audio bitrate
    // -movflags +faststart: optimize for web streaming
    await ffmpeg.exec([
      '-i', inputFileName,
      '-c:v', 'libx264',
      '-preset', 'fast',
      '-crf', '23',
      '-c:a', 'aac',
      '-b:a', '128k',
      '-movflags', '+faststart',
      outputFileName
    ]);
    
    // Read the output file from FFmpeg's virtual file system
    const data = await ffmpeg.readFile(outputFileName);
    
    // Create a Blob from the output data
    const blob = new Blob([data], { type: 'video/mp4' });
    
    // Clean up files from FFmpeg's virtual file system
    await ffmpeg.deleteFile(inputFileName);
    await ffmpeg.deleteFile(outputFileName);
    
    return blob;
  } catch (error) {
    console.error('Error converting video:', error);
    throw new Error(`Failed to convert video: ${error instanceof Error ? error.message : String(error)}`);
  }
};

/**
 * Check if a video file is likely to be compatible with the browser
 * @param file The video file to check
 * @returns True if the file is likely to be compatible
 */
export const isLikelyCompatible = (file: File): boolean => {
  const fileType = file.type.toLowerCase();
  const fileName = file.name.toLowerCase();
  const extension = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();
  
  // Check for common compatible formats
  if (fileType === 'video/mp4' || fileType === 'video/webm') {
    return true;
  }
  
  // Check for common compatible extensions
  if (extension === '.mp4' || extension === '.webm') {
    return true;
  }
  
  return false;
}; 