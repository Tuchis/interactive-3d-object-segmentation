/**
 * API Service for communicating with the Python backend
 */

import { API_CONFIG } from './constants';
import { ProcessingSettings } from '../types';

/**
 * Common fetch options for API requests
 */
const commonFetchOptions = {
  mode: 'cors' as RequestMode,
  credentials: 'omit' as RequestCredentials,
  headers: {
    'Accept': 'application/json, application/octet-stream'
  }
};

/**
 * Send both 3D point cloud markers and 2D video markers to the backend for processing
 * @param pointCloudMarkers Array of 3D markers with positions and types
 * @param videoMarkers Object containing video frame markers
 * @param settings Processing settings for the backend
 * @param currentPlyFile The current PLY file being viewed (optional)
 * @param currentVideoFile The current video file being viewed (optional)
 * @returns Promise with the new PLY file as a Blob
 */
export const sendMarkersToBackend = async (
  pointCloudMarkers: any[],
  videoMarkers: any = {},
  settings: ProcessingSettings,
  currentPlyFile?: File,
  currentVideoFile?: File
): Promise<Blob> => {
  try {
    // Create form data to send both markers and files
    const formData = new FormData();
    
    // Add markers as JSON
    formData.append('point_cloud_markers', JSON.stringify(pointCloudMarkers));
    formData.append('video_markers', JSON.stringify(videoMarkers));
    
    // Add settings as JSON
    formData.append('settings', JSON.stringify(settings));
    
    // Add the current PLY file if available
    if (currentPlyFile) {
      formData.append('ply_file', currentPlyFile);
    }
    
    // Add the current video file if available
    if (currentVideoFile) {
      formData.append('video_file', currentVideoFile);
    }
    
    // Make the API call
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.PROCESS_MARKERS}`, {
      method: 'POST',
      body: formData,
      ...commonFetchOptions
    });
    
    // Check if the request was successful
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend error: ${response.status} - ${errorText}`);
    }
    
    // Return the response as a blob (PLY file)
    return await response.blob();
  } catch (error) {
    console.error('Error sending markers to backend:', error);
    throw error;
  }
};

/**
 * Get a list of available files from the backend
 * @returns Promise with an array of available files
 */
export const getAvailablePlyFiles = async (): Promise<Array<{
  name: string;
  type: 'ply' | 'video';
  size: number;
}>> => {
  try {
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.AVAILABLE_PLY_FILES}`, {
      method: 'GET',
      ...commonFetchOptions
    });
    
    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching available files:', error);
    throw error;
  }
};

/**
 * Download a specific PLY file from the backend
 * @param filename Name of the PLY file to download or file object with name property
 * @returns Promise with the PLY file as a Blob
 */
export const downloadPlyFile = async (filename: string | { name: string }): Promise<Blob> => {
  try {
    // Extract the filename if an object was passed
    const filenameStr = typeof filename === 'string' ? filename : filename.name;
    
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.DOWNLOAD_PLY}/${filenameStr}`, {
      method: 'GET',
      ...commonFetchOptions
    });
    
    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }
    
    return await response.blob();
  } catch (error) {
    console.error('Error downloading PLY file:', error);
    throw error;
  }
}; 