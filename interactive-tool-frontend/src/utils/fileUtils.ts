import { ExportData } from '../types';
import { FILE_TYPES } from './constants';

// Function to validate a PLY file
export const validatePlyFile = (file: File): boolean => {
  return file.name.toLowerCase().endsWith('.ply');
};

// Function to validate an MP4 file
export const validateMp4File = (file: File): boolean => {
  return file.name.toLowerCase().endsWith('.mp4');
};

/**
 * Export data to JSON format
 * @param data Data to export
 * @returns JSON string
 */
export const exportToJson = (data: any): string => {
  return JSON.stringify(data, null, 2);
};

/**
 * Download a file
 * @param content Content of the file
 * @param fileName Name of the file
 */
export const downloadFile = (content: string, fileName: string): void => {
  const blob = new Blob([content], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(url);
};

/**
 * Filter files by extension
 * @param files Array of files
 * @param extensions Array of extensions to filter by (e.g. ['.ply', '.mp4'])
 * @returns Filtered files
 */
export const filterFilesByExtension = (files: File[], extensions: string[]): File[] => {
  return files.filter(file => {
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    return extensions.includes(extension);
  });
};

/**
 * Get the first file with a specific extension
 * @param files Array of files
 * @param extension File extension to look for (e.g. '.ply')
 * @returns The first file with the specified extension, or null if none found
 */
export const getFileByExtension = (files: File[], extension: string): File | null => {
  const filteredFiles = filterFilesByExtension(files, [extension]);
  return filteredFiles.length > 0 ? filteredFiles[0] : null;
};

/**
 * Process a folder import to extract PLY and video files
 * @param files Array of files from a folder import
 * @returns Object containing the PLY and video files
 */
export const processFolderImport = (files: FileList | null): { 
  plyFile: File | null; 
  videoFile: File | null;
  otherFiles: File[];
} => {
  if (!files || files.length === 0) {
    return { plyFile: null, videoFile: null, otherFiles: [] };
  }

  const fileArray = Array.from(files);
  const plyFile = getFileByExtension(fileArray, FILE_TYPES.PLY);
  const videoFiles = filterFilesByExtension(fileArray, [FILE_TYPES.MP4, FILE_TYPES.MOV]);
  const videoFile = videoFiles.length > 0 ? videoFiles[0] : null;
  
  // Get all other files that are not PLY or video
  const otherFiles = fileArray.filter(file => {
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    return extension !== FILE_TYPES.PLY && 
           extension !== FILE_TYPES.MP4 && 
           extension !== FILE_TYPES.MOV;
  });

  return { plyFile, videoFile, otherFiles };
}; 