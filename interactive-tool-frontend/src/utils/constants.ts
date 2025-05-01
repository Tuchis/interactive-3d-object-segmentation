/**
 * Constants and configuration values for the frontend application.
 */

// API configuration
export const API_CONFIG = {
  BASE_URL: 'http://0.0.0.0:5100',
  ENDPOINTS: {
    PROCESS_MARKERS: '/process_markers',
    AVAILABLE_PLY_FILES: '/available_ply_files',
    DOWNLOAD_PLY: '/download_ply'
  }
};

// File types
export const FILE_TYPES = {
  PLY: '.ply',
  MP4: '.mp4',
  MOV: '.mov'
};

// Specific files
export const SPECIFIC_FILES = {
  PLY_FILENAME: 'scene0008_00_seg_sam.ply'
};

// Marker types
export const MARKER_TYPES = {
  POSITIVE: '+',
  NEGATIVE: '-'
};

// UI configuration
export const UI_CONFIG = {
  SNACKBAR_DURATION: 3000, // milliseconds
  DEFAULT_MESH_VIEW: true
}; 