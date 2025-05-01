// Marker type for both pointcloud and video
export type MarkerType = '+' | '-';

// 3D marker in the pointcloud
export interface PointCloudMarker {
  id: string;
  position: [number, number, number]; // x, y, z coordinates
  type: MarkerType;
}

// 2D marker on video frames
export interface VideoMarker {
  id: string;
  position: [number, number]; // x, y coordinates on the frame
  type: MarkerType;
  frameNumber: number;
}

// Collection of markers per video frame
export interface VideoFrameMarkers {
  [frameNumber: number]: VideoMarker[];
}

// Export data structure
export interface ExportData {
  pointCloudMarkers: PointCloudMarker[];
  videoFrameMarkers: VideoFrameMarkers;
}

// Processing settings for the backend
export interface ProcessingSettings {
  negativeBackground: boolean;
  segmentationThreshold: number;
  useSAM2: boolean;
  propagateVideo: boolean;
  reversePropagation: boolean;
  frameInterval: number;
} 