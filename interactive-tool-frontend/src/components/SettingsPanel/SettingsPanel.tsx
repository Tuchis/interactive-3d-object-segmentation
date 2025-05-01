import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Checkbox, 
  FormControlLabel, 
  Slider, 
  TextField, 
  Paper, 
  Divider,
  Tooltip,
  IconButton,
  Collapse
} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import InfoIcon from '@mui/icons-material/Info';
import { ProcessingSettings } from '../../types';

interface SettingsPanelProps {
  settings: ProcessingSettings;
  onSettingsChange: (settings: ProcessingSettings) => void;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ settings, onSettingsChange }) => {
  const [expanded, setExpanded] = useState(false);

  const handleToggleExpand = () => {
    setExpanded(!expanded);
  };

  const handleNegativeBackgroundChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onSettingsChange({
      ...settings,
      negativeBackground: event.target.checked
    });
  };

  const handleSegmentationThresholdChange = (_event: Event, newValue: number | number[]) => {
    onSettingsChange({
      ...settings,
      segmentationThreshold: newValue as number
    });
  };

  const handleSegmentationThresholdInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(event.target.value);
    if (!isNaN(value) && value >= 0 && value <= 1) {
      onSettingsChange({
        ...settings,
        segmentationThreshold: value
      });
    }
  };

  const handleUseSAM2Change = (event: React.ChangeEvent<HTMLInputElement>) => {
    const useSAM2 = event.target.checked;
    onSettingsChange({
      ...settings,
      useSAM2,
      // If SAM2 is unchecked, also uncheck dependent options
      propagateVideo: useSAM2 ? settings.propagateVideo : false
    });
  };

  const handlePropagateVideoChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const propagateVideo = event.target.checked;
    onSettingsChange({
      ...settings,
      propagateVideo,
      // If propagateVideo is unchecked, also uncheck dependent options
      reversePropagation: propagateVideo ? settings.reversePropagation : false
    });
  };

  const handleReversePropagationChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onSettingsChange({
      ...settings,
      reversePropagation: event.target.checked
    });
  };

  const handleFrameIntervalChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(event.target.value);
    if (!isNaN(value) && value > 0) {
      onSettingsChange({
        ...settings,
        frameInterval: value
      });
    }
  };

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <SettingsIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Processing Settings</Typography>
        </Box>
        <IconButton onClick={handleToggleExpand}>
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>
      
      <Collapse in={expanded}>
        <Divider sx={{ my: 2 }} />
        
        <Box sx={{ mb: 2 }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={settings.negativeBackground}
                onChange={handleNegativeBackgroundChange}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography>Negative Background</Typography>
                <Tooltip title="Treat background as negative examples">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography>Segmentation Threshold</Typography>
            <Tooltip title="Threshold for segmentation confidence (0-1)">
              <IconButton size="small">
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Slider
              value={settings.segmentationThreshold}
              onChange={handleSegmentationThresholdChange}
              min={0}
              max={1}
              step={0.01}
              sx={{ mr: 2, flexGrow: 1 }}
            />
            <TextField
              value={settings.segmentationThreshold}
              onChange={handleSegmentationThresholdInputChange}
              inputProps={{
                step: 0.01,
                min: 0,
                max: 1,
                type: 'number',
              }}
              sx={{ width: '80px' }}
              size="small"
            />
          </Box>
        </Box>
        
        <Box sx={{ mb: 2 }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={settings.useSAM2}
                onChange={handleUseSAM2Change}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography>Use SAM2</Typography>
                <Tooltip title="Use Segment Anything Model 2 for processing">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        </Box>
        
        <Box sx={{ mb: 2, ml: 4 }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={settings.propagateVideo}
                onChange={handlePropagateVideoChange}
                disabled={!settings.useSAM2}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography color={settings.useSAM2 ? 'textPrimary' : 'text.disabled'}>
                  Propagate Video
                </Typography>
                <Tooltip title="Propagate segmentation through video frames">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        </Box>
        
        <Box sx={{ mb: 2, ml: 8 }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={settings.reversePropagation}
                onChange={handleReversePropagationChange}
                disabled={!settings.useSAM2 || !settings.propagateVideo}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography color={settings.useSAM2 && settings.propagateVideo ? 'textPrimary' : 'text.disabled'}>
                  Reverse Propagation
                </Typography>
                <Tooltip title="Propagate in reverse direction through video frames">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        </Box>
        
        <Box sx={{ mb: 2, ml: 4 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography color={settings.useSAM2 && settings.propagateVideo ? 'textPrimary' : 'text.disabled'}>
              Frame Interval
            </Typography>
            <Tooltip title="Interval between frames for propagation">
              <IconButton size="small">
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          <TextField
            value={settings.frameInterval}
            onChange={handleFrameIntervalChange}
            disabled={!settings.useSAM2 || !settings.propagateVideo}
            inputProps={{
              step: 1,
              min: 1,
              type: 'number',
            }}
            size="small"
            sx={{ width: '100px' }}
          />
        </Box>
      </Collapse>
    </Paper>
  );
};

export default SettingsPanel; 