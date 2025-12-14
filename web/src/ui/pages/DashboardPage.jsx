import React, { useEffect, useState } from 'react'
import { Grid, Paper, Typography, Box, TextField, Button, Chip, Dialog, DialogTitle, DialogContent, DialogActions, CircularProgress, Alert, Table, TableBody, TableCell, TableHead, TableRow, LinearProgress } from '@mui/material'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { useAuth } from '../state/auth.jsx'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8e44ad', '#e74c3c']

export default function DashboardPage() {
  const { api } = useAuth()
  const [byTime, setByTime] = useState([])
  const [byLabel, setByLabel] = useState([])
  const [search, setSearch] = useState({ label: '', minConfidence: 0.7 })
  const [recent, setRecent] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [captureDialogOpen, setCaptureDialogOpen] = useState(false)
  const [captureDuration, setCaptureDuration] = useState(10)
  const [capturing, setCapturing] = useState(false)
  const [captureResults, setCaptureResults] = useState(null)
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadFile, setUploadFile] = useState(null)
  const [uploadResults, setUploadResults] = useState(null)
  
  async function loadData() {
    try {
      setLoading(true)
      setError(null)
      const [t, l, a] = await Promise.all([
        api.get('/api/alerts/stats/by-time?interval=hour'),
        api.get('/api/alerts/stats/by-label'),
        api.get(`/api/alerts?${new URLSearchParams({ ...search, page: 1, pageSize: 10 }).toString()}`),
      ])
      setByTime(t.data.map(d => ({ time: d._id, count: d.count })))
      setByLabel(l.data.map(d => ({ name: d._id, value: d.count })))
      setRecent(a.data.items || [])
      setLoading(false)
    } catch (e) {
      console.error('Error loading data:', e)
      const status = e.response?.status
      const errorMsg = e.response?.data?.error || e.message || 'Failed to load data'
      const mongoConnected = e.response?.data?.mongoConnected
      
      if (status === 404) {
        setError('Backend API not found. Please ensure the backend server is running on port 4000.')
      } else if (mongoConnected === false) {
        setError('MongoDB is not connected. Please ensure MongoDB is running and restart the backend server.')
      } else if (errorMsg.includes('MongoDB') || errorMsg.includes('Mongo')) {
        setError('Database connection error. Please ensure MongoDB is running.')
      } else {
        setError(errorMsg)
      }
      setLoading(false)
    }
  }
  
  // Load initial data
  useEffect(() => {
    loadData()
  }, [])
  
  // SSE is disabled - alerts only update when user clicks capture button
  // This prevents automatic background updates
  
  async function apply() {
    const a = await api.get(`/api/alerts?${new URLSearchParams({ ...search, page: 1, pageSize: 10 }).toString()}`)
    setRecent(a.data.items)
  }

  async function startCapture() {
    setCapturing(true)
    setCaptureResults(null)
    setError(null)
    
    try {
      // Show progress message
      console.log(`Starting capture for ${captureDuration} seconds...`)
      
      const response = await api.post('/api/capture/start', { duration: captureDuration })
      
      // Wait for the response (backend waits for capture to complete)
      if (response.data && response.data.alerts) {
        setCaptureResults(response.data)
        // Update recent alerts with captured ones
        setRecent(response.data.alerts.slice(0, 10))
      }
      
      // Reload dashboard data to update stats and charts
      await loadData()
      
      console.log(`Capture completed: ${response.data?.alertsCaptured || 0} alerts found`)
    } catch (e) {
      console.error('Capture error:', e)
      setError(e.response?.data?.error || e.message || 'Capture failed')
    } finally {
      setCapturing(false)
      setCaptureDialogOpen(false)
    }
  }

  async function handleFileUpload() {
    if (!uploadFile) {
      setError('Please select a file')
      return
    }

    setUploading(true)
    setUploadResults(null)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', uploadFile)

      const response = await api.post('/api/upload/dataset', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 120000 // 2 minutes timeout for file upload (should return immediately for large files)
      })

      // Check if it's a background processing response
      if (response.data.processing) {
        setUploadResults({
          ...response.data,
          message: response.data.message,
          processing: true
        })
        // Poll for results every 10 seconds
        const pollInterval = setInterval(async () => {
          try {
            await loadData() // Refresh to get new alerts
            // Check if we have new alerts (simple heuristic: if recent alerts increased significantly)
            // For now, just keep polling and show message
          } catch (e) {
            console.error('Polling error:', e)
          }
        }, 10000) // Poll every 10 seconds
        
        // Stop polling after 2 hours
        setTimeout(() => {
          clearInterval(pollInterval)
          if (uploading) {
            setError('Processing is taking longer than expected. Please check back later or refresh the page.')
            setUploading(false)
          }
        }, 7200000) // 2 hours
        
        // Don't close dialog yet
        return
      }

      setUploadResults(response.data)
      // Update recent alerts with uploaded ones
      if (response.data.alerts && response.data.alerts.length > 0) {
        setRecent(response.data.alerts.slice(0, 10))
      }
      
      // Reload dashboard data
      await loadData()
      
      console.log(`Upload completed: ${response.data.totalAlerts} alerts processed`)
    } catch (e) {
      console.error('Upload error:', e)
      setError(e.response?.data?.error || e.message || 'Upload failed')
    } finally {
      setUploading(false)
      setUploadDialogOpen(false)
      setUploadFile(null)
    }
  }
  const totalAlerts = byLabel.reduce((s, r) => s + r.value, 0)
  
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <Typography variant="h6">Loading dashboard...</Typography>
      </Box>
    )
  }
  
  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Paper sx={{ p: 2, bgcolor: 'error.light' }}>
          <Typography color="error">Error: {error}</Typography>
          <Button onClick={loadData} sx={{ mt: 2 }}>Retry</Button>
        </Paper>
      </Box>
    )
  }
  
  return (
    <Box sx={{ display: 'grid', gap: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <Typography variant="h5">IDS Dashboard</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button 
            variant="outlined" 
            color="secondary" 
            onClick={() => setUploadDialogOpen(true)}
            disabled={uploading || capturing}
          >
            {uploading ? 'Processing...' : 'Upload Dataset'}
          </Button>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={() => setCaptureDialogOpen(true)}
            disabled={capturing || uploading}
          >
            {capturing ? 'Capturing...' : 'Capture Live Data'}
          </Button>
        </Box>
      </Box>

      {/* Capture Dialog */}
      <Dialog open={captureDialogOpen} onClose={() => !capturing && setCaptureDialogOpen(false)}>
        <DialogTitle>Capture Live Network Data</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Duration (seconds)"
              type="number"
              value={captureDuration}
              onChange={(e) => setCaptureDuration(parseInt(e.target.value) || 10)}
              inputProps={{ min: 1, max: 300 }}
              disabled={capturing}
              helperText="Enter how many seconds to capture network data"
            />
            {capturing && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 2 }}>
                <CircularProgress size={24} />
                <Typography>Capturing data for {captureDuration} seconds...</Typography>
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCaptureDialogOpen(false)} disabled={capturing}>Cancel</Button>
          <Button onClick={startCapture} variant="contained" disabled={capturing || captureDuration < 1}>
            Start Capture
          </Button>
        </DialogActions>
      </Dialog>

      {/* Upload Dialog */}
      <Dialog open={uploadDialogOpen} onClose={() => !uploading && setUploadDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Upload Recorded Dataset</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Upload a .txt or .csv file containing network flow data. 
              Supports files up to 1GB and variable column counts (will be auto-adjusted to 41 features).
            </Typography>
            <input
              accept=".txt,.csv"
              style={{ display: 'none' }}
              id="file-upload"
              type="file"
              onChange={(e) => setUploadFile(e.target.files[0])}
              disabled={uploading}
            />
            <label htmlFor="file-upload">
              <Button variant="outlined" component="span" disabled={uploading} fullWidth sx={{ mb: 2 }}>
                {uploadFile ? uploadFile.name : 'Select File'}
              </Button>
            </label>
            {uploadFile && (
              <Typography variant="caption" color="text.secondary">
                File: {uploadFile.name} ({(uploadFile.size / 1024).toFixed(2)} KB)
              </Typography>
            )}
            {uploading && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress />
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {uploadResults?.processing 
                    ? uploadResults.message || 'Processing large dataset in background. This may take 10-30 minutes. Results will appear automatically when complete.'
                    : 'Processing dataset and generating predictions...'}
                </Typography>
                {uploadResults?.processing && (
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Estimated rows: {uploadResults.estimatedRows?.toLocaleString() || 'N/A'} | 
                    You can close this dialog and check back later. The page will auto-refresh when complete.
                  </Typography>
                )}
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setUploadDialogOpen(false); setUploadFile(null) }} disabled={uploading}>
            Cancel
          </Button>
          <Button onClick={handleFileUpload} variant="contained" disabled={uploading || !uploadFile}>
            {uploading ? 'Processing...' : 'Upload & Analyze'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Capture Results */}
      {captureResults && (
        <Alert severity="success" onClose={() => setCaptureResults(null)}>
          Capture completed! Found {captureResults.alertsCaptured} alerts in {captureResults.duration} seconds.
        </Alert>
      )}

      {/* Upload Results */}
      {uploadResults && (
        <Alert severity="success" onClose={() => setUploadResults(null)} sx={{ mb: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Dataset processed successfully!
          </Typography>
          <Typography variant="body2">
            File: {uploadResults.filename} | Total Alerts: {uploadResults.totalAlerts} | 
            Average Confidence: {(uploadResults.averageConfidence * 100).toFixed(1)}%
          </Typography>
          {uploadResults.labelDistribution && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption">Label Distribution: </Typography>
              {Object.entries(uploadResults.labelDistribution).map(([label, count]) => (
                <Chip key={label} label={`${label}: ${count}`} size="small" sx={{ mr: 0.5, mt: 0.5 }} />
              ))}
            </Box>
          )}
        </Alert>
      )}

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}><Paper sx={{ p: 2 }}><Typography variant="overline">Total Alerts</Typography><Typography variant="h4">{totalAlerts}</Typography></Paper></Grid>
        <Grid item xs={12} md={4}><Paper sx={{ p: 2 }}><Typography variant="overline">Active Threats</Typography><Typography variant="h4">{recent.filter(r => r.label !== 'normal').length}</Typography></Paper></Grid>
        <Grid item xs={12} md={4}><Paper sx={{ p: 2 }}><Typography variant="overline">System</Typography><Typography variant="h6">Healthy</Typography></Paper></Grid>
      </Grid>
      <Grid container spacing={2}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: 320 }}>
            <Typography variant="subtitle1">Attack Trends</Typography>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={byTime}><XAxis dataKey="time" hide /><YAxis /><Tooltip /><Line type="monotone" dataKey="count" stroke="#1976d2" /></LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 320 }}>
            <Typography variant="subtitle1">Attack Types</Typography>
            <ResponsiveContainer width="100%" height="85%">
              <PieChart><Pie data={byLabel} dataKey="value" nameKey="name" outerRadius={90}>
                {byLabel.map((entry, index) => (<Cell key={entry.name} fill={COLORS[index % COLORS.length]} />))}
              </Pie></PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
      <Paper sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
          <TextField size="small" label="Label filter" value={search.label} onChange={(e) => setSearch({ ...search, label: e.target.value })} />
          <TextField size="small" label="Min confidence" type="number" inputProps={{ step: 0.05, min: 0, max: 1 }} value={search.minConfidence} onChange={(e) => setSearch({ ...search, minConfidence: e.target.value })} />
          <Button onClick={apply} variant="contained">Apply</Button>
        </Box>
        <Typography variant="subtitle1" sx={{ mb: 1 }}>Recent Alerts</Typography>
        {recent.length === 0 ? (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography color="text.secondary">No alerts found. Click "Capture Live Data" to start capturing.</Typography>
          </Box>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Time</TableCell>
                <TableCell>Label</TableCell>
                <TableCell>Confidence</TableCell>
                <TableCell>Severity</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {recent.map((a) => (
                <TableRow key={a._id || a.id}>
                  <TableCell>{new Date(a.occurredAt).toLocaleString()}</TableCell>
                  <TableCell>{a.label}</TableCell>
                  <TableCell>{a.confidence?.toFixed ? a.confidence.toFixed(3) : a.confidence}</TableCell>
                  <TableCell>
                    <Chip size="small" label={a.severity || 'low'} color={a.severity === 'critical' ? 'error' : a.severity === 'high' ? 'warning' : a.severity === 'medium' ? 'info' : 'default'} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
        
        {/* Show capture results if available */}
        {captureResults && captureResults.alerts && captureResults.alerts.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>Capture Results ({captureResults.alertsCaptured} alerts)</Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>Label</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Severity</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {captureResults.alerts.slice(0, 20).map((a) => (
                  <TableRow key={a.id}>
                    <TableCell>{new Date(a.occurredAt).toLocaleString()}</TableCell>
                    <TableCell>{a.label}</TableCell>
                    <TableCell>{a.confidence?.toFixed ? a.confidence.toFixed(3) : a.confidence}</TableCell>
                    <TableCell>
                      <Chip size="small" label={a.severity || 'low'} color={a.severity === 'critical' ? 'error' : a.severity === 'high' ? 'warning' : a.severity === 'medium' ? 'info' : 'default'} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}

        {/* Show upload results if available */}
        {uploadResults && uploadResults.alerts && uploadResults.alerts.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Upload Results - {uploadResults.filename} ({uploadResults.totalAlerts} alerts)
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>Label</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Severity</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {uploadResults.alerts.map((a) => (
                  <TableRow key={a.id}>
                    <TableCell>{new Date(a.occurredAt).toLocaleString()}</TableCell>
                    <TableCell>{a.label}</TableCell>
                    <TableCell>{a.confidence?.toFixed ? a.confidence.toFixed(3) : a.confidence}</TableCell>
                    <TableCell>
                      <Chip size="small" label={a.severity || 'low'} color={a.severity === 'critical' ? 'error' : a.severity === 'high' ? 'warning' : a.severity === 'medium' ? 'info' : 'default'} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {uploadResults.totalAlerts > 100 && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Showing first 100 of {uploadResults.totalAlerts} alerts. Check the Alerts page for all results.
              </Typography>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  )
}


