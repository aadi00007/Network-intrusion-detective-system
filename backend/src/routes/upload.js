import express from 'express'
import multer from 'multer'
import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import AlertEvent from '../models/AlertEvent.js'
import { broadcastAlert } from './sse.js'

const router = express.Router()
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Configure multer for file uploads - increased limit to 1GB
const upload = multer({
  dest: path.resolve(__dirname, '../../../tmp_uploads'),
  limits: { fileSize: 1024 * 1024 * 1024 }, // 1GB limit (increased from 100MB)
  fileFilter: (req, file, cb) => {
    // Accept text files and CSV
    if (file.mimetype === 'text/plain' || 
        file.mimetype === 'text/csv' || 
        file.originalname.endsWith('.txt') || 
        file.originalname.endsWith('.csv')) {
      cb(null, true)
    } else {
      cb(new Error('Only .txt and .csv files are allowed'))
    }
  }
})

// Ensure upload directory exists
const uploadDir = path.resolve(__dirname, '../../../tmp_uploads')
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true })
}

router.post('/dataset', upload.single('file'), async (req, res) => {
  let uploadedFile = null
  let outputPath = null
  
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' })
    }

    uploadedFile = req.file.path
    const originalName = req.file.originalname
    const fileSizeMB = req.file.size / (1024 * 1024)
    
    console.log(`Processing uploaded file: ${originalName} (${fileSizeMB.toFixed(2)} MB)`)
    console.log(`Uploaded to: ${uploadedFile}`)
    
    // For large files, return immediately and process in background
    const estimatedRows = Math.max(10000, fileSizeMB * 1000)
    const isLargeFile = estimatedRows > 100000 // >100k rows
    
    if (isLargeFile) {
      console.log(`Large file detected (${Math.round(estimatedRows).toLocaleString()} estimated rows). Processing in background...`)
      
      // Return immediately with a message
      res.json({
        success: true,
        message: `Large dataset detected. Processing ${Math.round(estimatedRows).toLocaleString()} rows in background. This may take 10-30 minutes. Results will appear automatically when complete.`,
        processing: true,
        estimatedRows: Math.round(estimatedRows),
        filename: originalName
      })
      
      // Process in background (don't await)
      processUploadInBackground(uploadedFile, originalName, estimatedRows).catch(err => {
        console.error('Background processing error:', err)
      })
      
      return // Exit early
    }

    // Use flexible prediction script that handles variable columns
    // On Windows, use 'py' (Python launcher) which is more reliable than 'python'
    const python = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'py' : '/usr/bin/python3')
    const script = path.resolve(__dirname, '../../../flexible_predict.py')
    const repoRoot = path.resolve(__dirname, '../../..')
    outputPath = path.resolve(repoRoot, 'tmp_upload_predictions.csv')
    
    // Verify files exist
    if (!fs.existsSync(uploadedFile)) {
      return res.status(500).json({ error: 'Uploaded file not found' })
    }
    if (!fs.existsSync(path.resolve(repoRoot, 'models/nsl_kdd_hdc.joblib'))) {
      return res.status(500).json({ error: 'Model file not found' })
    }
    if (!fs.existsSync(path.resolve(repoRoot, 'models/label_map.joblib'))) {
      return res.status(500).json({ error: 'Label map file not found' })
    }
    
    // Calculate timeout based on file size: ~1 second per 1000 rows, minimum 10 minutes, maximum 2 hours
    // For large files (692k rows), this gives plenty of time
    // fileSizeMB is already calculated on line 48
    const timeoutEstimatedRows = Math.max(10000, fileSizeMB * 1000) // Rough estimate: 1MB ≈ 1000 rows
    const timeoutMs = Math.max(600000, Math.min(7200000, timeoutEstimatedRows * 1000)) // 10 min to 2 hours
    
    console.log(`Estimated ${Math.round(timeoutEstimatedRows).toLocaleString()} rows, setting timeout to ${Math.round(timeoutMs / 60000)} minutes`)
    
    // Use flexible prediction script if available, otherwise fall back to standard
    let useFlexible = fs.existsSync(script)
    const predictionScript = useFlexible ? script : path.resolve(__dirname, '../../../nsl_kdd_analysis.py')
    
    console.log(`Using ${useFlexible ? 'flexible' : 'standard'} prediction script`)
    console.log(`Input: ${uploadedFile}`)
    console.log(`Output: ${outputPath}`)
    
    // Run prediction
    const proc = spawn(python, useFlexible ? [
      script,
      '--model_path', 'models/nsl_kdd_hdc.joblib',
      '--label_map_path', 'models/label_map.joblib',
      '--input_path', uploadedFile,
      '--output_path', outputPath,
      '--num_features', '41'  // NSL-KDD model expects 41 features
    ] : [
      predictionScript,
      'predict',
      '--model_path', 'models/nsl_kdd_hdc.joblib',
      '--label_map_path', 'models/label_map.joblib',
      '--input_path', uploadedFile,
      '--output_path', outputPath
    ], {
      cwd: repoRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: process.platform === 'win32', // Use shell on Windows for better path handling
      timeout: timeoutMs
    })

    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (data) => {
      stdout += data.toString()
      console.log(`[Prediction] ${data.toString()}`)
    })

    proc.stderr.on('data', (data) => {
      stderr += data.toString()
      console.error(`[Prediction Error] ${data.toString()}`)
    })

    // Wait for process to complete with dynamic timeout
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        proc.kill()
        reject(new Error(`Prediction timeout after ${Math.round(timeoutMs / 60000)} minutes. The dataset may be too large or processing is taking longer than expected.`))
      }, timeoutMs)
      
      proc.on('close', (code) => {
        clearTimeout(timeout)
        if (code === 0) {
          resolve()
        } else {
          reject(new Error(`Prediction failed with code ${code}: ${stderr || stdout}`))
        }
      })
      
      proc.on('error', (err) => {
        clearTimeout(timeout)
        reject(new Error(`Failed to start prediction process: ${err.message}`))
      })
    })

    // Read predictions
    if (!fs.existsSync(outputPath)) {
      console.error(`Output file not found: ${outputPath}`)
      return res.status(500).json({ error: 'Prediction output file not found. Check server logs for details.' })
    }

    const predictionsText = fs.readFileSync(outputPath, 'utf-8').trim()
    if (!predictionsText) {
      return res.status(500).json({ error: 'No predictions generated - output file is empty' })
    }

    const lines = predictionsText.split('\n').filter(line => line.trim())
    if (lines.length === 0) {
      return res.status(500).json({ error: 'No predictions in output file' })
    }
    
    console.log(`Parsing ${lines.length} prediction lines`)

    // The output format is: [original columns...] + predicted_label + confidence
    // No header row, so we need to determine indices based on expected format
    // NSL-KDD has 41 features, so predicted_label is at index 41, confidence at 42
    // But if input had label/difficulty, we need to account for that
    
    // Parse predictions and create alerts
    const alerts = []
    const sessionStartTime = new Date()

    let parsedCount = 0
    let skippedCount = 0
    
    for (let i = 0; i < lines.length; i++) {
      const row = lines[i].split(',')
      
      // The output has: [original columns] + predicted_label + confidence
      // predicted_label is second to last, confidence is last
      
      if (row.length < 2) {
        skippedCount++
        continue
      }
      
      // predicted_label is second to last, confidence is last
      const label = row[row.length - 2]?.trim()
      const confidence = parseFloat(row[row.length - 1]?.trim())

      if (!label || !isFinite(confidence) || isNaN(confidence)) {
        skippedCount++
        if (i < 5) {
          console.warn(`Skipping row ${i}: label="${label}", confidence="${row[row.length - 1]}"`)
        }
        continue
      }
      
      parsedCount++

      const severity = confidence >= 0.95 ? 'critical' : 
                      confidence >= 0.85 ? 'high' : 
                      confidence >= 0.7 ? 'medium' : 'low'

      // Create alert in database
      const alert = await AlertEvent.create({
        label,
        confidence,
        severity,
        raw: row,
        source: 'uploaded_dataset',
        occurredAt: new Date(sessionStartTime.getTime() + i * 1000) // Space them out by 1 second
      })

      // Broadcast via SSE
      broadcastAlert({
        id: alert._id,
        label: alert.label,
        confidence: alert.confidence,
        severity: alert.severity,
        occurredAt: alert.occurredAt
      })

      alerts.push({
        id: alert._id,
        label: alert.label,
        confidence: alert.confidence,
        severity: alert.severity,
        occurredAt: alert.occurredAt
      })
    }

    // Clean up uploaded file
    try {
      fs.unlinkSync(uploadedFile)
      if (fs.existsSync(outputPath)) {
        fs.unlinkSync(outputPath)
      }
    } catch (e) {
      console.warn('Failed to clean up temp files:', e)
    }

    console.log(`Successfully parsed ${parsedCount} alerts, skipped ${skippedCount} rows`)
    
    if (alerts.length === 0) {
      return res.status(500).json({ error: 'No valid alerts generated from predictions' })
    }

    // Calculate statistics
    const labelCounts = {}
    let totalConfidence = 0
    alerts.forEach(a => {
      labelCounts[a.label] = (labelCounts[a.label] || 0) + 1
      totalConfidence += a.confidence
    })

    console.log(`Returning ${alerts.length} alerts to client`)

    res.json({
      success: true,
      filename: originalName,
      totalAlerts: alerts.length,
      averageConfidence: alerts.length > 0 ? totalConfidence / alerts.length : 0,
      labelDistribution: labelCounts,
      alerts: alerts.slice(0, 100) // Return first 100 for display
    })

  } catch (err) {
    console.error('Upload processing failed:', err)
    console.error('Error stack:', err.stack)
    
    // Clean up on error
    if (uploadedFile && fs.existsSync(uploadedFile)) {
      try {
        fs.unlinkSync(uploadedFile)
        console.log('Cleaned up uploaded file')
      } catch (e) {
        console.warn('Failed to delete uploaded file:', e)
      }
    }
    
    if (outputPath && fs.existsSync(outputPath)) {
      try {
        fs.unlinkSync(outputPath)
        console.log('Cleaned up output file')
      } catch (e) {
        console.warn('Failed to delete output file:', e)
      }
    }

    // Return more detailed error message
    const errorMessage = err.message || String(err)
    console.error('Returning error to client:', errorMessage)
    
    // Don't send response if already sent
    if (!res.headersSent) {
      res.status(500).json({ 
        error: errorMessage,
        details: process.env.NODE_ENV === 'development' ? err.stack : undefined
      })
    }
  }
})

// Background processing function for large files
async function processUploadInBackground(uploadedFile, originalName, estimatedRows) {
  // On Windows, use 'py' (Python launcher) which is more reliable than 'python'
  const python = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'py' : '/usr/bin/python3')
  const script = path.resolve(__dirname, '../../../flexible_predict.py')
  const repoRoot = path.resolve(__dirname, '../../..')
  const outputPath = path.resolve(repoRoot, 'tmp_upload_predictions.csv')
  
  console.log(`[Background] Starting processing of ${originalName} (estimated ${Math.round(estimatedRows).toLocaleString()} rows)...`)
  
  const proc = spawn(python, [
    script,
    '--model_path', 'models/nsl_kdd_hdc.joblib',
    '--label_map_path', 'models/label_map.joblib',
    '--input_path', uploadedFile,
    '--output_path', outputPath,
    '--num_features', '41'
  ], {
    cwd: repoRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: process.platform === 'win32'
  })

  let stdout = ''
  let stderr = ''

  proc.stdout.on('data', (data) => {
    stdout += data.toString()
    const msg = data.toString()
    console.log(`[Background Prediction] ${msg}`)
    if (msg.includes('Processing batch')) {
      console.log(`[Background] ${msg.trim()}`)
    }
  })

  proc.stderr.on('data', (data) => {
    stderr += data.toString()
    console.error(`[Background Prediction Error] ${data.toString()}`)
  })

  try {
    await new Promise((resolve, reject) => {
      proc.on('close', (code) => {
        if (code === 0) {
          resolve()
        } else {
          reject(new Error(`Prediction failed with code ${code}: ${stderr || 'No stderr output'}`))
        }
      })
      proc.on('error', (err) => {
        reject(new Error(`Failed to start prediction process: ${err.message}`))
      })
    })

    if (!fs.existsSync(outputPath)) {
      throw new Error('Prediction output file not found after script execution.')
    }

    const predictionsText = fs.readFileSync(outputPath, 'utf-8').trim()
    if (!predictionsText) {
      throw new Error('No predictions generated or output file is empty.')
    }

    const lines = predictionsText.split('\n')
    const predictedLabelIdx = 41
    const confidenceIdx = 42
    const alerts = []
    const sessionStartTime = new Date()

    for (let i = 0; i < lines.length; i++) {
      const row = lines[i].split(',')
      if (row.length < confidenceIdx + 1) continue

      const label = row[predictedLabelIdx]
      const confidence = parseFloat(row[confidenceIdx])

      if (!label || !isFinite(confidence)) continue

      const severity = confidence >= 0.95 ? 'critical' :
                      confidence >= 0.85 ? 'high' :
                      confidence >= 0.7 ? 'medium' : 'low'

      const alert = await AlertEvent.create({
        label,
        confidence,
        severity,
        raw: row,
        source: 'uploaded_dataset',
        occurredAt: new Date(sessionStartTime.getTime() + i * 100)
      })

      broadcastAlert({
        id: alert._id,
        label: alert.label,
        confidence: alert.confidence,
        severity: alert.severity,
        occurredAt: alert.occurredAt
      })

      alerts.push({
        id: alert._id,
        label: alert.label,
        confidence: alert.confidence,
        severity: alert.severity,
        occurredAt: alert.occurredAt
      })
    }

    console.log(`[Background] Completed processing ${originalName}: ${alerts.length} alerts created`)
    
    if (uploadedFile && fs.existsSync(uploadedFile)) {
      try { fs.unlinkSync(uploadedFile) } catch (e) { }
    }
    if (outputPath && fs.existsSync(outputPath)) {
      try { fs.unlinkSync(outputPath) } catch (e) { }
    }
    
  } catch (err) {
    console.error('[Background] Upload processing failed:', err)
    if (uploadedFile && fs.existsSync(uploadedFile)) {
      try { fs.unlinkSync(uploadedFile) } catch (e) { }
    }
    if (outputPath && fs.existsSync(outputPath)) {
      try { fs.unlinkSync(outputPath) } catch (e) { }
    }
  }
}

export default router

