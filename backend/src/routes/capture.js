import express from 'express'
import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'
import AlertEvent from '../models/AlertEvent.js'
import { broadcastAlert } from './sse.js'

const router = express.Router()
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

router.post('/start', async (req, res) => {
  try {
    const { duration } = req.body
    if (!duration || duration < 1) {
      return res.status(400).json({ error: 'Duration must be at least 1 second' })
    }

    // On Windows, use 'py' (Python launcher) which is more reliable than 'python'
  const python = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'py' : '/usr/bin/python3')
    const script = path.resolve(__dirname, '../../../live_capture_to_api.py')
    const repoRoot = path.resolve(__dirname, '../../..')
    
    // Record start time to get alerts created during this session
    const sessionStartTime = new Date()
    
    // Start the capture process
    const proc = spawn(python, [
      script,
      '--interval', '1',  // Capture every 1 second for faster updates
      '--duration', duration.toString()
    ], {
      cwd: repoRoot,
      stdio: ['ignore', 'pipe', 'pipe']
    })

    let output = ''
    let errorOutput = ''

    proc.stdout.on('data', (data) => {
      output += data.toString()
      console.log(`[Capture] ${data.toString()}`)
    })

    proc.stderr.on('data', (data) => {
      errorOutput += data.toString()
      console.error(`[Capture Error] ${data.toString()}`)
    })

    // Wait for the process to complete
    proc.on('close', async (code) => {
      try {
        // Wait a bit for all alerts to be saved
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        // Get the alerts created during this capture session
        const alerts = await AlertEvent.find({
          source: 'live_capture',
          occurredAt: { $gte: sessionStartTime }
        }).sort({ occurredAt: -1 }).limit(1000)

        // Send response (this will work if client is still waiting)
        if (!res.headersSent) {
          res.json({
            success: true,
            duration: duration,
            alertsCaptured: alerts.length,
            alerts: alerts.map(a => ({
              id: a._id,
              label: a.label,
              confidence: a.confidence,
              severity: a.severity,
              occurredAt: a.occurredAt,
              features: a.features
            }))
          })
        }
      } catch (err) {
        console.error('Error getting alerts:', err)
        if (!res.headersSent) {
          res.status(500).json({ error: String(err.message || err) })
        }
      }
    })

    // Set timeout to prevent hanging
    setTimeout(() => {
      if (!res.headersSent) {
        proc.kill()
        res.status(500).json({ error: 'Capture timeout' })
      }
    }, (duration + 10) * 1000)

  } catch (err) {
    console.error('Capture failed:', err)
    if (!res.headersSent) {
      res.status(500).json({ error: String(err.message || err) })
    }
  }
})

export default router

