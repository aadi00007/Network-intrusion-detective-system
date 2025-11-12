import express from 'express'
import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { requireAuth } from '../middleware/auth.js'
import AlertEvent from '../models/AlertEvent.js'
import { broadcastAlert } from './sse.js'

const router = express.Router()
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

function runPythonPredict(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    // Prefer explicit override, else fall back to system Python. Avoid non-existent repo venv by default.
    const python = process.env.PYTHON_BIN || '/usr/bin/python3'
    const script = process.env.PYTHON_SCRIPT || path.resolve(__dirname, '../../../nsl_kdd_analysis.py')
    const args = [script, 'predict', '--model_path', 'models/nsl_kdd_model.joblib', '--label_map_path', 'models/label_map.joblib', '--input_path', inputPath, '--output_path', outputPath]
    const proc = spawn(python, args, { cwd: path.resolve(__dirname, '../../..') })
    proc.stdout.on('data', (d) => process.stdout.write(d))
    proc.stderr.on('data', (d) => process.stderr.write(d))
    proc.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`predict exited ${code}`))))
  })
}

router.post('/batch', requireAuth, async (req, res) => {
  try {
    const { inputPath } = req.body
    if (!inputPath) return res.status(400).json({ error: 'inputPath required (absolute or repo-relative)' })
    const repoRoot = path.resolve(__dirname, '../../..')
    const resolvedInput = path.isAbsolute(inputPath) ? inputPath : path.resolve(repoRoot, inputPath)
    if (!fs.existsSync(resolvedInput)) {
      return res.status(400).json({ error: `inputPath not found: ${resolvedInput}` })
    }

    const outPath = path.resolve(repoRoot, 'tmp_predictions.csv')
    await runPythonPredict(resolvedInput, outPath)

    if (!fs.existsSync(outPath)) {
      return res.status(500).json({ error: 'Prediction output not found' })
    }
    const text = fs.readFileSync(outPath, 'utf-8').trim()
    if (!text) {
      return res.json({ inserted: 0 })
    }
    const rows = text.split('\n').map((l) => l.split(','))
    let count = 0
    for (const row of rows) {
      if (row.length < 2) continue
      const label = row[row.length - 2]
      const conf = Number(row[row.length - 1])
      if (!Number.isFinite(conf)) continue
      const severity = conf >= 0.95 ? 'critical' : conf >= 0.85 ? 'high' : conf >= 0.7 ? 'medium' : 'low'
      const alert = await AlertEvent.create({ label, confidence: conf, severity, raw: row, source: 'batch' })
      broadcastAlert({ id: alert._id, label, confidence: conf, severity, occurredAt: alert.occurredAt })
      count++
    }
    res.json({ inserted: count })
  } catch (err) {
    console.error('Batch ingest failed:', err)
    res.status(500).json({ error: String(err.message || err) })
  }
})

export default router


