import express from 'express'
import mongoose from 'mongoose'
import AlertEvent from '../models/AlertEvent.js'
import { broadcastAlert } from './sse.js'

const router = express.Router()

// Helper to check MongoDB connection
function checkMongoConnection() {
  if (mongoose.connection.readyState !== 1) {
    throw new Error('MongoDB is not connected. Please ensure MongoDB is running.')
  }
}

router.post('/', async (req, res) => {
  const { label, confidence, severity, features, raw, source, meta, occurredAt } = req.body
  if (!label) return res.status(400).json({ error: 'label required' })
  const doc = await AlertEvent.create({ label, confidence, severity, features, raw, source, meta, occurredAt })
  
  // Broadcast the new alert to all connected SSE clients
  broadcastAlert({
    id: doc._id,
    label: doc.label,
    confidence: doc.confidence,
    severity: doc.severity,
    occurredAt: doc.occurredAt,
    source: doc.source
  })
  
  res.json({ id: doc._id })
})

router.get('/', async (req, res) => {
  try {
    checkMongoConnection()
    
    const { q, label, minConfidence, severity, from, to, page = 1, pageSize = 50 } = req.query
    const filter = {}
    if (label) filter.label = label
    if (severity) filter.severity = severity
    if (minConfidence) filter.confidence = { $gte: Number(minConfidence) }
    if (from || to) filter.occurredAt = { ...(from ? { $gte: new Date(from) } : {}), ...(to ? { $lte: new Date(to) } : {}) }
    // Only use $text search if q is provided and not empty
    if (q && q.trim()) {
      try {
        filter.$text = { $search: String(q) }
      } catch (textErr) {
        // If text index doesn't exist, fall back to regex search on label
        console.warn('Text search not available, using regex fallback:', textErr.message)
        filter.label = { $regex: String(q), $options: 'i' }
      }
    }
    const skip = (Number(page) - 1) * Number(pageSize)
    const [items, total] = await Promise.all([
      AlertEvent.find(filter).sort({ occurredAt: -1 }).skip(skip).limit(Number(pageSize)),
      AlertEvent.countDocuments(filter),
    ])
    res.json({ items: items || [], total: total || 0, page: Number(page), pageSize: Number(pageSize) })
  } catch (err) {
    console.error('Error in GET /alerts:', err)
    console.error('Error stack:', err.stack)
    const errorMessage = err.message || 'Failed to get alerts'
    res.status(500).json({ 
      error: errorMessage,
      mongoConnected: mongoose.connection.readyState === 1,
      details: process.env.NODE_ENV === 'development' ? err.stack : undefined 
    })
  }
})

router.get('/stats/by-label', async (_req, res) => {
  try {
    checkMongoConnection()
    
    // Check if collection exists and has data
    const count = await AlertEvent.countDocuments().catch(() => 0)
    if (count === 0) {
      return res.json([]) // Return empty array if no data
    }
    
    const rows = await AlertEvent.aggregate([
      { $match: { label: { $exists: true, $ne: null } } }, // Filter out null labels
      { $group: { _id: '$label', count: { $sum: 1 } } },
      { $sort: { count: -1 } },
    ])
    res.json(rows || [])
  } catch (err) {
    console.error('Error in /stats/by-label:', err)
    console.error('Error stack:', err.stack)
    const errorMessage = err.message || 'Failed to get label statistics'
    res.status(500).json({ 
      error: errorMessage,
      mongoConnected: mongoose.connection.readyState === 1
    })
  }
})

router.get('/stats/by-time', async (req, res) => {
  try {
    checkMongoConnection()
    
    // Check if collection exists and has data
    const count = await AlertEvent.countDocuments().catch(() => 0)
    if (count === 0) {
      return res.json([]) // Return empty array if no data
    }
    
    const { interval = 'hour' } = req.query
    const dateFormat = interval === 'day' ? '%Y-%m-%d' : '%Y-%m-%d %H:00'
    const rows = await AlertEvent.aggregate([
      { $match: { occurredAt: { $exists: true, $ne: null } } }, // Filter out null dates
      { $group: { _id: { $dateToString: { format: dateFormat, date: '$occurredAt' } }, count: { $sum: 1 } } },
      { $sort: { _id: 1 } },
    ])
    res.json(rows || [])
  } catch (err) {
    console.error('Error in /stats/by-time:', err)
    console.error('Error stack:', err.stack)
    const errorMessage = err.message || 'Failed to get time statistics'
    res.status(500).json({ 
      error: errorMessage,
      mongoConnected: mongoose.connection.readyState === 1
    })
  }
})

router.get('/export/csv', async (_req, res) => {
  const items = await AlertEvent.find({}).sort({ occurredAt: -1 }).limit(10000)
  const cols = ['occurredAt', 'label', 'confidence', 'severity']
  const header = cols.join(',')
  const rows = items.map((i) => [i.occurredAt.toISOString(), i.label, i.confidence ?? '', i.severity ?? ''].join(','))
  res.setHeader('Content-Type', 'text/csv')
  res.setHeader('Content-Disposition', 'attachment; filename="alerts.csv"')
  res.send([header, ...rows].join('\n'))
})

export default router


