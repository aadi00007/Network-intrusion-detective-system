import express from 'express'
import AlertEvent from '../models/AlertEvent.js'
import { requireAuth } from '../middleware/auth.js'

const router = express.Router()

router.post('/', requireAuth, async (req, res) => {
  const { label, confidence, severity, features, raw, source, meta, occurredAt } = req.body
  if (!label) return res.status(400).json({ error: 'label required' })
  const doc = await AlertEvent.create({ label, confidence, severity, features, raw, source, meta, occurredAt })
  res.json({ id: doc._id })
})

router.get('/', requireAuth, async (req, res) => {
  const { q, label, minConfidence, severity, from, to, page = 1, pageSize = 50 } = req.query
  const filter = {}
  if (label) filter.label = label
  if (severity) filter.severity = severity
  if (minConfidence) filter.confidence = { $gte: Number(minConfidence) }
  if (from || to) filter.occurredAt = { ...(from ? { $gte: new Date(from) } : {}), ...(to ? { $lte: new Date(to) } : {}) }
  if (q) filter.$text = { $search: String(q) }
  const skip = (Number(page) - 1) * Number(pageSize)
  const [items, total] = await Promise.all([
    AlertEvent.find(filter).sort({ occurredAt: -1 }).skip(skip).limit(Number(pageSize)),
    AlertEvent.countDocuments(filter),
  ])
  res.json({ items, total, page: Number(page), pageSize: Number(pageSize) })
})

router.get('/stats/by-label', requireAuth, async (_req, res) => {
  const rows = await AlertEvent.aggregate([
    { $group: { _id: '$label', count: { $sum: 1 } } },
    { $sort: { count: -1 } },
  ])
  res.json(rows)
})

router.get('/stats/by-time', requireAuth, async (req, res) => {
  const { interval = 'hour' } = req.query
  const dateFormat = interval === 'day' ? '%Y-%m-%d' : '%Y-%m-%d %H:00'
  const rows = await AlertEvent.aggregate([
    { $group: { _id: { $dateToString: { format: dateFormat, date: '$occurredAt' } }, count: { $sum: 1 } } },
    { $sort: { _id: 1 } },
  ])
  res.json(rows)
})

router.get('/export/csv', requireAuth, async (_req, res) => {
  const items = await AlertEvent.find({}).sort({ occurredAt: -1 }).limit(10000)
  const cols = ['occurredAt', 'label', 'confidence', 'severity']
  const header = cols.join(',')
  const rows = items.map((i) => [i.occurredAt.toISOString(), i.label, i.confidence ?? '', i.severity ?? ''].join(','))
  res.setHeader('Content-Type', 'text/csv')
  res.setHeader('Content-Disposition', 'attachment; filename="alerts.csv"')
  res.send([header, ...rows].join('\n'))
})

export default router


