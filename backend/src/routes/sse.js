import express from 'express'
import { requireAuth } from '../middleware/auth.js'

const router = express.Router()
const clients = new Set()

router.get('/stream', requireAuth, (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders()
  res.write(`event: ping\ndata: connected\n\n`)
  clients.add(res)
  req.on('close', () => clients.delete(res))
})

export function broadcastAlert(alert) {
  const payload = `data: ${JSON.stringify(alert)}\n\n`
  for (const res of clients) {
    res.write(payload)
  }
}

export default router


