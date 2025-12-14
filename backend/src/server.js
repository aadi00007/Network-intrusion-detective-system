import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import morgan from 'morgan'
import mongoose from 'mongoose'
import rateLimit from 'express-rate-limit'
import path from 'path'
import { fileURLToPath } from 'url'
import { config } from 'dotenv'
config()

import authRouter from './routes/auth.js'
import alertsRouter from './routes/alerts.js'
import sseRouter from './routes/sse.js'
import mlRouter from './routes/ml.js'
import captureRouter from './routes/capture.js'
import uploadRouter from './routes/upload.js'

const app = express()
const PORT = process.env.PORT || 4000
const ORIGIN = process.env.ALLOWED_ORIGIN || 'http://localhost:5173'
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

app.use(helmet())
app.use(cors({ origin: ORIGIN, credentials: true }))
app.use(express.json({ limit: '1mb' }))
app.use(morgan('dev'))

// Rate limiter - increased limits for development
const limiter = rateLimit({ 
  windowMs: 60 * 1000, // 1 minute
  limit: 1000, // Increased from 200 to 1000 requests per minute
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
})
app.use(limiter)

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, time: new Date().toISOString() })
})

app.use('/api/auth', authRouter)
app.use('/api/alerts', alertsRouter)
app.use('/api/events', sseRouter)
app.use('/api/ml', mlRouter)
app.use('/api/capture', captureRouter)
app.use('/api/upload', uploadRouter)
// Serve reports and model artifacts for frontend visualization (read-only)
app.use('/assets/reports', express.static(path.resolve(__dirname, '../../reports')))
app.use('/assets/models', express.static(path.resolve(__dirname, '../../models')))

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/ids'
mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log('MongoDB connected successfully')
    app.listen(PORT, () => console.log(`API listening on http://localhost:${PORT}`))
  })
  .catch((err) => {
    console.error('Mongo connection error:', err)
    console.error('Please ensure MongoDB is running and accessible at:', MONGO_URI)
    // Don't exit - allow server to start but API calls will fail with clear errors
    app.listen(PORT, () => {
      console.log(`API listening on http://localhost:${PORT} (MongoDB not connected)`)
      console.warn('WARNING: MongoDB connection failed. API endpoints will return errors.')
    })
  })

// 404 handler for unknown routes
app.use('/api/*', (req, res) => {
  console.warn(`404: Route not found: ${req.method} ${req.originalUrl}`)
  res.status(404).json({ error: `Route not found: ${req.method} ${req.path}` })
})

// Express error handler (last middleware)
// Ensures unexpected route errors don't crash the process
// eslint-disable-next-line no-unused-vars
app.use((err, _req, res, _next) => {
  console.error('Unhandled route error:', err)
  res.status(500).json({ error: 'Internal server error' })
})

// Global process-level handlers to avoid hard crashes
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason)
})
process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err)
})


