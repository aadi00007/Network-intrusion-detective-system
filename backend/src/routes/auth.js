import express from 'express'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import User from '../models/User.js'
import { requireAuth, requireRole } from '../middleware/auth.js'

const router = express.Router()

router.post('/register', requireAuth, requireRole('admin'), async (req, res) => {
  const { email, password, role } = req.body
  if (!email || !password) return res.status(400).json({ error: 'Email and password required' })
  const exists = await User.findOne({ email })
  if (exists) return res.status(409).json({ error: 'User exists' })
  const passwordHash = await bcrypt.hash(password, 10)
  const user = await User.create({ email, passwordHash, role: role || 'analyst' })
  res.json({ id: user._id, email: user.email, role: user.role })
})

router.post('/login', async (req, res) => {
  const { email, password } = req.body
  const user = await User.findOne({ email })
  if (!user) return res.status(401).json({ error: 'Invalid credentials' })
  const ok = await bcrypt.compare(password, user.passwordHash)
  if (!ok) return res.status(401).json({ error: 'Invalid credentials' })
  const token = jwt.sign({ sub: user._id.toString(), email: user.email, role: user.role }, process.env.JWT_SECRET || 'devsecret', { expiresIn: '12h' })
  res.json({ token, role: user.role })
})

router.get('/me', requireAuth, (req, res) => {
  res.json({ user: req.user })
})

// Admin: list users
router.get('/users', requireAuth, requireRole('admin'), async (_req, res) => {
  const users = await User.find({}, { email: 1, role: 1, createdAt: 1 }).sort({ createdAt: -1 })
  res.json(users.map(u => ({ id: u._id, email: u.email, role: u.role, createdAt: u.createdAt })))
})

// Admin: delete user
router.delete('/users/:id', requireAuth, requireRole('admin'), async (req, res) => {
  const { id } = req.params
  await User.deleteOne({ _id: id })
  res.json({ ok: true })
})

export default router


