import mongoose from 'mongoose'
import bcrypt from 'bcryptjs'
import User from '../models/User.js'

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/ids'
const EMAIL = process.env.SEED_EMAIL || 'admin@example.com'
const PASSWORD = process.env.SEED_PASSWORD || 'admin123'

async function main() {
  await mongoose.connect(MONGO_URI)
  const exists = await User.findOne({ email: EMAIL })
  if (exists) {
    console.log('Admin already exists:', EMAIL)
  } else {
    const passwordHash = await bcrypt.hash(PASSWORD, 10)
    const user = await User.create({ email: EMAIL, passwordHash, role: 'admin' })
    console.log('Seeded admin user:', user.email)
  }
  await mongoose.disconnect()
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})


