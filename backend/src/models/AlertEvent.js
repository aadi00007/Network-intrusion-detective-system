import mongoose from 'mongoose'

const AlertEventSchema = new mongoose.Schema(
  {
    label: { type: String, index: true },
    confidence: { type: Number, index: true },
    severity: { type: String, enum: ['low', 'medium', 'high', 'critical'], index: true },
    features: { type: Array, default: [] },
    raw: { type: Array, default: [] },
    source: { type: String, default: 'batch', index: true },
    meta: { type: Object, default: {} },
    occurredAt: { type: Date, default: Date.now, index: true },
  },
  { timestamps: true }
)

AlertEventSchema.index({ occurredAt: -1 })
AlertEventSchema.index({ label: 1, occurredAt: -1 })

export default mongoose.model('AlertEvent', AlertEventSchema)


