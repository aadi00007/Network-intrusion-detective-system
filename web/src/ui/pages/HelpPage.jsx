import React from 'react'
import { Paper, Typography } from '@mui/material'

export default function HelpPage() {
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 1 }}>Help</Typography>
      <Typography variant="body2">Use Dashboard to monitor real-time and historical alerts. Use Alerts to search/export. Admins can manage users in Users.</Typography>
    </Paper>
  )
}


