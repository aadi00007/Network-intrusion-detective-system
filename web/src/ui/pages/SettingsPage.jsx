import React, { useState } from 'react'
import { Paper, Switch, FormControlLabel, Typography } from '@mui/material'

export default function SettingsPage() {
  const [dark, setDark] = useState(false)
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 1 }}>Settings</Typography>
      <FormControlLabel control={<Switch checked={dark} onChange={(e) => setDark(e.target.checked)} />} label="Dark mode (placeholder)" />
    </Paper>
  )
}


