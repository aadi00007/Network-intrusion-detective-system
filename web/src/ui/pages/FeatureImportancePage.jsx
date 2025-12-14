import React, { useEffect, useState } from 'react'
import { Paper, Typography } from '@mui/material'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function FeatureImportancePage() {
  const [rows, setRows] = useState([])
  useEffect(() => {
    // Placeholder: If you later expose feature importances via an API, fetch and plot here.
    // For now, show a note and empty chart.
    setRows([])
  }, [])
  return (
    <Paper sx={{ p: 2, height: 400 }}>
      <Typography variant="subtitle1">Feature Importance</Typography>
      <Typography variant="body2" sx={{ mb: 1 }}>Expose importances via an API to display here.</Typography>
      <ResponsiveContainer width="100%" height="85%">
        <BarChart data={rows}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" hide /><YAxis /><Tooltip /><Bar dataKey="value" fill="#1976d2" /></BarChart>
      </ResponsiveContainer>
    </Paper>
  )
}


