import React, { useEffect, useState } from 'react'
import { Grid, Paper, Typography, Box, TextField, Button, Chip } from '@mui/material'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { useAuth } from '../state/auth.jsx'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8e44ad', '#e74c3c']

export default function DashboardPage() {
  const { api } = useAuth()
  const [byTime, setByTime] = useState([])
  const [byLabel, setByLabel] = useState([])
  const [search, setSearch] = useState({ label: '', minConfidence: 0.7 })
  const [recent, setRecent] = useState([])
  useEffect(() => {
    ;(async () => {
      const [t, l, a] = await Promise.all([
        api.get('/api/alerts/stats/by-time?interval=hour'),
        api.get('/api/alerts/stats/by-label'),
        api.get(`/api/alerts?${new URLSearchParams({ ...search, page: 1, pageSize: 10 }).toString()}`),
      ])
      setByTime(t.data.map(d => ({ time: d._id, count: d.count })))
      setByLabel(l.data.map(d => ({ name: d._id, value: d.count })))
      setRecent(a.data.items)
    })()
  }, [])
  async function apply() {
    const a = await api.get(`/api/alerts?${new URLSearchParams({ ...search, page: 1, pageSize: 10 }).toString()}`)
    setRecent(a.data.items)
  }
  const totalAlerts = byLabel.reduce((s, r) => s + r.value, 0)
  return (
    <Box sx={{ display: 'grid', gap: 2 }}>
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}><Paper sx={{ p: 2 }}><Typography variant="overline">Total Alerts</Typography><Typography variant="h4">{totalAlerts}</Typography></Paper></Grid>
        <Grid item xs={12} md={4}><Paper sx={{ p: 2 }}><Typography variant="overline">Active Threats</Typography><Typography variant="h4">{recent.filter(r => r.label !== 'normal').length}</Typography></Paper></Grid>
        <Grid item xs={12} md={4}><Paper sx={{ p: 2 }}><Typography variant="overline">System</Typography><Typography variant="h6">Healthy</Typography></Paper></Grid>
      </Grid>
      <Grid container spacing={2}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: 320 }}>
            <Typography variant="subtitle1">Attack Trends</Typography>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={byTime}><XAxis dataKey="time" hide /><YAxis /><Tooltip /><Line type="monotone" dataKey="count" stroke="#1976d2" /></LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 320 }}>
            <Typography variant="subtitle1">Attack Types</Typography>
            <ResponsiveContainer width="100%" height="85%">
              <PieChart><Pie data={byLabel} dataKey="value" nameKey="name" outerRadius={90}>
                {byLabel.map((entry, index) => (<Cell key={entry.name} fill={COLORS[index % COLORS.length]} />))}
              </Pie></PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
      <Paper sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
          <TextField size="small" label="Label filter" value={search.label} onChange={(e) => setSearch({ ...search, label: e.target.value })} />
          <TextField size="small" label="Min confidence" type="number" inputProps={{ step: 0.05, min: 0, max: 1 }} value={search.minConfidence} onChange={(e) => setSearch({ ...search, minConfidence: e.target.value })} />
          <Button onClick={apply} variant="contained">Apply</Button>
        </Box>
        {recent.map((a) => (
          <Box key={a._id} sx={{ display: 'flex', alignItems: 'center', gap: 2, py: .5, borderBottom: '1px solid #eee' }}>
            <Typography sx={{ width: 200 }}>{new Date(a.occurredAt).toLocaleString()}</Typography>
            <Typography sx={{ width: 160 }}>{a.label}</Typography>
            <Typography sx={{ width: 100 }}>{a.confidence?.toFixed ? a.confidence.toFixed(3) : a.confidence}</Typography>
            <Chip size="small" label={a.severity || 'low'} color={a.severity === 'critical' ? 'error' : a.severity === 'high' ? 'warning' : a.severity === 'medium' ? 'info' : 'default'} />
          </Box>
        ))}
      </Paper>
    </Box>
  )
}


