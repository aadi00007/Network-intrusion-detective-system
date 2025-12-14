import React, { useEffect, useState } from 'react'
import { useAuth } from '../state/auth.jsx'
import { Box, Button, Paper, Table, TableBody, TableCell, TableHead, TableRow, TextField } from '@mui/material'

export default function AlertsPage() {
  const { api } = useAuth()
  const [rows, setRows] = useState([])
  const [q, setQ] = useState({ label: '', minConfidence: 0.7, page: 1, pageSize: 50 })
  
  async function load() {
    const res = await api.get(`/api/alerts?${new URLSearchParams(q).toString()}`)
    setRows(res.data.items)
  }
  
  useEffect(() => { load() }, [])
  
  // Set up SSE for real-time updates
  useEffect(() => {
    const eventSource = new EventSource('/api/events/stream')
    
    eventSource.onmessage = (event) => {
      if (event.data === 'connected') return
      try {
        const alert = JSON.parse(event.data)
        // Add new alert to the top of the list
        setRows(prev => [alert, ...prev].slice(0, q.pageSize))
        // Optionally reload to get updated data
        load()
      } catch (e) {
        console.error('Error parsing SSE event:', e)
      }
    }
    
    eventSource.onerror = (e) => {
      console.error('SSE error:', e)
    }
    
    return () => {
      eventSource.close()
    }
  }, [q.pageSize])
  return (
    <Box sx={{ display: 'grid', gap: 2 }}>
      <Paper sx={{ p: 2, display: 'flex', gap: 1 }}>
        <TextField size="small" label="Label" value={q.label} onChange={e => setQ({ ...q, label: e.target.value })} />
        <TextField size="small" label="Min confidence" type="number" value={q.minConfidence} onChange={e => setQ({ ...q, minConfidence: e.target.value })} />
        <Button variant="contained" onClick={load}>Apply</Button>
        <Button href="/api/alerts/export/csv" target="_blank">Export CSV</Button>
      </Paper>
      <Paper>
        <Table size="small">
          <TableHead><TableRow><TableCell>Time</TableCell><TableCell>Label</TableCell><TableCell>Confidence</TableCell><TableCell>Severity</TableCell></TableRow></TableHead>
          <TableBody>
            {rows.map(r => (
              <TableRow key={r._id}>
                <TableCell>{new Date(r.occurredAt).toLocaleString()}</TableCell>
                <TableCell>{r.label}</TableCell>
                <TableCell>{r.confidence?.toFixed ? r.confidence.toFixed(3) : r.confidence}</TableCell>
                <TableCell>{r.severity}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
    </Box>
  )
}


