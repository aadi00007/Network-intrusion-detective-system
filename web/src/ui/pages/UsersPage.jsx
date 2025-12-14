import React, { useEffect, useState } from 'react'
import { useAuth } from '../state/auth.jsx'
import { Box, Button, Paper, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material'
import { Link } from 'react-router-dom'

export default function UsersPage() {
  const { api } = useAuth()
  const [rows, setRows] = useState([])
  async function load() {
    const res = await api.get('/api/auth/users')
    setRows(res.data)
  }
  async function remove(id) {
    await api.delete(`/api/auth/users/${id}`)
    await load()
  }
  useEffect(() => { load() }, [])
  return (
    <Box sx={{ display: 'grid', gap: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">Users</Typography>
        <Button component={Link} to="/signup" variant="contained">Add User</Button>
      </Box>
      <Paper>
        <Table size="small">
          <TableHead><TableRow><TableCell>Email</TableCell><TableCell>Role</TableCell><TableCell>Actions</TableCell></TableRow></TableHead>
          <TableBody>
            {rows.map(r => (
              <TableRow key={r.id}>
                <TableCell>{r.email}</TableCell>
                <TableCell>{r.role}</TableCell>
                <TableCell><Button color="error" onClick={() => remove(r.id)}>Delete</Button></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
    </Box>
  )
}


