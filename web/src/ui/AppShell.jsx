import React from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { AppBar, Toolbar, Typography, Drawer, List, ListItemButton, ListItemText, Box, Button } from '@mui/material'
import { useAuth } from './state/auth.jsx'

export default function AppShell() {
  const { user, setToken } = useAuth()
  const { pathname } = useLocation()
  const menu = [
    { to: '/', label: 'Dashboard' },
    { to: '/alerts', label: 'Alerts' },
    { to: '/feature-importance', label: 'Feature Importance' },
    ...(user?.role === 'admin' ? [{ to: '/users', label: 'Users' }] : []),
    { to: '/settings', label: 'Settings' },
    { to: '/help', label: 'Help' },
  ]
  function logout() {
    localStorage.removeItem('token')
    setToken('')
    window.location.href = '/login'
  }
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '240px 1fr', gridTemplateRows: '64px 1fr', height: '100vh' }}>
      <AppBar position="static" sx={{ gridColumn: '1 / span 2' }}>
        <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="h6">IDS Dashboard</Typography>
          <Button color="inherit" onClick={logout}>Logout</Button>
        </Toolbar>
      </AppBar>
      <Drawer variant="permanent" open sx={{ position: 'relative' }}>
        <List sx={{ width: 240, mt: 8 }}>
          {menu.map((m) => (
            <ListItemButton key={m.to} component={Link} to={m.to} selected={pathname === m.to}>
              <ListItemText primary={m.label} />
            </ListItemButton>
          ))}
        </List>
      </Drawer>
      <Box sx={{ overflow: 'auto', p: 2 }}>
        <Outlet />
      </Box>
    </Box>
  )
}


