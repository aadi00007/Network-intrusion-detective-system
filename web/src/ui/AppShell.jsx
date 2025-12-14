import React from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { AppBar, Toolbar, Typography, Drawer, List, ListItemButton, ListItemText, Box } from '@mui/material'

export default function AppShell() {
  const { pathname } = useLocation()
  const menu = [
    { to: '/', label: 'Dashboard' },
    { to: '/alerts', label: 'Alerts' },
    { to: '/feature-importance', label: 'Feature Importance' },
    { to: '/users', label: 'Users' },
    { to: '/settings', label: 'Settings' },
    { to: '/help', label: 'Help' },
  ]
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '240px 1fr', gridTemplateRows: '64px 1fr', height: '100vh' }}>
      <AppBar position="static" sx={{ gridColumn: '1 / span 2' }}>
        <Toolbar>
          <Typography variant="h6">IDS Dashboard</Typography>
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


