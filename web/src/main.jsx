import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AppShell from './ui/AppShell.jsx'
import LoginPage from './ui/pages/LoginPage.jsx'
import SignupPage from './ui/pages/SignupPage.jsx'
import DashboardPage from './ui/pages/DashboardPage.jsx'
import AlertsPage from './ui/pages/AlertsPage.jsx'
import UsersPage from './ui/pages/UsersPage.jsx'
import FeatureImportancePage from './ui/pages/FeatureImportancePage.jsx'
import SettingsPage from './ui/pages/SettingsPage.jsx'
import HelpPage from './ui/pages/HelpPage.jsx'
import { AuthProvider, useAuth } from './ui/state/auth.jsx'

function Protected({ children, role }) {
  const { token, user } = useAuth()
  if (!token) return <Navigate to="/login" replace />
  if (role && user?.role !== role) return <Navigate to="/" replace />
  return children
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<Protected role="admin"><SignupPage /></Protected>} />
          <Route path="/" element={<Protected><AppShell /></Protected>}>
            <Route index element={<DashboardPage />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="users" element={<Protected role="admin"><UsersPage /></Protected>} />
            <Route path="feature-importance" element={<FeatureImportancePage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="help" element={<HelpPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}

createRoot(document.getElementById('root')).render(<App />)


