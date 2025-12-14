import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import AppShell from './ui/AppShell.jsx'
import DashboardPage from './ui/pages/DashboardPage.jsx'
import AlertsPage from './ui/pages/AlertsPage.jsx'
import UsersPage from './ui/pages/UsersPage.jsx'
import FeatureImportancePage from './ui/pages/FeatureImportancePage.jsx'
import SettingsPage from './ui/pages/SettingsPage.jsx'
import HelpPage from './ui/pages/HelpPage.jsx'
import { AuthProvider } from './ui/state/auth.jsx'

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppShell />}>
            <Route index element={<DashboardPage />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="users" element={<UsersPage />} />
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


