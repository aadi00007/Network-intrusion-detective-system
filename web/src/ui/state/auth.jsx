import React, { createContext, useContext, useMemo } from 'react'
import axios from 'axios'

const AuthCtx = createContext(null)

export function AuthProvider({ children }) {
  // No authentication required - always provide public access
  const user = { id: 'public', email: 'public@local', role: 'admin' }
  const api = useMemo(() => axios.create({
    // No baseURL - routes already include /api prefix
    timeout: 120000, // 2 minutes default timeout for all requests
  }), [])
  const value = { token: '', setToken: () => {}, user, api }
  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>
}

export function useAuth() {
  return useContext(AuthCtx)
}


