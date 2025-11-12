import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import axios from 'axios'

const AuthCtx = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken] = useState(localStorage.getItem('token') || '')
  const [user, setUser] = useState(null)
  const api = useMemo(() => axios.create({ headers: token ? { Authorization: `Bearer ${token}` } : {} }), [token])
  useEffect(() => {
    if (!token) return setUser(null)
    api.get('/api/auth/me').then(r => setUser(r.data.user)).catch(() => setUser(null))
  }, [token])
  const value = { token, setToken: (t) => { localStorage.setItem('token', t); setToken(t) }, user, api }
  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>
}

export function useAuth() {
  return useContext(AuthCtx)
}


