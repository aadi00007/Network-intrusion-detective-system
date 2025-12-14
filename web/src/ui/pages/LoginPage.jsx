import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Container, TextField, Typography, Paper, Box } from '@mui/material'
import { useFormik } from 'formik'
import * as Yup from 'yup'
import axios from 'axios'
import { useAuth } from '../state/auth.jsx'

export default function LoginPage() {
  const { setToken } = useAuth()
  const nav = useNavigate()
  const form = useFormik({
    initialValues: { email: 'admin@example.com', password: 'admin123' },
    validationSchema: Yup.object({
      email: Yup.string().email().required(),
      password: Yup.string().min(6).required(),
    }),
    onSubmit: async (values, { setSubmitting, setStatus }) => {
      try {
        const res = await axios.post('/api/auth/login', values)
        setToken(res.data.token)
        nav('/')
      } catch (e) {
        setStatus('Invalid credentials')
      } finally {
        setSubmitting(false)
      }
    }
  })
  return (
    <Container maxWidth="xs" sx={{ mt: 10 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Sign in</Typography>
        <Box component="form" onSubmit={form.handleSubmit} sx={{ display: 'grid', gap: 2 }}>
          <TextField label="Email" name="email" value={form.values.email} onChange={form.handleChange} error={!!form.errors.email && form.touched.email} helperText={form.touched.email && form.errors.email} />
          <TextField label="Password" type="password" name="password" value={form.values.password} onChange={form.handleChange} error={!!form.errors.password && form.touched.password} helperText={form.touched.password && form.errors.password} />
          {form.status && <Typography color="error">{form.status}</Typography>}
          <Button type="submit" variant="contained" disabled={form.isSubmitting}>Login</Button>
        </Box>
      </Paper>
    </Container>
  )
}


