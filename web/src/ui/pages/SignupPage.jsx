import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Container, TextField, Typography, Paper, Box, MenuItem } from '@mui/material'
import { useFormik } from 'formik'
import * as Yup from 'yup'
import { useAuth } from '../state/auth.jsx'

export default function SignupPage() {
  const { api } = useAuth()
  const nav = useNavigate()
  const form = useFormik({
    initialValues: { email: '', password: '', role: 'analyst' },
    validationSchema: Yup.object({
      email: Yup.string().email().required(),
      password: Yup.string().min(6).required(),
      role: Yup.string().oneOf(['admin', 'analyst']).required(),
    }),
    onSubmit: async (values, { setSubmitting, setStatus }) => {
      try {
        await api.post('/api/auth/register', values)
        nav('/users')
      } catch (e) {
        setStatus(e?.response?.data?.error || 'Failed to create user')
      } finally {
        setSubmitting(false)
      }
    }
  })
  return (
    <Container maxWidth="sm" sx={{ mt: 6 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Create User</Typography>
        <Box component="form" onSubmit={form.handleSubmit} sx={{ display: 'grid', gap: 2 }}>
          <TextField label="Email" name="email" value={form.values.email} onChange={form.handleChange} error={!!form.errors.email && form.touched.email} helperText={form.touched.email && form.errors.email} />
          <TextField label="Password" type="password" name="password" value={form.values.password} onChange={form.handleChange} error={!!form.errors.password && form.touched.password} helperText={form.touched.password && form.errors.password} />
          <TextField select label="Role" name="role" value={form.values.role} onChange={form.handleChange}>
            <MenuItem value="analyst">analyst</MenuItem>
            <MenuItem value="admin">admin</MenuItem>
          </TextField>
          {form.status && <Typography color="error">{form.status}</Typography>}
          <Button type="submit" variant="contained" disabled={form.isSubmitting}>Create</Button>
        </Box>
      </Paper>
    </Container>
  )
}


