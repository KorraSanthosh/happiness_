/**
 * API service layer — all calls go through the Vite proxy to FastAPI backend.
 */
import axios from 'axios'

const BASE = '/api'

const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
})

export const getHealth = () => api.get('/health').then(r => r.data)

export const getStats = () => api.get('/stats').then(r => r.data)

export const getPatients = (params = {}) =>
  api.get('/patients', { params }).then(r => r.data)

export const getPatient = (id) => api.get(`/patients/${id}`).then(r => r.data)

export const getReadings = (id, limit = 144) =>
  api.get(`/patients/${id}/readings`, { params: { limit } }).then(r => r.data)

export const getDigitalTwin = (id) =>
  api.get(`/patients/${id}/twin`).then(r => r.data)

export const postPredict = (id) =>
  api.post(`/predict/${id}`).then(r => r.data)

// ── Real-data Digital Twin (v7) ──────────────────────────────
export const getV7Patients = () => api.get('/v7/patients').then(r => r.data)

export const getV7Patient = (id) => api.get(`/v7/patients/${id}`).then(r => r.data)

export const getV7Stream = (id, bin, hours = 12, future = true) =>
  api.get(`/v7/patients/${id}/stream`, { params: { bin, hours, future } }).then(r => r.data)

export const postV7Predict = (id, bin, scenario = null) =>
  api.post(`/v7/predict/${id}`, scenario, { params: { bin } }).then(r => r.data)

export const getV7Overview = (id, bin) => api.get(`/v7/patients/${id}/overview`, { params: { bin } }).then(r => r.data)

export const getV7Trends = (id, bin) => api.get(`/v7/patients/${id}/trends`, { params: { bin } }).then(r => r.data)

export const uploadV7 = (form) => api.post('/v7/upload', form, { timeout: 60000 }).then(r => r.data)

export const deleteV7Upload = (id) => api.delete(`/v7/upload/${id}`).then(r => r.data)

export const V7_EXAMPLE_CSV = `${BASE}/v7/upload/example.csv`

export const getV7ModelCard = () => api.get('/v7/model-card').then(r => r.data)

export default api
