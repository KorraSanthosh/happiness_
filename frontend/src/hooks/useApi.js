import { useState, useEffect, useCallback } from 'react'
import { getPatients, getDigitalTwin, postPredict, getStats, getReadings } from '../services/api'

export function useStats() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    getStats()
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return { stats, loading, error }
}

export function usePatients(search = '') {
  const [patients, setPatients] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    const params = { limit: 120, offset: 0 }
    if (search) params.search = search
    getPatients(params)
      .then(data => {
        setPatients(data.patients || [])
        setTotal(data.total || 0)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [search])

  return { patients, total, loading, error }
}

export function useDigitalTwin(patientId) {
  const [twin, setTwin] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!patientId) { setTwin(null); return }
    setLoading(true)
    setError(null)
    getDigitalTwin(patientId)
      .then(setTwin)
      .catch(e => setError(e.response?.data?.detail || e.message))
      .finally(() => setLoading(false))
  }, [patientId])

  return { twin, loading, error }
}

export function useReadings(patientId, limit = 144) {
  const [readings, setReadings] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!patientId) { setReadings([]); return }
    setLoading(true)
    getReadings(patientId, limit)
      .then(data => setReadings(data.readings || []))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [patientId, limit])

  return { readings, loading }
}

export function usePrediction(patientId) {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runPrediction = useCallback(async () => {
    if (!patientId) return
    setLoading(true)
    setError(null)
    try {
      const result = await postPredict(patientId)
      setPrediction(result)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }, [patientId])

  // Reset when patient changes
  useEffect(() => {
    setPrediction(null)
    setError(null)
  }, [patientId])

  return { prediction, loading, error, runPrediction }
}
