/**
 * Studio state: the four modes (persisted per browser), the selected open-data patient, the replay moment,
 * the clinician's daily context, and the data loaded for that moment.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { getV7Patients, getV7Patient, getV7Overview, postV7Predict } from '../services/api'

const DEFAULT_MODES = { data: 'open', alert: 'standard', theme: 'light', layout: 'studio', bg: null }
const HEX = /^#[0-9a-f]{6}$/i
const KEY = 'glucotwin.modes'

const ALLOWED = { data: ['open', 'synthetic'], alert: ['standard', 'safety'], theme: ['light', 'dark'], layout: ['studio', 'classic'] }

/** Saved modes, overridden by URL parameters (e.g. ?theme=dark&data=synthetic) so a view can be linked. */
export function loadModes() {
  let saved = {}
  try { saved = JSON.parse(localStorage.getItem(KEY) || '{}') } catch { /* private window etc. */ }
  const q = new URLSearchParams(window.location.search)
  const fromUrl = Object.fromEntries(Object.keys(ALLOWED).filter(k => ALLOWED[k].includes(q.get(k))).map(k => [k, q.get(k)]))
  if (HEX.test(q.get('bg') || '')) fromUrl.bg = q.get('bg')
  const m = { ...DEFAULT_MODES, ...saved, ...fromUrl }
  if (m.bg != null && !HEX.test(m.bg)) m.bg = null
  return m
}

export function saveModes(m) {
  try { localStorage.setItem(KEY, JSON.stringify(m)) } catch { /* storage unavailable: modes reset on reload */ }
}

export const EMPTY_CONTEXT = { meal: 'none', carbs_g: '', medication: 'none', insulin_u: '', oral_tablets: '1', activity: 'not set', sleep: 'not set', notes: '' }

/** Custom page background: the colour itself, a softer second stop, and readable text for it (WCAG luminance). */
export function backgroundVars(hex) {
  if (!hex || !HEX.test(hex)) return {}
  const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16))
  const lin = c => { const v = c / 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4 }
  const dark = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b) < 0.18
  return {
    '--gt-bg-1': hex,
    '--gt-bg-2': `color-mix(in srgb, ${hex} 70%, ${dark ? '#000000' : '#ffffff'})`,
    '--gt-on-bg': dark ? '#f4fbfc' : '#0f2a35',
    '--gt-on-bg-2': dark ? 'rgba(244,251,252,0.78)' : '#46616d',
    ...(dark ? { '--gt-surface': 'var(--gt-surface-solid)' } : {}),   // glass over a dark colour turns muddy grey
  }
}

/** Only the inputs the models were trained on become a scenario; activity, sleep and notes are kept as context. */
export function contextToScenario(c) {
  const s = {}
  if (c.meal !== 'none' && Number(c.carbs_g) > 0) s.carbs_g = Number(c.carbs_g)
  if (c.medication === 'insulin' && Number(c.insulin_u) > 0) s.insulin_u = Number(c.insulin_u)
  if (c.medication === 'tablet' && Number(c.oral_tablets) > 0) s.oral_tablets = Number(c.oral_tablets)
  return Object.keys(s).length ? s : null
}

/** Risk level from the model's own validated thresholds; the alert mode decides which levels raise an alert. */
export function riskOf(h, alertMode) {
  if (!h) return null
  const p = h.probability
  const level = p >= h.threshold_accuracy_mode ? 'high' : p >= h.threshold_safety_mode ? 'medium' : 'low'
  const alert = alertMode === 'safety' ? level !== 'low' : level === 'high'
  return { level, alert, label: { high: 'High risk', medium: 'Medium risk', low: 'Low risk' }[level] }
}

const Ctx = createContext(null)
export const useStudio = () => useContext(Ctx)

export function StudioProvider({ modes, setModes, children }) {
  const [page, setPageState] = useState(() => window.location.hash.slice(1) || 'patient')
  const setPage = useCallback(p => { setPageState(p); window.history.replaceState(null, '', `#${p}`) }, [])
  const [patients, setPatients] = useState([])
  const [pid, setPid] = useState(null)
  const [detail, setDetail] = useState(null)
  const [bin, setBin] = useState(null)
  const [overview, setOverview] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [predError, setPredError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [context, setContext] = useState(EMPTY_CONTEXT)
  const [applied, setApplied] = useState(null)          // scenario sent with predictions (after "Generate")
  const [error, setError] = useState(null)
  const req = useRef(0)

  const reloadPatients = useCallback((select) => getV7Patients().then(r => {
    setPatients(r.patients)
    setPid(p => select ?? (r.patients.some(x => x.patient_id === p) ? p : r.patients[0]?.patient_id ?? null))
  }).catch(e => setError(e?.response?.data?.detail || e.message)), [])

  useEffect(() => { if (modes.data === 'open') reloadPatients() }, [modes.data, reloadPatients])

  useEffect(() => {
    if (!pid) return
    setDetail(null); setPrediction(null); setOverview(null); setApplied(null); setContext(EMPTY_CONTEXT)
    getV7Patient(pid).then(d => { setDetail(d); setBin(d.default_bin) })
      .catch(e => setError(e?.response?.data?.detail || e.message))
  }, [pid])

  const refresh = useCallback((scenario) => {
    if (!pid || bin == null) return Promise.resolve()
    const id = ++req.current
    setLoading(true)
    return Promise.all([
      getV7Overview(pid, bin),
      postV7Predict(pid, bin, scenario).then(r => ({ ok: r })).catch(e => ({ err: e?.response?.data?.detail || e.message })),
    ]).then(([ov, pr]) => {
      if (id !== req.current) return
      setOverview(ov)
      setPrediction(pr.ok ?? null)
      setPredError(pr.err ?? null)
    }).catch(e => id === req.current && setError(e?.response?.data?.detail || e.message))
      .finally(() => id === req.current && setLoading(false))
  }, [pid, bin])

  useEffect(() => {
    const t = setTimeout(() => refresh(applied), 120)
    return () => clearTimeout(t)
  }, [refresh, applied])

  const generate = useCallback(() => {
    const sc = contextToScenario(context)
    setApplied(sc)
    if (JSON.stringify(sc) === JSON.stringify(applied)) refresh(sc)
  }, [context, applied, refresh])

  const value = useMemo(() => ({
    modes, setModes, page, setPage, patients, reloadPatients, pid, setPid, detail, bin, setBin, overview, prediction, predError,
    loading, context, setContext, applied, generate, error,
  }), [modes, setModes, page, setPage, patients, reloadPatients, pid, detail, bin, overview, prediction, predError, loading, context, applied, generate, error])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
