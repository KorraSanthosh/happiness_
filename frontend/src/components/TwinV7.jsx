/**
 * Real-data Digital Twin (v7) — doctor dashboard.
 * Replays a real (open-licence) patient record: pick a moment, and the backend runs the live pipeline
 * (CGM / insulin / meal stream + EHR → features → 30 min / 60 min / 2 h models) for that moment.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ReferenceArea,
} from 'recharts'
import {
  Activity, AlertTriangle, Brain, ClipboardList, Dna, Pause, Play, SkipBack, SkipForward,
  ChevronLeft, ChevronRight, Target, CheckCircle2, XCircle, BarChart3,
} from 'lucide-react'
import { getV7Patients, getV7Patient, getV7Stream, postV7Predict, getV7ModelCard } from '../services/api'
import { getAvatarColor } from '../utils'

const LEVEL_STYLE = {
  'spike likely': { color: 'var(--risk-high)', bg: 'rgba(239,68,68,0.12)', label: 'Spike likely' },
  watch:          { color: 'var(--risk-mod)',  bg: 'rgba(245,158,11,0.12)', label: 'Watch' },
  unlikely:       { color: 'var(--risk-low)',  bg: 'rgba(34,197,94,0.12)',  label: 'Unlikely' },
}
const HORIZON_LABEL = { '30m': '30 minutes', '60m': '1 hour', '2h': '2 hours' }
const sectionTitle = { fontSize: 'var(--fs-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 8 }
const pct = v => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

/* ── Sidebar ─────────────────────────────────────────────── */
function V7Sidebar({ patients, selectedId, onSelect }) {
  const [search, setSearch] = useState('')
  const shown = patients.filter(p => `${p.dataset} ${p.display_id} ${p.diabetes_type}`.toLowerCase().includes(search.toLowerCase()))
  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-row">
          <div className="logo-icon"><Dna /></div>
          <div><div className="logo-title">GlucoTwin</div></div>
        </div>
        <div className="logo-subtitle">Real-data Digital Twin · v7</div>
      </div>
      <div className="search-container">
        <div className="search-wrapper">
          <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
          </svg>
          <input className="search-input" placeholder="Search dataset / ID / T1D / T2D…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
      </div>
      <div className="sidebar-section-header">{shown.length} real patients · never seen by the model</div>
      <div className="patient-list">
        {shown.map(p => {
          const [bg, fg] = getAvatarColor(p.display_id)
          return (
            <div key={p.patient_id} className={`patient-item${selectedId === p.patient_id ? ' active' : ''}`} onClick={() => onSelect(p.patient_id)}>
              <div className="patient-avatar" style={{ background: bg, color: fg, fontSize: 10 }}>{p.diabetes_type}</div>
              <div className="patient-meta">
                <div className="patient-name">Patient #{p.display_id}</div>
                <div className="patient-id">
                  {p.dataset}{p.age ? ` · ${Math.round(p.age)}y ${p.sex ?? ''}` : ''}{p.hba1c ? ` · HbA1c ${p.hba1c.toFixed(1)}%` : ''}
                </div>
              </div>
              {p.has_ehr && <span className="risk-badge low" title="Real clinical record available">EHR</span>}
            </div>
          )
        })}
      </div>
      <div className="sidebar-footer">
        <div className="sidebar-footer-count">Open data (CC BY 4.0) · research prototype</div>
      </div>
    </div>
  )
}

/* ── Glucose / insulin / meal chart with the prediction window ── */
function StreamChart({ stream }) {
  const pts = stream?.points ?? []
  const nowIdx = pts.findIndex(p => p.bin === stream?.now_bin)
  const data = pts.map((p, i) => ({
    time: p.time,
    past: !p.future ? p.glucose : null,
    future: p.future || i === nowIdx ? p.glucose : null,
    insulin: p.insulin ? Math.round(p.insulin * 10) / 10 : 0,
    carbs: p.carbs || 0,
  }))
  const ticks = data.filter((_, i) => i % 8 === 0).map(d => d.time)
  const last = data[data.length - 1]?.time
  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 24, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis dataKey="time" ticks={ticks} tick={{ fill: '#4b5c73', fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis yAxisId="g" domain={[40, 'auto']} tick={{ fill: '#4b5c73', fontSize: 10 }} axisLine={false} tickLine={false} width={34} />
          <YAxis yAxisId="e" orientation="right" tick={{ fill: '#4b5c73', fontSize: 10 }} axisLine={false} tickLine={false} width={30} />
          <Tooltip contentStyle={{ background: 'rgba(8,14,26,0.95)', border: '1px solid rgba(96,165,250,0.3)', borderRadius: 10, fontSize: 12 }} />
          {nowIdx >= 0 && last && (
            <ReferenceArea yAxisId="g" x1={data[nowIdx].time} x2={last} fill="rgba(167,139,250,0.07)"
              label={{ value: 'prediction window (replay: what really happened)', fill: '#a78bfa', fontSize: 9, position: 'insideTop' }} />
          )}
          <ReferenceLine yAxisId="g" y={180} stroke="rgba(239,68,68,0.6)" strokeDasharray="5 3" label={{ value: '180', fill: '#ef4444', fontSize: 9, position: 'right' }} />
          <ReferenceLine yAxisId="g" y={70} stroke="rgba(245,158,11,0.45)" strokeDasharray="5 3" />
          {nowIdx >= 0 && <ReferenceLine yAxisId="g" x={data[nowIdx].time} stroke="#a78bfa" strokeWidth={1.5} label={{ value: 'NOW', fill: '#a78bfa', fontSize: 10, position: 'top' }} />}
          <Bar yAxisId="e" dataKey="insulin" name="Insulin (U / 15 min)" fill="rgba(6,182,212,0.35)" barSize={4} isAnimationActive={false} />
          <Bar yAxisId="e" dataKey="carbs" name="Carbs (g)" fill="rgba(245,158,11,0.45)" barSize={4} isAnimationActive={false} />
          <Line yAxisId="g" dataKey="past" name="Glucose (mg/dL)" stroke="#3b82f6" strokeWidth={2.5} dot={false} connectNulls isAnimationActive={false} />
          <Line yAxisId="g" dataKey="future" name="Actual next 2 h (mg/dL)" stroke="#a78bfa" strokeWidth={2} strokeDasharray="4 3" dot={false} connectNulls isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ── One horizon's prediction ────────────────────────────── */
function HorizonCard({ name, h }) {
  const st = LEVEL_STYLE[h.level] ?? LEVEL_STYLE.unlikely
  const predictedSpike = h.level === 'spike likely'
  const correct = h.actual ? h.actual.spiked === predictedSpike : null
  return (
    <div style={{ padding: '12px 14px', borderRadius: 'var(--r-lg)', border: '1px solid var(--border-glass)', background: st.bg, marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: 'var(--fs-sm)', fontWeight: 600, color: 'var(--text-white)' }}>Next {HORIZON_LABEL[name]}</div>
        <span style={{ fontSize: 11, fontWeight: 700, color: st.color, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{st.label}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 4 }}>
        <span style={{ fontSize: '1.6rem', fontWeight: 800, color: st.color }}>{Math.round(h.probability * 100)}%</span>
        <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>chance glucose goes above 180 mg/dL</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)', marginTop: 6, position: 'relative' }}>
        <div style={{ width: `${h.probability * 100}%`, height: '100%', borderRadius: 3, background: st.color }} />
        <div title="safety threshold" style={{ position: 'absolute', left: `${h.threshold_safety_mode * 100}%`, top: -3, height: 12, width: 2, background: 'var(--risk-mod)' }} />
        <div title="accuracy threshold" style={{ position: 'absolute', left: `${h.threshold_accuracy_mode * 100}%`, top: -3, height: 12, width: 2, background: 'var(--risk-high)' }} />
      </div>
      {h.early_warning && (
        <div title={h.early_warning.rule} style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, fontSize: 'var(--fs-xs)',
          color: h.early_warning.active ? 'var(--risk-high)' : 'var(--text-muted)', fontWeight: h.early_warning.active ? 700 : 400 }}>
          <AlertTriangle style={{ width: 13, height: 13 }} />
          {h.early_warning.active ? 'Early warning: act now, glucose is still in range'
            : 'No early warning (fires only while glucose ≤ 180, at most once an hour)'}
        </div>
      )}
      {h.model_version && <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>model {h.model_version}</div>}
      {h.actual && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)' }}>
          {correct ? <CheckCircle2 style={{ width: 13, height: 13, color: 'var(--risk-low)' }} /> : <XCircle style={{ width: 13, height: 13, color: 'var(--risk-high)' }} />}
          Replay: actually {h.actual.spiked ? 'spiked' : 'stayed ≤ 180'} (max {Math.round(h.actual.max_glucose)} mg/dL)
        </div>
      )}
    </div>
  )
}

/* ── EHR panel ───────────────────────────────────────────── */
function EhrPanel({ detail }) {
  const e = detail?.ehr
  if (!e) return null
  return (
    <div className="fade-in">
      <div className="profile-tags" style={{ marginBottom: 14 }}>
        <span className="tag">🩸 {detail.diabetes_type === 'T2D' ? 'Type 2 diabetes' : 'Type 1 diabetes'}</span>
        <span className="tag">{detail.dataset}</span>
        {e.sex && <span className="tag">{e.sex}</span>}
        {e.complications != null && <span className="tag">{e.complications} diabetic complication{e.complications === 1 ? '' : 's'}</span>}
      </div>
      {!e.available && (
        <div className="explanation-box" style={{ marginBottom: 14 }}>
          This dataset has no clinical summary. The static (EHR) input for this patient is the diabetes type only;
          the twin relies on the live CGM / insulin / meal stream.
        </div>
      )}
      {e.labs?.length > 0 && (
        <>
          <div style={sectionTitle}>Lab results & history (real clinical record)</div>
          <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-glass)', borderRadius: 'var(--r-lg)', padding: '4px 16px', marginBottom: 14 }}>
            {e.labs.map(l => (
              <div key={l.name} className="lab-item">
                <span className="lab-name">{l.name}</span>
                <div className="lab-value"><span className="lab-val">{l.value}</span><span className="lab-unit">{l.unit}</span></div>
              </div>
            ))}
          </div>
        </>
      )}
      {e.conditions_and_drugs?.length > 0 && (
        <>
          <div style={sectionTitle}>Conditions & medicines</div>
          <div className="profile-tags">{e.conditions_and_drugs.map(c => <span key={c} className="tag">{c}</span>)}</div>
        </>
      )}
    </div>
  )
}

/* ── Model card ──────────────────────────────────────────── */
function ModelCardV7() {
  const [card, setCard] = useState(null)
  useEffect(() => { getV7ModelCard().then(setCard).catch(() => setCard({})) }, [])
  const s = card?.summary
  if (!s?.horizons) return <div className="explanation-box">Model summary not available yet.</div>
  const row = (label, cells, bold) => (
    <div style={{ display: 'grid', gridTemplateColumns: '1.4fr repeat(5, 1fr)', gap: 6, padding: '7px 10px', fontSize: 'var(--fs-xs)', fontWeight: bold ? 700 : 400, color: bold ? 'var(--text-white)' : 'var(--text-secondary)', borderBottom: '1px solid var(--border-glass)' }}>
      <div>{label}</div>{cells.map((c, i) => <div key={i} style={{ fontFamily: 'var(--font-mono)', textAlign: 'center' }}>{c}</div>)}
    </div>
  )
  return (
    <div className="fade-in">
      <div style={sectionTitle}>Tested on people the model never saw (7-fold, person-held-out)</div>
      {row('Horizon', ['Accuracy', 'Spikes caught', 'AUC', 'Simple rule', 'Ohio (external)'], true)}
      {Object.entries(s.horizons).map(([h, r]) => row(`Next ${HORIZON_LABEL[h]}`, [
        pct(r.accuracy_mode.all_open.accuracy), pct(r.accuracy_mode.all_open.recall),
        r.accuracy_mode.all_open.roc_auc?.toFixed(3), pct(r.accuracy_mode.all_open.rule_accuracy), pct(r.ohio_external.accuracy),
      ]))}
      <div style={{ ...sectionTitle, marginTop: 18 }}>Does fusing EHR + sensor data help? (2 h, patients with a clinical record)</div>
      {row('Input', ['Accuracy', 'Spikes caught', 'AUC', '', ''], true)}
      {Object.entries(s.fusion_ablation).map(([k, r]) => row(k, [pct(r.accuracy), pct(r.recall), r.roc_auc?.toFixed(3), '', '']))}
      <div style={{ ...sectionTitle, marginTop: 18 }}>2 h accuracy by dataset</div>
      {row('Dataset', ['Accuracy', 'Spikes caught', 'AUC', 'Simple rule', 'People'], true)}
      {Object.entries(s.horizons['2h'].accuracy_mode).filter(([k]) => k !== 'all_open').map(([k, r]) => row(k, [
        pct(r.accuracy), pct(r.recall), r.roc_auc?.toFixed(3), pct(r.rule_accuracy), r.people,
      ]))}
      {s.error_analysis && (() => {
        const e = s.error_analysis, a = e.label_reliability_2h.all_open, ep = e.serious_spikes_2h.episode_level
        const f = e.cgm_vs_fingerstick.overall
        return (
          <>
            <div style={{ ...sectionTitle, marginTop: 18 }}>Where the remaining error comes from (2 h, verified)</div>
            {row('Finding', ['Value', 'Based on', '', '', ''], true)}
            {row('Errors where the actual peak was 162–198 mg/dL', [pct(a.uncertain_share_of_errors), `${a.errors.toLocaleString()} errors`, '', '', ''])}
            {row('Accuracy: clear outcome / near 180', [`${pct(a.accuracy_clear)} / ${pct(a.accuracy_uncertain)}`, `${a.rows.toLocaleString()} moments`, '', '', ''])}
            {row('Serious spike episodes with no warning 30–120 min ahead', [pct(ep.miss_rate), `${ep.missed.toLocaleString()} of ${ep.episodes.toLocaleString()}`, '', '', ''])}
            {row('Calibration error (ECE) 30 min / 1 h / 2 h', [['30m', '60m', '2h'].map(h => (100 * e.calibration[h].ece).toFixed(2)).join(' / ') + ' %', 'held-out', '', '', ''])}
            {row('CGM vs fingerstick, MARD (Shanghai only)', [`${f.mard_pct} %`, `${f.pairs.toLocaleString()} pairs, ${f.people} people`, '', '', ''])}
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, lineHeight: 1.5 }}>
              Serious = {e.serious_spikes_2h.definition}. "Near 180" is defined using the actual outcome (an error breakdown, not something the model can see).
              Fingerstick pairs share a 15-min CGM row; meters have their own error. Full method: docs/model_investigation.md.
            </div>
          </>
        )
      })()}
      {s.v9 && (() => {
        const t7 = s.v9.test.v7_config, t9 = s.v9.test.v9, arms = s.v9.alert_policy.arms
        const v7a = arms['baseline + v7 policy'], v9a = arms['candidate + v9 policy']
        const warned = a => `${a.serious_warned.toLocaleString()} of ${a.serious_episodes.toLocaleString()} (${pct(a.serious_recall)})`
        return (
          <>
            <div style={{ ...sectionTitle, marginTop: 18 }}>v9 update of the 2 h model: tested once on {s.v9.test_people} locked-away people</div>
            {row('2 h, locked test set', ['v7', 'v9 (served)', '', '', ''], true)}
            {row('Accuracy', [pct(t7.accuracy), pct(t9.accuracy), '', '', ''])}
            {row('AUC (ranking)', [t7.roc_auc.toFixed(4), t9.roc_auc.toFixed(4), '', '', ''])}
            {row('Ohio external accuracy', [pct(t7.ohio_accuracy), pct(t9.ohio_accuracy), '', '', ''])}
            {row('Serious episodes warned 30–120 min ahead', [warned(v7a), warned(v9a), '', '', ''])}
            {row('False alerts per patient-day', [v7a.false_alerts_per_patient_day.toFixed(2), v9a.false_alerts_per_patient_day.toFixed(2), '', '', ''])}
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, lineHeight: 1.5 }}>
              v9 adds longer personal history, recorded meals & tablets and multi-window trends, and warns only while glucose is
              still ≤ 180 (at most once an hour). Ranking improves slightly (significant); the accuracy change is within noise.
              The 30 min / 1 h models are unchanged (v7). Everything here was chosen on separate development people only.
            </div>
          </>
        )
      })()}
      <div className="explanation-box" style={{ marginTop: 14 }}>
        Training data: {Object.values(s.datasets).slice(0, -1).join(' · ')}. OhioT1DM is used only as an external test.
        Research prototype — not a medical device.
      </div>
    </div>
  )
}

/* ── Patient view ────────────────────────────────────────── */
function V7PatientView({ pid }) {
  const [detail, setDetail] = useState(null)
  const [bin, setBin] = useState(null)
  const [stream, setStream] = useState(null)
  const [pred, setPred] = useState(null)
  const [error, setError] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [tab, setTab] = useState('twin')
  const timer = useRef(null)

  useEffect(() => {
    setDetail(null); setStream(null); setPred(null); setError(null); setPlaying(false)
    getV7Patient(pid).then(d => { setDetail(d); setBin(d.default_bin) }).catch(e => setError(e.response?.data?.detail || e.message))
  }, [pid])

  useEffect(() => {
    if (bin == null) return
    const t = setTimeout(() => {
      Promise.all([getV7Stream(pid, bin), postV7Predict(pid, bin)])
        .then(([s, p]) => { setStream(s); setPred(p); setError(null) })
        .catch(e => { setPred(null); setError(e.response?.data?.detail || e.message) })
    }, 200)
    return () => clearTimeout(t)
  }, [pid, bin])

  const maxBin = detail ? detail.n_bins - 9 : 0
  useEffect(() => {
    if (!playing) { clearInterval(timer.current); return }
    timer.current = setInterval(() => setBin(b => (b < maxBin ? b + 1 : b)), 1500)
    return () => clearInterval(timer.current)
  }, [playing, maxBin])

  const move = d => setBin(b => Math.min(maxBin, Math.max(detail?.min_bin ?? 4, b + d)))
  if (error && !detail) return <div className="content-body"><div className="error-box"><AlertTriangle style={{ width: 16, height: 16 }} /> {error}</div></div>
  if (!detail) return <div className="content-body"><div className="skeleton" style={{ height: 300, borderRadius: 12 }} /></div>

  const btn = { display: 'inline-flex', alignItems: 'center', gap: 4, padding: '5px 9px', borderRadius: 8, border: '1px solid var(--border-glass)', background: 'rgba(255,255,255,0.03)', color: 'var(--text-secondary)', fontSize: 11, cursor: 'pointer' }
  return (
    <div className="content-body fade-in">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14, padding: '10px 16px', background: 'rgba(59,130,246,0.06)', border: '1px solid rgba(59,130,246,0.14)', borderRadius: 'var(--r-lg)', flexWrap: 'wrap' }}>
        <div style={{ fontWeight: 600, color: 'var(--text-white)', fontSize: 'var(--fs-sm)' }} title={detail.dataset_citation}>{detail.dataset} · patient #{detail.display_id}</div>
        <span className="tag">{detail.diabetes_type}</span>
        <div style={{ marginLeft: 'auto', fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>{stream?.now ?? ''} · live pipeline {pred ? `${pred.inference_ms} ms` : ''}</div>
      </div>

      {/* replay controls */}
      <div className="card" style={{ marginBottom: 14, padding: '12px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <button style={btn} onClick={() => setPlaying(p => !p)}>{playing ? <Pause style={{ width: 12 }} /> : <Play style={{ width: 12 }} />}{playing ? 'Pause' : 'Play'}</button>
          <button style={btn} onClick={() => move(-4)}><SkipBack style={{ width: 12 }} />1 h</button>
          <button style={btn} onClick={() => move(-1)}><ChevronLeft style={{ width: 12 }} />15 min</button>
          <button style={btn} onClick={() => move(1)}>15 min<ChevronRight style={{ width: 12 }} /></button>
          <button style={btn} onClick={() => move(4)}>1 h<SkipForward style={{ width: 12 }} /></button>
          <button style={btn} onClick={() => setBin(detail.default_bin)} title="A moment chosen to illustrate an early warning; move the slider to see every moment, including misses">
            <Target style={{ width: 12 }} />{detail.default_is_example_warning ? 'Example early warning' : 'Start moment'}
          </button>
          <input type="range" min={detail.min_bin} max={maxBin} value={bin ?? 0} onChange={e => setBin(+e.target.value)} style={{ flex: 1, minWidth: 160 }} />
        </div>
      </div>

      <div className="tab-group">
        <button className={`tab-btn${tab === 'twin' ? ' active' : ''}`} onClick={() => setTab('twin')}><Activity style={{ width: 13, height: 13 }} /> Digital Twin</button>
        <button className={`tab-btn${tab === 'ehr' ? ' active' : ''}`} onClick={() => setTab('ehr')}><ClipboardList style={{ width: 13, height: 13 }} /> EHR</button>
        <button className={`tab-btn${tab === 'model' ? ' active' : ''}`} onClick={() => setTab('model')}><BarChart3 style={{ width: 13, height: 13 }} /> Model performance</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 340px', gap: 16, alignItems: 'start' }}>
        <div className="card" style={{ minWidth: 0 }}>
          {tab === 'twin' && (
            <>
              <div className="card-header"><div className="card-title"><Activity /> Live stream: glucose, insulin & meals (last 12 h)</div></div>
              {error && <div className="error-box" style={{ marginBottom: 10 }}>{error}</div>}
              <StreamChart stream={stream} />
              <div className="section-divider"><div className="section-divider-line" /><span className="section-divider-label">Signals the model is reading now</span><div className="section-divider-line" /></div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
                {(pred?.signals ?? []).map(s => (
                  <div key={s.name} style={{ padding: '9px 12px', borderRadius: 'var(--r-md)', border: '1px solid var(--border-glass)', background: 'rgba(255,255,255,0.02)' }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>{s.name}</div>
                    <div style={{ fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                      {s.value ?? '—'} <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 400 }}>{s.value != null ? s.unit : ''}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {tab === 'ehr' && <EhrPanel detail={detail} />}
          {tab === 'model' && <ModelCardV7 />}
        </div>

        <div className="card" style={{ height: 'fit-content' }}>
          <div className="card-header"><div className="card-title"><Brain /> Spike forecast</div><span className="card-badge">3 horizons</span></div>
          {pred ? ['30m', '60m', '2h'].map(h => <HorizonCard key={h} name={h} h={pred.horizons[h]} />) : <div className="skeleton" style={{ height: 240, borderRadius: 12 }} />}
          <div style={{ fontSize: 10, color: '#92704a', lineHeight: 1.5, marginTop: 6 }}>
            Red marker = accuracy-mode threshold, amber = safety-mode threshold. {pred?.note}
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── Root ────────────────────────────────────────────────── */
export default function TwinV7({ topbar }) {
  const [patients, setPatients] = useState([])
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    getV7Patients().then(d => { setPatients(d.patients); setSelected(d.patients[0]?.patient_id ?? null) })
      .catch(e => setError(e.response?.data?.detail || e.message))
  }, [])
  const body = useMemo(() => (selected ? <V7PatientView pid={selected} /> : null), [selected])
  return (
    <div className="app-layout">
      <V7Sidebar patients={patients} selectedId={selected} onSelect={setSelected} />
      <div className="main-content">
        {topbar}
        {error ? <div className="content-body"><div className="error-box"><AlertTriangle style={{ width: 16, height: 16 }} /> {error}</div></div> : body}
      </div>
    </div>
  )
}
