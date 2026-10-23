import { useState } from 'react'
import Studio from './studio/Studio'
import { loadModes, saveModes } from './studio/store'
import { Activity, Cpu, User, AlertTriangle, Dna } from 'lucide-react'
import Sidebar from './components/Sidebar'
import StatsRow from './components/StatsRow'
import PatientProfile from './components/PatientProfile'
import DigitalTwinView from './components/DigitalTwin'
import PredictionPanel from './components/PredictionPanel'
import TwinV7 from './components/TwinV7'
import { useDigitalTwin, useReadings, usePrediction } from './hooks/useApi'

/* ── Welcome ──────────────────────────────────────────────── */
function WelcomeScreen() {
  const features = [
    { icon: Dna,           title: '120 Virtual Patients',     desc: 'Synthetic EHR + CGM records' },
    { icon: Activity,      title: 'Real-Time CGM Trace',      desc: '12-hour glucose timeline' },
    { icon: Cpu,           title: 'AI Spike Prediction',      desc: 'HistGradientBoosting · synthetic ROC-AUC 0.958' },
    { icon: AlertTriangle, title: 'Risk Stratification',      desc: 'Low / Moderate / High flags' },
  ]
  return (
    <div className="welcome-wrap">
      <div className="welcome-content">
        <div className="welcome-icon"><Activity /></div>
        <h1 className="welcome-title">GlucoTwin AI</h1>
        <p className="welcome-subtitle">
          Select a virtual patient from the sidebar to explore their Digital Twin —
          real-time CGM readings, EHR history, and AI-powered 2-hour glucose spike prediction.
        </p>
        <div className="welcome-features">
          {features.map(f => (
            <div key={f.title} className="welcome-feature">
              <div className="welcome-feature-icon"><f.icon /></div>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--text-white)', fontSize: 'var(--fs-sm)', marginBottom: 2 }}>{f.title}</div>
                <div className="welcome-feature-text">{f.desc}</div>
              </div>
            </div>
          ))}
        </div>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '8px 16px',
          background: 'rgba(245,158,11,0.08)',
          border: '1px solid rgba(245,158,11,0.2)',
          borderRadius: 20,
          fontSize: 'var(--fs-xs)', color: '#f59e0b', fontWeight: 500,
        }}>
          <AlertTriangle style={{ width: 12, height: 12 }} />
          Experimental prototype · Synthetic data only · Not for clinical use
        </div>
      </div>
    </div>
  )
}

/* ── Patient View ─────────────────────────────────────────── */
function PatientView({ patientId }) {
  const { twin, loading, error } = useDigitalTwin(patientId)
  const { readings } = useReadings(patientId, 144)
  const { prediction, loading: predLoading, error: predError, runPrediction } = usePrediction(patientId)
  const [activeTab, setActiveTab] = useState('twin')

  if (loading) {
    return (
      <div className="content-body">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 330px', gap: 16 }}>
          <div className="card">
            <div className="skeleton" style={{ height: 20, width: '40%', marginBottom: 18 }} />
            <div className="skeleton" style={{ height: 52, borderRadius: 12, marginBottom: 16 }} />
            <div className="skeleton" style={{ height: 200, borderRadius: 12 }} />
          </div>
          <div className="card">
            <div className="skeleton" style={{ height: 20, width: '60%', marginBottom: 14 }} />
            <div className="skeleton" style={{ height: 48, borderRadius: 12, marginBottom: 14 }} />
            <div className="skeleton" style={{ height: 120, borderRadius: 12 }} />
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="content-body">
        <div className="error-box">
          <AlertTriangle style={{ width: 16, height: 16, flexShrink: 0 }} />
          Failed to load patient data: {error}
        </div>
      </div>
    )
  }

  const patient = twin?.patient_info

  return (
    <div className="content-body fade-in">
      {/* Patient banner */}
      {patient && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12, marginBottom: 18,
          padding: '10px 16px',
          background: 'rgba(59,130,246,0.06)',
          border: '1px solid rgba(59,130,246,0.14)',
          borderRadius: 'var(--r-lg)',
        }}>
          <User style={{ width: 16, height: 16, color: 'var(--accent-light)', flexShrink: 0 }} />
          <div style={{ fontWeight: 600, color: 'var(--text-white)', fontSize: 'var(--fs-sm)' }}>{patient.name}</div>
          <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{patient.patient_id}</div>
          {patient.hba1c && (
            <div style={{ marginLeft: 'auto', fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)', fontWeight: 500 }}>
              HbA1c <span style={{ color: patient.hba1c > 6.5 ? 'var(--risk-high)' : 'var(--risk-low)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{patient.hba1c}%</span>
            </div>
          )}
        </div>
      )}

      {/* Tabs */}
      <div className="tab-group">
        <button
          id="tab-twin"
          className={`tab-btn${activeTab === 'twin' ? ' active' : ''}`}
          onClick={() => setActiveTab('twin')}
        >
          <Cpu style={{ width: 13, height: 13 }} /> Digital Twin
        </button>
        <button
          id="tab-profile"
          className={`tab-btn${activeTab === 'profile' ? ' active' : ''}`}
          onClick={() => setActiveTab('profile')}
        >
          <User style={{ width: 13, height: 13 }} /> EHR Profile
        </button>
      </div>

      {/* Content grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 330px', gap: 16, alignItems: 'start' }}>
        <div className="card" style={{ minWidth: 0 }}>
          <div className="card-header">
            <div className="card-title">
              {activeTab === 'twin'
                ? <><Cpu /> Digital Twin — {patient?.name}</>
                : <><User /> EHR Profile</>}
            </div>
            {activeTab === 'twin' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div className="pulse-dot" />
                <span style={{ fontSize: 11, color: 'var(--risk-low)', fontWeight: 500 }}>Synthetic Live</span>
              </div>
            )}
          </div>

          {activeTab === 'twin'
            ? <DigitalTwinView twin={twin} readings={readings} />
            : <PatientProfile twin={twin} />
          }
        </div>

        {/* Prediction panel */}
        <div className="card" style={{ height: 'fit-content' }}>
          <div className="card-header">
            <div className="card-title">
              <Activity /> AI Prediction
            </div>
            <span className="card-badge">2-Hour</span>
          </div>
          <PredictionPanel
            patientId={patientId}
            prediction={prediction}
            loading={predLoading}
            error={predError}
            onPredict={runPrediction}
          />
        </div>
      </div>
    </div>
  )
}

/* ── Top bar with data-mode switch ────────────────────────── */
function Topbar({ mode, setMode, title, onNewLayout }) {
  const pill = active => ({
    padding: '5px 12px', borderRadius: 16, fontSize: 11, fontWeight: 600, cursor: 'pointer',
    border: `1px solid ${active ? 'rgba(59,130,246,0.5)' : 'var(--border-glass)'}`,
    background: active ? 'rgba(59,130,246,0.15)' : 'transparent',
    color: active ? 'var(--accent-light)' : 'var(--text-muted)',
  })
  return (
    <div className="topbar">
      <div className="topbar-left">
        <div>
          <div className="topbar-title">{title}</div>
          <div className="topbar-subtitle">GlucoTwin AI — Predictive Healthcare Digital Twin</div>
        </div>
      </div>
      <div className="topbar-right">
        <button style={pill(mode === 'v7')} onClick={() => setMode('v7')}>Real patients · v7</button>
        <button style={pill(mode === 'synthetic')} onClick={() => setMode('synthetic')}>Synthetic demo</button>
        <button style={pill(false)} onClick={onNewLayout}>New layout</button>
        <div className="experimental-badge">
          <AlertTriangle style={{ width: 11, height: 11 }} />
          {mode === 'v7' ? 'Research prototype · Open data' : 'Experimental · Synthetic'}
        </div>
      </div>
    </div>
  )
}

/* ── Classic layout (kept as a layout mode) ───────────────── */
function ClassicApp({ onNewLayout }) {
  const [mode, setMode] = useState('v7')
  const [selectedPatientId, setSelectedPatientId] = useState(null)

  if (mode === 'v7') {
    return <TwinV7 topbar={<Topbar mode={mode} setMode={setMode} title="Real-data Digital Twin" onNewLayout={onNewLayout} />} />
  }

  return (
    <div className="app-layout">
      <Sidebar selectedId={selectedPatientId} onSelect={setSelectedPatientId} />

      <div className="main-content">
        <Topbar mode={mode} setMode={setMode} onNewLayout={onNewLayout}
          title={selectedPatientId ? `Patient ${selectedPatientId}` : 'Dashboard Overview'} />

        {/* Stats always visible */}
        <div style={{ padding: '16px 24px 0', flexShrink: 0 }}>
          <StatsRow />
        </div>

        {/* Main area */}
        {selectedPatientId
          ? <PatientView patientId={selectedPatientId} />
          : <WelcomeScreen />
        }
      </div>
    </div>
  )
}

/* ── Root: modes (data / alert / theme / layout), remembered in this browser ── */
export default function App() {
  const [modes, setModesState] = useState(loadModes)
  const setModes = m => { setModesState(m); saveModes(m) }
  if (modes.layout === 'classic') return <ClassicApp onNewLayout={() => setModes({ ...modes, layout: 'studio' })} />
  return <Studio modes={modes} setModes={setModes} />
}
