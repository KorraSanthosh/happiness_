/**
 * GlucoTwin Studio — the light clinical layout: sidebar navigation, page header with the data-mode badge,
 * alert bell and settings, and the seven pages. Modes (data / alert / theme / layout) live in App.
 */
import { useEffect } from 'react'
import {
  Home, Activity, MonitorDot, TrendingUp, FileText, PieChart, Settings, Bell, Droplets, User, UploadCloud,
} from 'lucide-react'
import './studio.css'
import { StudioProvider, useStudio, riskOf, backgroundVars } from './store'
import PatientPage from './pages/PatientPage'
import PredictionPage from './pages/PredictionPage'
import MonitorPage from './pages/MonitorPage'
import TrendsPage from './pages/TrendsPage'
import ReportsPage from './pages/ReportsPage'
import AnalyticsPage from './pages/AnalyticsPage'
import SettingsPage from './pages/SettingsPage'
import SyntheticPage from './pages/SyntheticPage'
import MyDataPage from './pages/MyDataPage'

const PAGES = {
  patient: { label: 'Patient', icon: Home, title: 'Patient input', sub: 'Select a patient, pick a moment and add today’s context to generate a 2-hour glucose spike prediction.', Comp: PatientPage },
  mydata: { label: 'My data', icon: UploadCloud, title: 'Upload my data', sub: 'Upload your own CGM export (Dexcom, FreeStyle Libre or any CSV) and see your own digital twin.', Comp: MyDataPage, real: true },
  prediction: { label: 'Prediction', icon: Activity, title: 'Prediction', sub: 'Spike risk for the next 30 minutes, 1 hour and 2 hours, and what the model read.', Comp: PredictionPage, real: true },
  monitor: { label: 'Live monitor', icon: MonitorDot, title: 'Live monitor', sub: 'Replay the recording as if live and watch the early warnings fire.', Comp: MonitorPage, real: true },
  trends: { label: 'Trends', icon: TrendingUp, title: 'Trends', sub: 'Time in range, daily pattern and day-by-day summary up to the chosen moment.', Comp: TrendsPage, real: true },
  reports: { label: 'Reports', icon: FileText, title: 'Reports', sub: 'A printable one-page summary for the chosen moment.', Comp: ReportsPage, real: true },
  analytics: { label: 'Analytics', icon: PieChart, title: 'Model analytics', sub: 'How well the models do on people they never saw, and where they fail.', Comp: AnalyticsPage },
  settings: { label: 'Settings', icon: Settings, title: 'Settings & modes', sub: 'Data, alert, theme and layout modes.', Comp: SettingsPage },
}

function Shell() {
  const { modes, page, setPage, prediction, pid } = useStudio()
  const own = pid?.startsWith('upload_')
  const synthetic = modes.data === 'synthetic'
  const cur = PAGES[page] ?? PAGES.patient
  const Comp = synthetic && page === 'patient' ? SyntheticPage : cur.Comp
  const alerts = synthetic || !prediction ? 0 : ['30m', '60m', '2h'].filter(k => {
    const h = prediction.horizons[k]
    return riskOf(h, modes.alert).alert || h.early_warning?.active
  }).length

  useEffect(() => { if (synthetic && cur.real) setPage('patient') }, [synthetic, cur.real, setPage])

  return (
    <div className="gt-app" data-theme={modes.theme} style={backgroundVars(modes.bg)}>
      <aside className="gt-side">
        <div className="gt-brand">
          <div className="gt-brand-mark"><Droplets style={{ width: 24, height: 24, color: '#fff' }} /></div>
          <div><div className="gt-brand-name">GlucoTwin AI</div><div className="gt-brand-sub">Digital patient twin</div></div>
        </div>
        <nav className="gt-nav" aria-label="Main">
          {Object.entries(PAGES).map(([k, p]) => (
            <button key={k} className={page === k ? 'active' : ''} aria-current={page === k ? 'page' : undefined}
              disabled={synthetic && p.real} title={synthetic && p.real ? 'Available with real patients (Settings → Data mode)' : undefined}
              onClick={() => setPage(k)}><p.icon />{p.label}</button>
          ))}
        </nav>
        <div className="gt-side-tip">
          <b>How to use</b>
          <ol>
            <li>Select a patient</li>
            <li>Pick the moment (“now”)</li>
            <li>Optionally add today’s meal or medication</li>
            <li>Generate the 2-hour prediction</li>
          </ol>
        </div>
        <div className="gt-side-foot">Open data (CC BY 4.0) · research prototype · not for clinical use</div>
      </aside>
      <main className="gt-main">
        <header className="gt-top">
          <div className="gt-top-icon"><cur.icon /></div>
          <div><h1 className="gt-h1">{cur.title}</h1><div className="gt-sub">{synthetic && page === 'patient' ? 'Select a simulated patient and generate a 2-hour glucose spike prediction (synthetic model).' : cur.sub}</div></div>
          <div className="gt-top-right">
            <span className={`gt-pill ${synthetic ? 'synthetic' : ''}`}><span className="gt-led" />{synthetic ? 'Synthetic data' : own ? 'Your data' : 'Real data · replay'}</span>
            <button className="gt-icon-btn" aria-label={`${alerts} active alerts`} title="Alerts at this moment" onClick={() => setPage(synthetic ? 'patient' : 'prediction')}>
              <Bell />{alerts > 0 && <span className="gt-dot">{alerts}</span>}
            </button>
            <button className="gt-icon-btn" aria-label="Settings" onClick={() => setPage('settings')}><Settings /></button>
            <div className="gt-user">
              <div className="gt-user-av"><User style={{ width: 16 }} /></div>
              <div><div className="gt-user-name">Clinician view</div><div className="gt-user-role">{modes.alert === 'safety' ? 'Safety alerts' : 'Standard alerts'}</div></div>
            </div>
          </div>
        </header>
        <Comp />
      </main>
    </div>
  )
}

export default function Studio({ modes, setModes }) {
  useEffect(() => {
    document.body.classList.add('gt-body')
    return () => document.body.classList.remove('gt-body')
  }, [])
  return (
    <StudioProvider modes={modes} setModes={setModes}>
      <Shell />
    </StudioProvider>
  )
}
