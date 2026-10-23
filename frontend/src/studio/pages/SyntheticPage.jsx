/**
 * Synthetic demo in the studio layout: simulated patients and the separate synthetic model (/api/*).
 * Kept clearly labelled as synthetic; there is no daily-context what-if here (that model has no such inputs).
 */
import { useEffect, useState } from 'react'
import { User, Calendar, Droplet, Activity, Database, ClipboardList, LineChart, ChevronDown, Dna, ArrowRight, CheckCircle2, XCircle, FlaskConical } from 'lucide-react'
import { getPatients, getDigitalTwin, getReadings, postPredict } from '../../services/api'
import { Card, Stat, Gauge, BodyTwin, CgmChart, Banner, fmt, css } from '../ui'

const LEVEL = { low: 'low', moderate: 'medium', high: 'high' }

export default function SyntheticPage() {
  const [list, setList] = useState([])
  const [id, setId] = useState(null)
  const [twin, setTwin] = useState(null)
  const [readings, setReadings] = useState([])
  const [pred, setPred] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => { getPatients({ limit: 200 }).then(r => { setList(r.patients); setId(r.patients[0]?.patient_id) }) }, [])
  useEffect(() => {
    if (!id) return
    setTwin(null); setPred(null)
    getDigitalTwin(id).then(setTwin)
    getReadings(id, 144).then(r => setReadings(r.readings || []))
  }, [id])

  const run = () => { setBusy(true); postPredict(id).then(setPred).finally(() => setBusy(false)) }
  const p = twin?.patient_info
  const cur = list.find(x => x.patient_id === id)
  const points = readings.map(r => ({ time: new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), glucose: r.glucose_mgdl, future: false }))
  const last = readings[readings.length - 1]
  const has = k => readings.some(r => r[k] != null)
  const items = [['Glucose (CGM)', has('glucose_mgdl')], ['Heart rate', has('heart_rate_bpm')], ['Sleep', has('sleep_hours')], ['Activity (steps)', has('steps')], ['Health record', !!p]]
  const level = pred ? LEVEL[pred.risk_category] : null

  return (
    <div className="gt-patient-grid">
      <div className="gt-grid" style={{ gap: 18 }}>
        <Banner kind="warn"><b>Synthetic demo.</b> Simulated patients and a model trained only on simulated data. Names are generated. Switch to real patients in Settings.</Banner>
        <div className="gt-top-row">
          <Card title="Select patient" icon={User} style={{ padding: '16px 18px' }}>
            <div style={{ position: 'relative' }}>
              <div className="gt-row" style={{ padding: '10px 12px', borderRadius: 14, border: `1px solid ${css('border-strong')}`, background: css('surface-solid') }}>
                <div className="gt-user-av" style={{ width: 44, height: 44 }}>{cur?.name?.split(' ').map(s => s[0]).join('') ?? '—'}</div>
                <div><div style={{ fontWeight: 700 }}>{cur ? `${cur.patient_id} – ${cur.diabetes_type}` : 'Loading…'}</div><div className="gt-small gt-muted">Synthetic patient · {cur?.name}</div></div>
                <ChevronDown style={{ marginLeft: 'auto', width: 18, color: css('text-2') }} />
              </div>
              <select aria-label="Select patient" value={id ?? ''} onChange={e => setId(e.target.value)} style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', width: '100%' }}>
                {list.map(x => <option key={x.patient_id} value={x.patient_id}>{x.patient_id} · {x.diabetes_type} · {x.name}</option>)}
              </select>
            </div>
          </Card>
          <Stat label="Age" icon={Calendar} value={p?.age ?? '—'} unit="years" />
          <Stat label="HbA1c" icon={Droplet} value={p?.hba1c ? `${p.hba1c}%` : '—'} />
          <Stat label="Diabetes type" icon={Activity} value={p?.diabetes_type ?? '—'} />
          <Stat label="Data source" icon={Database} value="Synthetic" unit="simulated" />
        </div>
        <div className="gt-two">
          <Card title="Patient profile" icon={ClipboardList}>
            <div className="gt-kv">
              {[['Gender', p?.gender], ['Weight', p ? `${p.weight_kg} kg` : '—'], ['Height', p ? `${p.height_cm} cm` : '—'], ['BMI', p?.bmi]].map(([k, v]) => <div key={k}><div className="k">{k}</div><div className="v">{v ?? '—'}</div></div>)}
            </div>
            <div className="gt-two" style={{ marginTop: 14 }}>
              <div><b>Medical history</b><div className="gt-list">{(p?.diagnoses || []).map(d => <div key={d.code} className="gt-small">{d.description}</div>)}</div></div>
              <div><b>Lab results</b><div className="gt-list">{(p?.lab_results || []).map(l => <div key={l.test_name} className="gt-small"><span>{l.test_name}</span><b className="gt-right">{l.value} {l.unit}</b></div>)}</div></div>
            </div>
          </Card>
          <Card title="Glucose, last 24 hours" icon={LineChart}>{points.length ? <CgmChart points={points} height={250} /> : <div className="gt-skel" style={{ height: 250 }} />}</Card>
        </div>
        <Card title="Data completeness" icon={Database}>
          <div className="gt-list">{items.map(([k, ok]) => <div key={k}>{ok ? <CheckCircle2 style={{ width: 17, color: css('good') }} /> : <XCircle style={{ width: 17, color: css('text-3') }} />}{k}<span className="gt-right gt-small gt-muted">{ok ? 'simulated' : 'none'}</span></div>)}</div>
        </Card>
      </div>
      <div className="gt-grid">
        <div className="gt-card gt-twin-hero"><div className="gt-twin-hex"><Dna /></div><div><div style={{ fontSize: '1.2rem', fontWeight: 800 }}>Digital Twin</div><div className="gt-small gt-muted">Simulated record and sensor stream.</div></div></div>
        <div className="gt-twin-body">
          <BodyTwin glow={level ?? 'low'} />
          <div className="gt-twin-metric" style={{ top: 14 }}><span className="big">{fmt(last?.glucose_mgdl)}</span><div className="lbl">mg/dL · current glucose</div></div>
          <div className="gt-twin-metric" style={{ top: 112 }}><span className="big">{fmt(last?.heart_rate_bpm)}</span><div className="lbl">bpm · heart rate</div></div>
          <div className="gt-twin-metric" style={{ top: 210 }}><span style={{ fontWeight: 800 }}>{pred ? `${pred.risk_category} risk` : '—'}</span><div className="lbl">next 2 hours</div></div>
        </div>
        <div className="gt-ai">
          <div className="gt-row" style={{ fontWeight: 800, fontSize: '1.1rem' }}><FlaskConical style={{ width: 20 }} />AI prediction (synthetic model)</div>
          <div className="gt-row"><Gauge p={pred?.spike_probability} level={level} onDark size={190} ticks={[0.3, 0.6]} />
            <div className="gt-small" style={{ color: '#bfe9ec' }}>Risk of glucose above 180 mg/dL in the next 2 hours</div></div>
          <button className="gt-btn primary" onClick={run} disabled={busy || !id}>Generate 2-hour prediction<ArrowRight /></button>
          {pred && <div className="gt-small" style={{ color: '#cdeff1', marginTop: 10 }}>{pred.explanation}</div>}
          <div className="gt-small" style={{ color: '#8fc3c8', marginTop: 8 }}>{pred?.experimental_status ?? 'EXPERIMENTAL — synthetic data only — not for clinical use'}</div>
        </div>
      </div>
    </div>
  )
}
