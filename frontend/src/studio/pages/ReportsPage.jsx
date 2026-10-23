import { Printer, FileText } from 'lucide-react'
import { useStudio, riskOf } from '../store'
import { Card, Banner, fmt } from '../ui'
import { labsMap, MomentBar } from './PatientPage'

const HL = { '30m': 'Next 30 minutes', '60m': 'Next 1 hour', '2h': 'Next 2 hours' }

export default function ReportsPage() {
  const { detail, patients, pid, overview, prediction, modes, context } = useStudio()
  const cur = patients.find(p => p.patient_id === pid)
  const L = labsMap(detail)
  const d = overview?.last_24h
  return (
    <div className="gt-grid" style={{ gap: 18 }}>
      <MomentBar />
      <div className="gt-row gt-noprint"><button className="gt-btn" onClick={() => window.print()}><Printer />Print / save as PDF</button>
        <span className="gt-small gt-muted">The report shows exactly what is on screen for the chosen moment.</span></div>
      <Card title={`Glucose risk report · Patient #${cur?.display_id ?? ''}`} icon={FileText} right={<span className="gt-small gt-muted">{overview?.now}</span>}>
        <div className="gt-two">
          <div>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Patient</div>
            <table className="gt-table"><tbody>
              <tr><td>Diabetes type</td><td className="num">{detail?.diabetes_type === 'T2D' ? 'Type 2' : 'Type 1'}</td></tr>
              <tr><td>Age / sex</td><td className="num">{L.Age ? Math.round(L.Age.value) : '—'} / {detail?.ehr?.sex ?? '—'}</td></tr>
              <tr><td>HbA1c (enrolment)</td><td className="num">{L.HbA1c ? `${L.HbA1c.value}%` : '—'}</td></tr>
              <tr><td>Conditions & medicines</td><td className="num">{(detail?.ehr?.conditions_and_drugs || []).join(', ') || '—'}</td></tr>
              <tr><td>Data source</td><td className="num">{cur?.dataset}</td></tr>
            </tbody></table>
          </div>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Last 24 hours</div>
            <table className="gt-table"><tbody>
              <tr><td>Current glucose</td><td className="num">{fmt(overview?.glucose_now)} mg/dL</td></tr>
              <tr><td>Mean (SD)</td><td className="num">{fmt(d?.mean)} ({fmt(d?.sd)}) mg/dL</td></tr>
              <tr><td>In range 70–180</td><td className="num">{fmt(d?.tir_pct, 1)}%</td></tr>
              <tr><td>Above 180 / below 70</td><td className="num">{fmt(d?.tar_pct, 1)}% / {fmt(d?.tbr_pct, 1)}%</td></tr>
              <tr><td>CGM coverage</td><td className="num">{fmt(d?.coverage_pct, 0)}%</td></tr>
            </tbody></table>
          </div>
        </div>
        <div style={{ fontWeight: 700, margin: '16px 0 6px' }}>Spike risk (glucose above 180 mg/dL)</div>
        {prediction ? (
          <table className="gt-table">
            <thead><tr><th>Horizon</th><th className="num">Probability</th><th>Risk ({modes.alert} mode)</th>{prediction.scenario && <th className="num">With today's context</th>}</tr></thead>
            <tbody>{['30m', '60m', '2h'].map(k => {
              const h = prediction.horizons[k], r = riskOf(h, modes.alert)
              return <tr key={k}><td>{HL[k]}</td><td className="num">{Math.round(100 * h.probability)}%</td><td>{r.label}{r.alert ? ' · alert' : ''}</td>
                {prediction.scenario && <td className="num">{Math.round(100 * prediction.scenario.horizons[k].probability)}%</td>}</tr>
            })}</tbody>
          </table>
        ) : <div className="gt-small gt-muted">No prediction at this moment (sensor gap).</div>}
        {prediction?.horizons?.['2h']?.early_warning?.active && <div style={{ marginTop: 10 }}><Banner kind="crit">Early warning active: glucose still in range, 2-hour risk above the early-warning threshold.</Banner></div>}
        {(context.activity !== 'not set' || context.sleep !== 'not set' || context.notes) && (
          <div className="gt-small" style={{ marginTop: 10 }}><b>Clinician context (not used by the model):</b> activity {context.activity}; sleep {context.sleep}{context.notes ? `; notes: ${context.notes}` : ''}</div>
        )}
        <div className="gt-note" style={{ marginTop: 14 }}>GlucoTwin research prototype. Predictions come from models trained on open datasets without this patient; they are not validated for clinical use and must not guide treatment.</div>
      </Card>
    </div>
  )
}
