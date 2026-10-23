import { Activity, AlertTriangle, CheckCircle2, XCircle, Eye, Gauge as GaugeIcon } from 'lucide-react'
import { useStudio, riskOf } from '../store'
import { Card, Gauge, Level, Banner, fmt, css } from '../ui'
import { MomentBar } from './PatientPage'

const HL = { '30m': 'Next 30 minutes', '60m': 'Next 1 hour', '2h': 'Next 2 hours' }

function Horizon({ name, h, s }) {
  const { modes } = useStudio()
  const shown = s ?? h
  const risk = riskOf(shown, modes.alert)
  const correct = h.actual ? h.actual.spiked === (riskOf(h, modes.alert).alert) : null
  return (
    <Card>
      <div className="gt-row" style={{ justifyContent: 'space-between' }}>
        <b>{HL[name]}</b><Level risk={risk} />
      </div>
      <div style={{ display: 'grid', placeItems: 'center', margin: '6px 0' }}>
        <Gauge p={shown.probability} level={risk.level} ticks={[shown.threshold_safety_mode, shown.threshold_accuracy_mode]} size={200} />
      </div>
      <div className="gt-small gt-muted" style={{ textAlign: 'center' }}>chance glucose goes above 180 mg/dL</div>
      {s && <div className="gt-small" style={{ textAlign: 'center', marginTop: 4 }}>Recorded data only: <b>{Math.round(100 * h.probability)}%</b></div>}
      <div className="gt-list" style={{ marginTop: 10 }}>
        <div><span className="gt-muted">Watch from</span><span className="gt-right gt-mono">{Math.round(100 * h.threshold_safety_mode)}%</span></div>
        <div><span className="gt-muted">High risk from</span><span className="gt-right gt-mono">{Math.round(100 * h.threshold_accuracy_mode)}%</span></div>
        <div><span className="gt-muted">Model</span><span className="gt-right gt-mono gt-small">{h.model_version}</span></div>
        {h.actual && (
          <div>
            {correct ? <CheckCircle2 style={{ width: 16, color: css('good') }} /> : <XCircle style={{ width: 16, color: css('crit') }} />}
            <span className="gt-small">Replay: {h.actual.spiked ? 'it spiked' : 'it stayed ≤ 180'} (max {Math.round(h.actual.max_glucose)})</span>
          </div>
        )}
      </div>
      {name === '2h' && h.early_warning && (
        <div style={{ marginTop: 10 }}>
          {(s ?? h).early_warning.active
            ? <Banner kind="crit"><b>Early warning.</b> Glucose is still in range and the 2-hour risk is above {Math.round(100 * h.early_warning.threshold)}%. This is when acting helps most.</Banner>
            : <div className="gt-note">Early warning fires only while glucose ≤ 180 and risk ≥ {Math.round(100 * h.early_warning.threshold)}% (at most once an hour). Not active now.</div>}
        </div>
      )}
    </Card>
  )
}

export default function PredictionPage() {
  const { prediction, predError, modes, applied } = useStudio()
  const sc = prediction?.scenario
  return (
    <div className="gt-grid" style={{ gap: 18 }}>
      <MomentBar />
      {predError && <Banner kind="warn">{predError}</Banner>}
      {prediction && (
        <>
          <div className="gt-three">
            {['30m', '60m', '2h'].map(k => <Horizon key={k} name={k} h={prediction.horizons[k]} s={sc?.horizons?.[k]} />)}
          </div>
          {sc && (
            <Card title="Today's context (what-if)" icon={Activity}>
              <div className="gt-small" style={{ marginBottom: 8 }}>
                Applied now: {Object.entries(sc.inputs).filter(([, v]) => v).map(([k, v]) => `${k.replace('_', ' ')} ${v}`).join(', ')}.
              </div>
              <table className="gt-table">
                <thead><tr><th>Horizon</th><th className="num">Recorded data</th><th className="num">With context</th><th>Inputs the model could use</th></tr></thead>
                <tbody>{['30m', '60m', '2h'].map(k => (
                  <tr key={k}><td>{HL[k]}</td><td className="num">{Math.round(100 * prediction.horizons[k].probability)}%</td>
                    <td className="num">{Math.round(100 * sc.horizons[k].probability)}%</td>
                    <td className="gt-small">{sc.used_by_horizon[k].length ? sc.used_by_horizon[k].join(', ') : 'none (unchanged)'}</td></tr>))}</tbody>
              </table>
              {sc.ignored.map(t => <div key={t} className="gt-hint" style={{ marginTop: 6 }}>Ignored — {t}</div>)}
              <div className="gt-note" style={{ marginTop: 8 }}>{sc.note}</div>
            </Card>
          )}
          <div className="gt-two">
            <Card title="Signals the model read at this moment" icon={Eye}>
              <div className="gt-list">
                {prediction.signals.map(s => (
                  <div key={s.name}><span>{s.name}</span><span className="gt-right gt-mono" style={{ fontWeight: 700 }}>{s.value ?? '—'}</span><span className="gt-small gt-muted" style={{ width: 110 }}>{s.unit}</span></div>
                ))}
              </div>
              <div className="gt-note" style={{ marginTop: 8 }}>Computed live from the raw stream up to “now” in {fmt(prediction.inference_ms)} ms. {prediction.note}</div>
            </Card>
            <Card title="How to read the risk" icon={GaugeIcon}>
              <div className="gt-list">
                <div><span className="gt-level low">Low</span><span className="gt-small gt-right">below the watch threshold</span></div>
                <div><span className="gt-level medium">Medium</span><span className="gt-small gt-right">between the watch and high-risk thresholds</span></div>
                <div><span className="gt-level high">High</span><span className="gt-small gt-right">above the high-risk threshold</span></div>
              </div>
              <div className="gt-note" style={{ marginTop: 10 }}>
                Alert mode: <b>{modes.alert === 'safety' ? 'Safety' : 'Standard'}</b>.{' '}
                {modes.alert === 'safety' ? 'Medium and high risk raise an alert: more spikes are caught, with more false alarms.'
                  : 'Only high risk raises an alert: fewer false alarms, some spikes are missed.'}{' '}
                Thresholds were chosen on held-out patients during validation; change the mode in Settings.
              </div>
              {applied && <div className="gt-hint used" style={{ marginTop: 6 }}>Daily context is applied to these predictions.</div>}
            </Card>
          </div>
        </>
      )}
      {!prediction && !predError && <div className="gt-skel" style={{ height: 320 }} />}
      <div className="gt-note"><AlertTriangle style={{ width: 12, verticalAlign: -2 }} /> Research prototype — not a medical device, not for clinical use.</div>
    </div>
  )
}
