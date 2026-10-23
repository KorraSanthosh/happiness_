import { useEffect, useState } from 'react'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LabelList } from 'recharts'
import { BarChart3, ShieldCheck, Database, Search, Layers } from 'lucide-react'
import { getV7ModelCard } from '../../services/api'
import { Card, Banner, pct, css } from '../ui'

const HL = { '30m': '30 min', '60m': '1 hour', '2h': '2 hours' }
const DS = { type1: 'All Type 1', type2: 'All Type 2', brist: 'BrisT1D', hupa: 'HUPA-UCM', uom: 'T1D-UOM', azt1d: 'AZT1D', sh_t1: 'Shanghai T1DM', sh_t2: 'Shanghai T2DM' }
const axis = { stroke: css('axis'), fontSize: 11, tickLine: false, axisLine: false }

export default function AnalyticsPage() {
  const [card, setCard] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => { getV7ModelCard().then(setCard).catch(e => setErr(e.message)) }, [])
  const s = card?.summary
  if (err) return <Banner kind="warn">{err}</Banner>
  if (!s?.horizons) return <div className="gt-skel" style={{ height: 400 }} />
  const H = Object.keys(s.horizons)
  const bars = H.map(h => ({ h: HL[h], model: 100 * s.horizons[h].accuracy_mode.all_open.accuracy, rule: 100 * s.horizons[h].accuracy_mode.all_open.rule_accuracy }))
  const v9 = s.v9
  const e = s.error_analysis
  return (
    <div className="gt-grid" style={{ gap: 18 }}>
      <div className="gt-two">
        <Card title="Accuracy on 187 people the model never saw" icon={BarChart3}>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={bars} margin={{ top: 20, right: 8, bottom: 0, left: -4 }} barGap={2}>
              <CartesianGrid stroke={css('grid')} vertical={false} />
              <XAxis dataKey="h" {...axis} />
              <YAxis {...axis} domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} unit="%" width={52} />
              <Tooltip cursor={{ fill: css('brand-soft') }} content={({ active, payload, label }) => active && payload?.length ? (
                <div className="gt-tip"><div className="t">{label}</div><div>GlucoTwin {payload[0].payload.model.toFixed(1)}%</div><div>Simple rule {payload[0].payload.rule.toFixed(1)}%</div></div>) : null} />
              <Bar dataKey="rule" fill={css('text-3')} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false}>
                <LabelList dataKey="rule" position="top" formatter={v => `${v.toFixed(1)}%`} style={{ fill: css('text-2'), fontSize: 11 }} />
              </Bar>
              <Bar dataKey="model" fill={css('in')} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false}>
                <LabelList dataKey="model" position="top" formatter={v => `${v.toFixed(1)}%`} style={{ fill: css('text-1'), fontSize: 11, fontWeight: 700 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="gt-legend"><span><i style={{ background: css('text-3') }} />Simple rule: "glucose above 180 now"</span><span><i style={{ background: css('in') }} />GlucoTwin</span></div>
        </Card>
        <Card title="Per horizon" icon={Layers}>
          <table className="gt-table">
            <thead><tr><th>Horizon</th><th className="num">Accuracy</th><th className="num">Spikes caught</th><th className="num">AUC</th><th className="num">OhioT1DM (external)</th></tr></thead>
            <tbody>{H.map(h => {
              const r = s.horizons[h].accuracy_mode.all_open
              return <tr key={h}><td>{HL[h]}</td><td className="num">{pct(r.accuracy, 1)}</td><td className="num">{pct(r.recall, 1)}</td><td className="num">{r.roc_auc?.toFixed(3)}</td><td className="num">{pct(s.horizons[h].ohio_external.accuracy, 1)}</td></tr>
            })}</tbody>
          </table>
          <div className="gt-note" style={{ marginTop: 8 }}>7-fold cross-validation where each person is held out; OhioT1DM was never used for training. Every number is re-derived by the project's verifier.</div>
        </Card>
      </div>
      {v9 && (
        <Card title={`v9 update of the 2-hour model — tested once on ${v9.test_people} locked-away people`} icon={ShieldCheck}>
          <table className="gt-table">
            <thead><tr><th>2 hours, locked test set</th><th className="num">v7</th><th className="num">v9 (in use)</th></tr></thead>
            <tbody>
              <tr><td>Accuracy</td><td className="num">{pct(v9.test.v7_config.accuracy, 2)}</td><td className="num">{pct(v9.test.v9.accuracy, 2)}</td></tr>
              <tr><td>Ranking (AUC)</td><td className="num">{v9.test.v7_config.roc_auc.toFixed(4)}</td><td className="num">{v9.test.v9.roc_auc.toFixed(4)}</td></tr>
              <tr><td>OhioT1DM accuracy</td><td className="num">{pct(v9.test.v7_config.ohio_accuracy, 2)}</td><td className="num">{pct(v9.test.v9.ohio_accuracy, 2)}</td></tr>
              {(() => {
                const a = v9.alert_policy.arms, b = a['baseline + v7 policy'], c = a['candidate + v9 policy']
                const fad = x => (x.false_alerts / (v9.test_rows / 96)).toFixed(2)
                return <>
                  <tr><td>Serious spikes warned 30–120 min ahead</td><td className="num">{b.serious_warned.toLocaleString()} of {b.serious_episodes.toLocaleString()}</td><td className="num">{c.serious_warned.toLocaleString()} of {c.serious_episodes.toLocaleString()}</td></tr>
                  <tr><td>False alerts per patient-day</td><td className="num">{fad(b)}</td><td className="num">{fad(c)}</td></tr>
                </>
              })()}
            </tbody>
          </table>
          <div className="gt-note" style={{ marginTop: 8 }}>Ranking improved slightly (significant); the accuracy change is within noise. The new alert rule warns only while glucose is still in range. 30-minute and 1-hour models are unchanged (v7).</div>
        </Card>
      )}
      <div className="gt-two">
        <Card title="2-hour accuracy by dataset" icon={Database}>
          <table className="gt-table">
            <thead><tr><th>Dataset</th><th className="num">Accuracy</th><th className="num">Simple rule</th><th className="num">People</th></tr></thead>
            <tbody>{Object.entries(s.horizons['2h'].accuracy_mode).filter(([k]) => k !== 'all_open').map(([k, r]) => (
              <tr key={k}><td>{DS[k] ?? k}</td><td className="num">{pct(r.accuracy, 1)}</td><td className="num">{pct(r.rule_accuracy, 1)}</td><td className="num">{r.people ?? '—'}</td></tr>))}</tbody>
          </table>
        </Card>
        {e && (
          <Card title="Where the remaining error comes from" icon={Search}>
            <div className="gt-list">
              <div><span>2-hour errors with the actual peak within ±10% of 180</span><b className="gt-right">{pct(e.label_reliability_2h.all_open.uncertain_share_of_errors, 1)}</b></div>
              <div><span>Serious spike episodes with no warning 30–120 min ahead</span><b className="gt-right">{pct(e.serious_spikes_2h.episode_level.miss_rate, 1)}</b></div>
              <div><span>CGM vs fingerstick difference (MARD, Shanghai)</span><b className="gt-right">{e.cgm_vs_fingerstick.overall.mard_pct}%</b></div>
              <div><span>Calibration error, 2 hours (ECE)</span><b className="gt-right">{(100 * e.calibration['2h'].ece).toFixed(2)}%</b></div>
            </div>
            <div className="gt-note" style={{ marginTop: 8 }}>"Near 180" uses the real outcome, so it explains errors rather than predicting them.</div>
          </Card>
        )}
      </div>
    </div>
  )
}
