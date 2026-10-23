import { useEffect, useRef, useState } from 'react'
import { Play, Pause, Bell, Activity, RotateCcw } from 'lucide-react'
import { useStudio, riskOf } from '../store'
import { Card, Segmented, Level, Banner, css } from '../ui'
import { GlucoseCard } from './PatientPage'

const HL = { '30m': '30 min', '60m': '1 hour', '2h': '2 hours' }

export default function MonitorPage() {
  const { detail, bin, setBin, prediction, overview, modes, pid } = useStudio()
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState('1')
  const [log, setLog] = useState([])
  const last = useRef(-1e9)

  useEffect(() => { setLog([]); last.current = -1e9; setPlaying(false) }, [pid])
  useEffect(() => {
    if (!playing || !detail) return
    const t = setInterval(() => setBin(b => {
      if (b >= detail.n_bins - 1) { setPlaying(false); return b }
      return b + 1
    }), 1200 / Number(speed))
    return () => clearInterval(t)
  }, [playing, speed, detail, setBin])

  // early-warning log: same rule as the validated v9 policy (in range, threshold, re-alert at most once an hour)
  useEffect(() => {
    const h = prediction?.horizons?.['2h']
    if (!h?.early_warning?.active || prediction.bin - last.current < 4 || prediction.bin < last.current) return
    last.current = prediction.bin
    setLog(l => [{ bin: prediction.bin, time: overview?.now, p: h.probability, actual: h.actual }, ...l].slice(0, 30))
  }, [prediction, overview])

  return (
    <div className="gt-grid" style={{ gap: 18 }}>
      <Card style={{ padding: '12px 16px' }}>
        <div className="gt-row" style={{ flexWrap: 'wrap' }}>
          <button className="gt-btn primary" style={{ padding: '10px 18px' }} onClick={() => setPlaying(p => !p)}>
            {playing ? <><Pause />Pause</> : <><Play />Play the record</>}
          </button>
          <Segmented label="Speed" value={speed} options={[['1', '1×'], ['3', '3×'], ['6', '6×']]} onChange={setSpeed} />
          <button className="gt-btn small" onClick={() => { setLog([]); last.current = -1e9 }}><RotateCcw />Clear log</button>
          <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
            <div className="gt-small gt-muted">Now</div><b>{overview?.now ?? '…'}</b>
          </div>
        </div>
        {detail && <input className="gt-range" style={{ marginTop: 10 }} type="range" min={detail.min_bin} max={detail.n_bins - 1} value={bin ?? 0}
          onChange={e => setBin(Number(e.target.value))} aria-label="Replay moment" />}
      </Card>
      <div className="gt-split">
        <GlucoseCard title="Live stream" showFutureToggle defaultHours="24" />
        <div className="gt-grid">
          {prediction ? ['30m', '60m', '2h'].map(k => {
            const h = prediction.horizons[k]
            const r = riskOf(h, modes.alert)
            return (
              <Card key={k} style={{ padding: '12px 16px' }}>
                <div className="gt-row"><b>Next {HL[k]}</b><span className="gt-right"><Level risk={r} /></span></div>
                <div className="gt-row" style={{ marginTop: 6 }}>
                  <span style={{ fontSize: '1.6rem', fontWeight: 800 }}>{Math.round(100 * h.probability)}%</span>
                  <div className="gt-progress" style={{ flex: 1 }}><div style={{ width: `${100 * h.probability}%`, background: r.level === 'high' ? css('crit') : r.level === 'medium' ? css('above') : css('in') }} /></div>
                </div>
              </Card>
            )
          }) : <div className="gt-skel" style={{ height: 260 }} />}
        </div>
      </div>
      <Card title="Early warnings fired during this replay" icon={Bell} right={<span className="gt-small gt-muted">{log.length} so far</span>}>
        {log.length === 0 ? <div className="gt-small gt-muted">None yet. Press play: a warning fires when glucose is still in range and the 2-hour risk crosses the early-warning threshold.</div> : (
          <table className="gt-table">
            <thead><tr><th>Time</th><th className="num">2-hour risk</th><th>What happened next</th></tr></thead>
            <tbody>{log.map(e => (
              <tr key={e.bin}><td>{e.time}</td><td className="num">{Math.round(100 * e.p)}%</td>
                <td>{e.actual ? (e.actual.spiked ? `Spiked (max ${Math.round(e.actual.max_glucose)} mg/dL)` : `False alarm (max ${Math.round(e.actual.max_glucose)})`) : 'not enough data'}</td></tr>))}</tbody>
          </table>
        )}
      </Card>
      <Banner>The monitor replays a real recorded week as if it were live. Every prediction uses only data up to “now”; the “what happened next” column is the replay answer key.</Banner>
    </div>
  )
}
