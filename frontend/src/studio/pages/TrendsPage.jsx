import { useEffect, useState } from 'react'
import { BarChart3, Clock, Droplet, Target, Activity, TrendingUp } from 'lucide-react'
import { getV7Trends } from '../../services/api'
import { useStudio } from '../store'
import { Card, Stat, RangeDays, HourlyProfile, fmt } from '../ui'
import { MomentBar } from './PatientPage'

export default function TrendsPage() {
  const { pid, bin } = useStudio()
  const [t, setT] = useState(null)
  useEffect(() => {
    if (!pid || bin == null) return
    let live = true
    const id = setTimeout(() => getV7Trends(pid, bin).then(r => live && setT(r)).catch(() => live && setT(null)), 150)
    return () => { live = false; clearTimeout(id) }
  }, [pid, bin])
  const o = t?.overall
  const days = t?.days?.length ?? 0
  return (
    <div className="gt-grid" style={{ gap: 18 }}>
      <MomentBar />
      <div className="gt-four">
        <Stat label="Time in range (70–180)" icon={Target} value={o ? `${fmt(o.tir_pct, 0)}%` : '—'} unit="clinical target: over 70%" />
        <Stat label="Mean glucose" icon={Droplet} value={o ? fmt(o.mean, 0) : '—'} unit="mg/dL" />
        <Stat label="Variability (CV)" icon={Activity} value={o ? `${fmt(o.cv_pct, 0)}%` : '—'} unit="target: 36% or less" />
        <Stat label="Estimated A1c (GMI)" icon={TrendingUp} value={o ? `${fmt(o.gmi_pct, 1)}%` : '—'}
          unit={days < 14 ? `only ${days} day${days === 1 ? '' : 's'} of data (needs ≥ 14)` : 'from CGM mean'} />
      </div>
      <div className="gt-two">
        <Card title="Time in range by day" icon={BarChart3}>{t ? <RangeDays days={t.days} /> : <div className="gt-skel" style={{ height: 240 }} />}</Card>
        <Card title="Daily pattern (time of day)" icon={Clock}>{t ? <HourlyProfile profile={t.hourly_profile} /> : <div className="gt-skel" style={{ height: 240 }} />}</Card>
      </div>
      {t && (
        <Card title="Day by day" icon={BarChart3}>
          <table className="gt-table">
            <thead><tr><th>Day</th><th className="num">Mean</th><th className="num">In range</th><th className="num">Above 180</th><th className="num">Below 70</th><th className="num">Rises above 180</th><th className="num">Peak</th><th className="num">CGM coverage</th></tr></thead>
            <tbody>{t.days.map(d => (
              <tr key={d.day}><td>Day {d.day}</td><td className="num">{fmt(d.mean)}</td><td className="num">{fmt(d.tir_pct, 1)}%</td><td className="num">{fmt(d.tar_pct, 1)}%</td>
                <td className="num">{fmt(d.tbr_pct, 1)}%</td><td className="num">{d.spikes}</td><td className="num">{fmt(d.max)}</td><td className="num">{fmt(d.coverage_pct, 0)}%</td></tr>))}</tbody>
          </table>
          <div className="gt-note" style={{ marginTop: 8 }}>Everything here uses the record up to the chosen moment only. "Rises above 180" counts moments where glucose crossed 180 from an in-range reading.</div>
        </Card>
      )}
    </div>
  )
}
