import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, ComposedChart, Bar, Line
} from 'recharts'
import { formatTime } from '../utils'

/* ── Custom tooltip ──────────────────────────────────────── */
const GlucoseTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const glucose = payload.find(p => p.dataKey === 'glucose')?.value
  const hr = payload.find(p => p.dataKey === 'heart_rate')?.value
  return (
    <div style={{
      background: 'rgba(8,14,26,0.95)',
      border: '1px solid rgba(96,165,250,0.3)',
      borderRadius: 10, padding: '10px 14px',
      backdropFilter: 'blur(12px)',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
      fontSize: 12, fontFamily: 'Inter, sans-serif'
    }}>
      <div style={{ color: '#94a3b8', marginBottom: 6, fontSize: 11 }}>{label}</div>
      {glucose != null && (
        <div style={{ color: '#e2e8f0', fontWeight: 600 }}>
          Glucose: <span style={{ color: glucose > 180 ? '#ef4444' : glucose > 140 ? '#f59e0b' : '#22c55e' }}>{glucose} mg/dL</span>
        </div>
      )}
      {hr != null && (
        <div style={{ color: '#e2e8f0', fontWeight: 500, marginTop: 3 }}>
          HR: <span style={{ color: '#06b6d4' }}>{hr} bpm</span>
        </div>
      )}
    </div>
  )
}

const SensorTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(8,14,26,0.95)',
      border: '1px solid rgba(96,165,250,0.3)',
      borderRadius: 10, padding: '10px 14px',
      backdropFilter: 'blur(12px)',
      fontSize: 12, fontFamily: 'Inter, sans-serif'
    }}>
      <div style={{ color: '#94a3b8', marginBottom: 5, fontSize: 11 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: '#e2e8f0', fontWeight: 500, marginTop: 2 }}>
          <span style={{ color: p.color }}>{p.name}: </span>{p.value}
        </div>
      ))}
    </div>
  )
}

/* ── Glucose timeline chart ──────────────────────────────── */
export function GlucoseChart({ readings = [] }) {
  const data = readings.slice(-72).map(r => ({
    time: formatTime(r.timestamp),
    glucose: Math.round(r.glucose_mgdl),
    heart_rate: Math.round(r.heart_rate_bpm),
  }))

  const ticks = data.filter((_, i) => i % 12 === 0).map(d => d.time)

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={210}>
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
          <defs>
            <linearGradient id="glucoseGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis
            dataKey="time"
            ticks={ticks}
            tick={{ fill: '#4b5c73', fontSize: 10, fontFamily: 'Inter' }}
            axisLine={false} tickLine={false}
          />
          <YAxis
            domain={[60, 260]}
            tick={{ fill: '#4b5c73', fontSize: 10, fontFamily: 'Inter' }}
            axisLine={false} tickLine={false} width={34}
          />
          <Tooltip content={<GlucoseTooltip />} />
          <ReferenceLine y={180} stroke="rgba(239,68,68,0.5)" strokeDasharray="5 3" strokeWidth={1.5}
            label={{ value: '180', fill: '#ef4444', fontSize: 9, position: 'right' }} />
          <ReferenceLine y={70}  stroke="rgba(245,158,11,0.4)" strokeDasharray="5 3" strokeWidth={1}
            label={{ value: '70', fill: '#f59e0b', fontSize: 9, position: 'right' }} />
          <Area dataKey="glucose" stroke="#3b82f6" strokeWidth={2.5} fill="url(#glucoseGrad)"
            dot={false} activeDot={{ r: 4, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }} name="Glucose (mg/dL)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ── Sensor trend chart (HR + Activity) ──────────────────── */
export function SensorChart({ readings = [] }) {
  const data = readings.slice(-48).map(r => ({
    time: formatTime(r.timestamp),
    heart_rate: Math.round(r.heart_rate_bpm),
    activity: Math.round((r.activity_score ?? 0) * 10),
  }))

  const ticks = data.filter((_, i) => i % 8 === 0).map(d => d.time)

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <defs>
            <linearGradient id="actGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#a78bfa" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis dataKey="time" ticks={ticks}
            tick={{ fill: '#4b5c73', fontSize: 10, fontFamily: 'Inter' }}
            axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#4b5c73', fontSize: 10, fontFamily: 'Inter' }} axisLine={false} tickLine={false} width={30} />
          <Tooltip content={<SensorTooltip />} />
          <Bar dataKey="activity" fill="rgba(167,139,250,0.2)" stroke="rgba(167,139,250,0.4)"
            strokeWidth={1} radius={[2, 2, 0, 0]} name="Activity" />
          <Line dataKey="heart_rate" stroke="#06b6d4" strokeWidth={2} dot={false}
            activeDot={{ r: 3, fill: '#06b6d4', stroke: '#fff', strokeWidth: 1.5 }} name="Heart Rate" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ── Feature importance bar chart ────────────────────────── */
export function FeatureChart({ features = [] }) {
  const top = features.slice(0, 6)
  const max = top[0]?.importance ?? 1
  return (
    <div>
      {top.map((f, i) => (
        <div key={f.feature} className="feature-bar-item fade-in" style={{ animationDelay: `${i * 0.06}s` }}>
          <div className="feature-bar-header">
            <span className="feature-name" title={f.description}>{f.description || f.feature}</span>
            <span className="feature-pct">{(f.importance * 100).toFixed(1)}%</span>
          </div>
          <div className="feature-bar-track">
            <div className="feature-bar-fill" style={{ width: `${(f.importance / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}
