/**
 * Studio building blocks: cards, stat tiles, risk gauge, the digital-twin body figure and the charts.
 * Chart colours come from CSS tokens (validated CVD-safe for below / in / above range in both themes).
 */
import { useId, useState } from 'react'
import {
  ResponsiveContainer, ComposedChart, Area, Line, Bar, BarChart, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ReferenceArea,
} from 'recharts'
import { AlertTriangle, CheckCircle2, Info, ShieldAlert } from 'lucide-react'

export const css = name => `var(--gt-${name})`
export const fmt = (v, d = 0) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d))
export const pct = (v, d = 0) => (v == null ? '—' : `${(100 * v).toFixed(d)}%`)

export function Card({ title, icon: Icon, right, children, style, className = '' }) {
  return (
    <div className={`gt-card ${className}`} style={style}>
      {title && (
        <div className="gt-card-title">{Icon && <Icon />}{title}{right && <div className="gt-right">{right}</div>}</div>
      )}
      {children}
    </div>
  )
}

export function Stat({ label, icon: Icon, value, unit }) {
  return (
    <div className="gt-card gt-stat">
      <div className="gt-stat-label">{label}</div>
      {Icon && <Icon className="gt-stat-icon" />}
      <div className="gt-stat-value">{value}</div>
      {unit && <div className="gt-stat-unit">{unit}</div>}
    </div>
  )
}

export function Segmented({ value, options, onChange, label }) {
  return (
    <div className="gt-seg" role="radiogroup" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={v} role="radio" aria-checked={value === v} className={value === v ? 'active' : ''} onClick={() => onChange(v)}>{text}</button>
      ))}
    </div>
  )
}

export function Level({ risk }) {
  if (!risk) return null
  const Icon = risk.level === 'high' ? ShieldAlert : risk.level === 'medium' ? AlertTriangle : CheckCircle2
  return <span className={`gt-level ${risk.level}`}><Icon />{risk.label}{risk.alert ? ' · alert' : ''}</span>
}

export function Banner({ kind = 'info', children }) {
  const Icon = kind === 'info' ? Info : AlertTriangle
  return <div className={`gt-banner ${kind}`}><Icon /><div>{children}</div></div>
}

export const levelColor = level => (level === 'high' ? css('crit') : level === 'medium' ? css('above') : css('in'))

/** Half-circle gauge. Tick marks show the model's two validated thresholds. */
export function Gauge({ p, level, ticks = [], size = 220, onDark = false }) {
  const r = 80, cx = 100, cy = 96, sw = 16
  const pt = t => [cx - r * Math.cos(Math.PI * t), cy - r * Math.sin(Math.PI * t)]
  const arc = t => { const [x, y] = pt(Math.max(0.001, Math.min(1, t))); return `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${x} ${y}` }
  return (
    <svg viewBox="0 0 200 112" width={size} style={{ flexShrink: 0, maxWidth: '100%' }} role="img" aria-label={`Spike probability ${Math.round(100 * (p ?? 0))} percent`}>
      <path d={arc(1)} fill="none" stroke={onDark ? 'rgba(255,255,255,0.14)' : css('surface-2')} strokeWidth={sw} strokeLinecap="round" />
      {p != null && <path d={arc(p)} fill="none" stroke={levelColor(level)} strokeWidth={sw} strokeLinecap="round" />}
      {ticks.map((t, i) => {
        const [x1, y1] = [cx - (r - 12) * Math.cos(Math.PI * t), cy - (r - 12) * Math.sin(Math.PI * t)]
        const [x2, y2] = [cx - (r + 12) * Math.cos(Math.PI * t), cy - (r + 12) * Math.sin(Math.PI * t)]
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={onDark ? '#e7fbfc' : css('text-2')} strokeWidth={1.5} opacity={0.75} />
      })}
      <text x={cx} y={cy - 6} textAnchor="middle" fontSize="34" fontWeight="800" fill={onDark ? '#ffffff' : css('text-1')}>
        {p == null ? '—' : p > 0.995 ? '>99' : p < 0.005 ? '<1' : Math.round(100 * p)}<tspan fontSize="16" dx="2">%</tspan>
      </text>
    </svg>
  )
}

/** Stylised digital-twin figure (decorative); the numbers around it are the real ones. */
export function BodyTwin({ glow = 'low' }) {
  const id = useId().replace(/:/g, '')
  const organ = glow === 'high' ? '#ff6b4a' : glow === 'medium' ? '#ffb347' : '#5be3c8'
  return (
    <svg viewBox="0 0 200 330" style={{ position: 'absolute', left: 6, top: 10, height: 'calc(100% - 20px)' }} aria-hidden="true">
      <defs>
        <linearGradient id={`b${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#7ff0f0" stopOpacity="0.55" />
          <stop offset="1" stopColor="#1aa3b5" stopOpacity="0.18" />
        </linearGradient>
        <pattern id={`g${id}`} width="8" height="8" patternUnits="userSpaceOnUse">
          <path d="M8 0H0V8" fill="none" stroke="#a6f4f4" strokeOpacity="0.18" strokeWidth="0.6" />
        </pattern>
        <radialGradient id={`o${id}`}><stop offset="0" stopColor={organ} /><stop offset="1" stopColor={organ} stopOpacity="0" /></radialGradient>
        <filter id={`f${id}`}><feGaussianBlur stdDeviation="3" /></filter>
      </defs>
      <circle cx="100" cy="36" r="22" fill={`url(#b${id})`} stroke="#9ff3f3" strokeOpacity="0.7" />
      <path id={`p${id}`} d="M88 60 C88 68 86 73 80 75 L62 81 C52 85 48 93 46 105 L38 168 C37 178 40 186 44 190 L50 188 C52 182 52 174 54 166 L62 112 L66 150 C66 166 68 176 70 184 L72 236 L76 312 L93 312 L96 236 L100 198 L104 236 L107 312 L124 312 L128 236 L130 184 C132 176 134 166 134 150 L138 112 L146 166 C148 174 148 182 150 188 L156 190 C160 186 163 178 162 168 L154 105 C152 93 148 85 138 81 L120 75 C114 73 112 68 112 60 Z"
        fill={`url(#b${id})`} stroke="#9ff3f3" strokeOpacity="0.7" strokeWidth="1.2" />
      <path d="M88 60 C88 68 86 73 80 75 L62 81 C52 85 48 93 46 105 L38 168 C37 178 40 186 44 190 L50 188 C52 182 52 174 54 166 L62 112 L66 150 C66 166 68 176 70 184 L72 236 L76 312 L93 312 L96 236 L100 198 L104 236 L107 312 L124 312 L128 236 L130 184 C132 176 134 166 134 150 L138 112 L146 166 C148 174 148 182 150 188 L156 190 C160 186 163 178 162 168 L154 105 C152 93 148 85 138 81 L120 75 C114 73 112 68 112 60 Z"
        fill={`url(#g${id})`} />
      <path d="M100 82 V196 M78 100 Q100 110 122 100 M76 126 Q100 134 124 126" stroke="#bff8f8" strokeOpacity="0.25" fill="none" />
      <ellipse className="gt-pulse" cx="104" cy="148" rx="20" ry="9" fill={`url(#o${id})`} filter={`url(#f${id})`} />
      <path d="M88 150 C94 142 106 141 116 145 C122 147 122 152 116 153 C108 155 96 156 88 150 Z" fill={organ} fillOpacity="0.9" />
      <circle cx="94" cy="108" r="6" fill="#ff8aa0" fillOpacity="0.55" className="gt-pulse" />
    </svg>
  )
}

/* ── charts ─────────────────────────────────────────────── */

const axis = { stroke: css('axis'), fontSize: 11, tickLine: false, axisLine: false }

function GlucoseTip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const g = d.past ?? d.future
  return (
    <div className="gt-tip">
      <div className="t">{label}</div>
      <div>{g == null ? 'no reading' : `${Math.round(g)} mg/dL`}{d.future != null && d.past == null ? ' (what happened next)' : ''}</div>
      {d.insulin ? <div>Insulin {d.insulin} U</div> : null}
      {d.carbs ? <div>Carbs {d.carbs} g</div> : null}
    </div>
  )
}

/** CGM trace with the 70-180 target band; "future" points (replay) are dashed and only drawn when asked. */
export function CgmChart({ points, nowTime, showFuture = false, height = 240 }) {
  const id = useId().replace(/:/g, '')
  const data = (points || []).map(p => ({ ...p, past: p.future ? null : p.glucose, future: p.future || p.time === nowTime ? p.glucose : null }))
  const ys = data.map(d => d.glucose).filter(v => v != null)
  const top = Math.max(250, Math.ceil((Math.max(...ys, 0) + 20) / 50) * 50)
  return (
    <>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 20, right: 14, bottom: 0, left: -18 }}>
          <defs>
            <linearGradient id={`a${id}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor={css('in')} stopOpacity={0.28} />
              <stop offset="1" stopColor={css('in')} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={css('grid')} vertical={false} />
          <ReferenceArea y1={70} y2={180} fill={css('in')} fillOpacity={0.06} />
          <ReferenceLine y={180} stroke={css('above')} strokeDasharray="5 4" />
          <ReferenceLine y={70} stroke={css('below')} strokeDasharray="5 4" />
          <XAxis dataKey="time" {...axis} minTickGap={42} tickFormatter={t => (t || '').replace(/^Day \d+ · /, '')} />
          <YAxis {...axis} domain={[0, top]} ticks={[0, 70, 180, top]} width={48} />
          <Tooltip content={<GlucoseTip />} cursor={{ stroke: css('text-3'), strokeDasharray: '3 3' }} />
          <Area dataKey="past" type="monotone" stroke={css('in')} strokeWidth={2} fill={`url(#a${id})`} connectNulls dot={false} isAnimationActive={false} />
          {showFuture && <Line dataKey="future" type="monotone" stroke={css('text-2')} strokeWidth={2} strokeDasharray="5 4" connectNulls dot={false} isAnimationActive={false} />}
          {nowTime && <ReferenceLine x={nowTime} stroke={css('brand-strong')} strokeWidth={1.5} label={{ value: 'NOW', position: 'top', fill: css('brand-strong'), fontSize: 10, fontWeight: 700 }} />}
        </ComposedChart>
      </ResponsiveContainer>
      <div className="gt-legend" style={{ marginTop: 6 }}>
        <span><i className="line" style={{ background: css('in') }} />Glucose (mg/dL)</span>
        {showFuture && <span><i className="line" style={{ background: css('text-2') }} />What actually happened next (replay)</span>}
        <span><i style={{ background: css('in'), opacity: 0.25 }} />Target 70–180</span>
      </div>
    </>
  )
}

/** Insulin or carbs as thin bars on their own axis (never a second y-axis on the glucose chart). */
export function EventBars({ points, dataKey, unit, color, height = 90, nowTime }) {
  const data = (points || []).filter(p => !p.future)
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={css('grid')} vertical={false} />
        <XAxis dataKey="time" hide />
        <YAxis {...axis} width={48} allowDecimals={false} />
        <Tooltip cursor={{ fill: css('brand-soft') }} content={({ active, payload, label }) => active && payload?.length && payload[0].value
          ? <div className="gt-tip"><div className="t">{label}</div>{payload[0].value} {unit}</div> : null} />
        <Bar dataKey={dataKey} fill={color} radius={[4, 4, 0, 0]} maxBarSize={6} isAnimationActive={false} />
        {nowTime && <ReferenceLine x={nowTime} stroke={css('brand-strong')} />}
      </BarChart>
    </ResponsiveContainer>
  )
}

/** Share of readings below / in / above range per day, stacked to 100 %, with a table view. */
export function RangeDays({ days }) {
  const [table, setTable] = useState(false)
  const data = (days || []).map(d => ({ day: `Day ${d.day}`, below: d.tbr_pct, in: d.tir_pct, above: d.tar_pct, n: d.readings }))
  return (
    <>
      <div className="gt-row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
        <div className="gt-legend">
          <span><i style={{ background: css('below') }} />Below 70</span>
          <span><i style={{ background: css('in') }} />In range 70–180</span>
          <span><i style={{ background: css('above') }} />Above 180</span>
        </div>
        <button className="gt-btn small" onClick={() => setTable(t => !t)}>{table ? 'Chart' : 'Table'}</button>
      </div>
      {table ? (
        <table className="gt-table">
          <thead><tr><th>Day</th><th className="num">Below 70</th><th className="num">In range</th><th className="num">Above 180</th><th className="num">Readings</th></tr></thead>
          <tbody>{data.map(d => <tr key={d.day}><td>{d.day}</td><td className="num">{fmt(d.below, 1)}%</td><td className="num">{fmt(d.in, 1)}%</td><td className="num">{fmt(d.above, 1)}%</td><td className="num">{d.n}</td></tr>)}</tbody>
        </table>
      ) : (
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -6 }}>
            <CartesianGrid stroke={css('grid')} vertical={false} />
            <XAxis dataKey="day" {...axis} />
            <YAxis {...axis} domain={[0, 100]} unit="%" width={52} />
            <Tooltip cursor={{ fill: css('brand-soft') }} content={({ active, payload, label }) => active && payload?.length ? (
              <div className="gt-tip"><div className="t">{label}</div>
                <div>Below 70: {fmt(payload[0].payload.below, 1)}%</div>
                <div>In range: {fmt(payload[0].payload.in, 1)}%</div>
                <div>Above 180: {fmt(payload[0].payload.above, 1)}%</div>
                <div className="gt-muted">{payload[0].payload.n} readings</div></div>) : null} />
            <Bar dataKey="below" stackId="r" fill={css('below')} stroke={css('surface-solid')} strokeWidth={2} isAnimationActive={false} maxBarSize={34} />
            <Bar dataKey="in" stackId="r" fill={css('in')} stroke={css('surface-solid')} strokeWidth={2} isAnimationActive={false} maxBarSize={34} />
            <Bar dataKey="above" stackId="r" fill={css('above')} stroke={css('surface-solid')} strokeWidth={2} radius={[4, 4, 0, 0]} isAnimationActive={false} maxBarSize={34} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </>
  )
}

/** Time-of-day profile: median with 25-75 % and 10-90 % bands (ambulatory glucose profile style). */
export function HourlyProfile({ profile }) {
  const data = (profile || []).map(h => ({
    hour: `${String(h.hour).padStart(2, '0')}:00`, n: h.n, p50: h.p50 ?? null,
    outer: h.p10 != null ? [h.p10, h.p90] : null, inner: h.p25 != null ? [h.p25, h.p75] : null,
  }))
  return (
    <>
      <ResponsiveContainer width="100%" height={230}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -18 }}>
          <CartesianGrid stroke={css('grid')} vertical={false} />
          <ReferenceLine y={180} stroke={css('above')} strokeDasharray="5 4" />
          <ReferenceLine y={70} stroke={css('below')} strokeDasharray="5 4" />
          <XAxis dataKey="hour" {...axis} interval={2} />
          <YAxis {...axis} width={48} domain={[0, 'auto']} />
          <Tooltip cursor={{ stroke: css('text-3'), strokeDasharray: '3 3' }} content={({ active, payload, label }) => active && payload?.length ? (
            <div className="gt-tip"><div className="t">{label}</div>
              {payload[0].payload.p50 == null ? <div>too few readings</div> : <>
                <div>Median {Math.round(payload[0].payload.p50)} mg/dL</div>
                <div>Middle half {payload[0].payload.inner.map(Math.round).join('–')}</div>
                <div>10–90 % {payload[0].payload.outer.map(Math.round).join('–')}</div></>}
              <div className="gt-muted">{payload[0].payload.n} readings</div></div>) : null} />
          <Area dataKey="outer" stroke="none" fill={css('in')} fillOpacity={0.12} isAnimationActive={false} connectNulls />
          <Area dataKey="inner" stroke="none" fill={css('in')} fillOpacity={0.28} isAnimationActive={false} connectNulls />
          <Line dataKey="p50" stroke={css('brand-strong')} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="gt-legend" style={{ marginTop: 6 }}>
        <span><i className="line" style={{ background: css('brand-strong') }} />Median</span>
        <span><i style={{ background: css('in'), opacity: 0.45 }} />Middle half (25–75 %)</span>
        <span><i style={{ background: css('in'), opacity: 0.18 }} />10–90 %</span>
      </div>
    </>
  )
}
