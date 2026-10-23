/** Formatting & style helpers for GlucoTwin AI */

export function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })
}

export function formatGlucose(v) {
  return v != null ? `${Math.round(v)} mg/dL` : '—'
}

export function getRiskClass(level) {
  if (!level) return 'low'
  const l = level.toLowerCase()
  if (l === 'high') return 'high'
  if (l === 'moderate' || l === 'medium') return 'moderate'
  return 'low'
}

export function getTrendClass(trend) {
  if (!trend) return 'trend-stable'
  const t = trend.toLowerCase()
  if (t === 'rising' || t === 'increasing') return 'trend-rising'
  if (t === 'falling' || t === 'decreasing') return 'trend-falling'
  return 'trend-stable'
}

export function getInitials(name = '') {
  return name.split(' ').map(n => n[0]).slice(0, 2).join('').toUpperCase()
}

const AVATAR_PALETTES = [
  ['rgba(59,130,246,0.25)', '#93c5fd'],
  ['rgba(6,182,212,0.22)', '#67e8f9'],
  ['rgba(167,139,250,0.22)', '#c4b5fd'],
  ['rgba(245,158,11,0.2)', '#fcd34d'],
  ['rgba(34,197,94,0.18)', '#86efac'],
  ['rgba(236,72,153,0.2)', '#f9a8d4'],
  ['rgba(239,68,68,0.18)', '#fca5a5'],
  ['rgba(16,185,129,0.2)', '#6ee7b7'],
]

export function getAvatarColor(id = '') {
  const n = parseInt(id.replace(/\D/g, ''), 10) || 0
  return AVATAR_PALETTES[n % AVATAR_PALETTES.length]
}
