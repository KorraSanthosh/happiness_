import { Users, AlertTriangle, Activity, Heart } from 'lucide-react'
import { useStats } from '../hooks/useApi'

function StatCard({ icon: Icon, value, label, iconBg, iconColor, delta, deltaLabel }) {
  return (
    <div className="stat-card fade-in">
      <div className="stat-icon" style={{ background: iconBg }}>
        <Icon style={{ color: iconColor }} />
      </div>
      <div>
        <div className="stat-value">{value ?? '—'}</div>
        <div className="stat-label">{label}</div>
        {delta !== undefined && (
          <div className="stat-delta" style={{ color: iconColor }}>{delta} {deltaLabel}</div>
        )}
      </div>
    </div>
  )
}

export default function StatsRow() {
  const { stats, loading } = useStats()

  if (loading) {
    return (
      <div className="stats-row">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="stat-card">
            <div className="skeleton" style={{ width: 46, height: 46, borderRadius: 12, flexShrink: 0 }} />
            <div style={{ flex: 1 }}>
              <div className="skeleton" style={{ height: 22, width: '50%', marginBottom: 6 }} />
              <div className="skeleton" style={{ height: 11, width: '70%' }} />
            </div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="stats-row">
      <StatCard
        icon={Users}
        value={stats?.total_patients}
        label="Total Patients"
        iconBg="rgba(59,130,246,0.15)"
        iconColor="var(--accent-light)"
      />
      <StatCard
        icon={AlertTriangle}
        value={stats?.high_risk_count}
        label="High Risk"
        iconBg="rgba(239,68,68,0.12)"
        iconColor="var(--risk-high)"
        delta={`${stats?.moderate_risk_count ?? 0} moderate`}
        deltaLabel=""
      />
      <StatCard
        icon={Activity}
        value={stats?.avg_glucose ? `${stats.avg_glucose}` : null}
        label="Avg Glucose mg/dL"
        iconBg="rgba(6,182,212,0.12)"
        iconColor="var(--teal)"
      />
      <StatCard
        icon={Heart}
        value={stats?.low_risk_count}
        label="Low Risk"
        iconBg="rgba(34,197,94,0.12)"
        iconColor="var(--risk-low)"
      />
    </div>
  )
}
