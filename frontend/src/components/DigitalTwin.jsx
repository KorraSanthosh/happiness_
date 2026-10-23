import { Heart, Activity, Moon, Footprints, TrendingUp, TrendingDown, Minus, Zap } from 'lucide-react'
import { GlucoseChart, SensorChart } from './Charts'
import { formatGlucose, getTrendClass } from '../utils'

function MetricTile({ icon: Icon, value, unit, label, iconBg, iconColor }) {
  return (
    <div className="metric-tile">
      <div className="metric-tile-icon" style={{ background: iconBg }}>
        <Icon style={{ color: iconColor }} />
      </div>
      <div>
        <span className="metric-tile-value">{value ?? '—'}</span>
        {unit && <span className="metric-tile-unit">{unit}</span>}
      </div>
      <div className="metric-tile-label">{label}</div>
    </div>
  )
}

function DerivedFeature({ label, value, accent }) {
  return (
    <div style={{
      padding: '10px 14px',
      background: 'rgba(255,255,255,0.02)',
      border: '1px solid var(--border-glass)',
      borderRadius: 'var(--r-md)',
      borderLeft: `2px solid ${accent}`,
    }}>
      <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 3, fontWeight: 500 }}>{label}</div>
      <div style={{ fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{value ?? '—'}</div>
    </div>
  )
}

export default function DigitalTwinView({ twin, readings = [] }) {
  if (!twin) return null

  const features = twin.derived_features || {}
  const latest   = twin.latest_readings?.[0] || {}
  const trend    = twin.trend_summary?.glucose_trend || 'stable'
  const tClass   = getTrendClass(trend)
  const TrendIcon = trend === 'rising' ? TrendingUp : trend === 'falling' ? TrendingDown : Minus

  const latestGlucose = latest.glucose_mgdl ? Math.round(latest.glucose_mgdl) : null

  return (
    <div className="fade-in">
      {/* Quick metrics */}
      <div className="metric-grid">
        <MetricTile
          icon={Zap}
          value={latestGlucose}
          unit="mg/dL"
          label="Current Glucose"
          iconBg="rgba(59,130,246,0.12)"
          iconColor="var(--accent-light)"
        />
        <MetricTile
          icon={Heart}
          value={latest.heart_rate_bpm ? Math.round(latest.heart_rate_bpm) : null}
          unit="bpm"
          label="Heart Rate"
          iconBg="rgba(239,68,68,0.1)"
          iconColor="#f87171"
        />
        <MetricTile
          icon={Moon}
          value={latest.sleep_hours_last_night ? latest.sleep_hours_last_night.toFixed(1) : null}
          unit="hrs"
          label="Sleep"
          iconBg="rgba(167,139,250,0.1)"
          iconColor="var(--purple)"
        />
        <MetricTile
          icon={Footprints}
          value={latest.steps_last_hour != null ? latest.steps_last_hour : null}
          unit="steps"
          label="Steps (1h)"
          iconBg="rgba(6,182,212,0.1)"
          iconColor="var(--teal)"
        />
      </div>

      {/* Glucose chart */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <div style={{ fontSize: 'var(--fs-sm)', fontWeight: 600, color: 'var(--text-white)', display: 'flex', alignItems: 'center', gap: 7 }}>
          <Activity style={{ width: 15, height: 15, color: 'var(--accent-light)' }} />
          12-Hour Glucose Timeline
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className={`trend-badge ${tClass}`}>
            <TrendIcon style={{ width: 11, height: 11 }} />
            {trend}
          </span>
          {latestGlucose > 180 && (
            <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--risk-high)', fontWeight: 600 }}>⚠ Above threshold</span>
          )}
        </div>
      </div>
      <GlucoseChart readings={readings} />

      <div className="section-divider">
        <div className="section-divider-line" />
        <span className="section-divider-label">Sensor Trends</span>
        <div className="section-divider-line" />
      </div>

      {/* HR + Activity chart */}
      <div style={{ fontSize: 'var(--fs-sm)', fontWeight: 600, color: 'var(--text-white)', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 7 }}>
        <Heart style={{ width: 15, height: 15, color: '#f87171' }} />
        Heart Rate & Activity
        <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', fontWeight: 400 }}>· last 8 hours</span>
      </div>
      <SensorChart readings={readings} />

      {/* Derived features */}
      <div className="section-divider">
        <div className="section-divider-line" />
        <span className="section-divider-label">Engineered Features</span>
        <div className="section-divider-line" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        <DerivedFeature label="Δ Glucose 1h"
          value={features.glucose_change_1h != null ? `${features.glucose_change_1h > 0 ? '+' : ''}${features.glucose_change_1h?.toFixed(1)} mg/dL` : null}
          accent="var(--accent-light)" />
        <DerivedFeature label="Rolling Mean 2h"
          value={features.glucose_rolling_mean_2h != null ? `${features.glucose_rolling_mean_2h?.toFixed(1)} mg/dL` : null}
          accent="var(--teal)" />
        <DerivedFeature label="Variability (σ)"
          value={features.glucose_rolling_std_2h != null ? `${features.glucose_rolling_std_2h?.toFixed(2)}` : null}
          accent="var(--purple)" />
      </div>
    </div>
  )
}
