import { Activity, Dna } from 'lucide-react'
import { useState, useEffect } from 'react'
import { getPatients } from '../services/api'
import { getRiskClass, getInitials, getAvatarColor } from '../utils'

export default function Sidebar({ selectedId, onSelect }) {
  const [patients, setPatients] = useState([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getPatients({ limit: 120, offset: 0 }).then(d => {
      setPatients(d.patients || [])
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const filtered = patients.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    p.patient_id.toLowerCase().includes(search.toLowerCase())
  )

  const highRisk = patients.filter(p => p.risk_level === 'high').length

  return (
    <div className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="logo-row">
          <div className="logo-icon"><Dna /></div>
          <div>
            <div className="logo-title">GlucoTwin</div>
          </div>
        </div>
        <div className="logo-subtitle">Predictive Healthcare AI</div>
      </div>

      {/* Search */}
      <div className="search-container">
        <div className="search-wrapper">
          <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
          </svg>
          <input
            className="search-input"
            id="patient-search"
            placeholder="Search patients…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Alerts header */}
      <div className="sidebar-section-header">
        {search
          ? `${filtered.length} result${filtered.length !== 1 ? 's' : ''}`
          : `Patients · ${highRisk} high risk`}
      </div>

      {/* List */}
      <div className="patient-list" id="patient-list">
        {loading
          ? [...Array(8)].map((_, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 10px', marginBottom: 2 }}>
                <div className="skeleton" style={{ width: 34, height: 34, borderRadius: 8, flexShrink: 0 }} />
                <div style={{ flex: 1 }}>
                  <div className="skeleton" style={{ height: 12, width: '70%', marginBottom: 5 }} />
                  <div className="skeleton" style={{ height: 10, width: '40%' }} />
                </div>
              </div>
            ))
          : filtered.map(p => {
              const rClass = getRiskClass(p.risk_level)
              const [bg, fg] = getAvatarColor(p.patient_id)
              return (
                <div
                  key={p.patient_id}
                  id={`patient-${p.patient_id}`}
                  className={`patient-item${selectedId === p.patient_id ? ' active' : ''}`}
                  onClick={() => onSelect(p.patient_id)}
                >
                  <div className="patient-avatar" style={{ background: bg, color: fg }}>
                    {getInitials(p.name)}
                  </div>
                  <div className="patient-meta">
                    <div className="patient-name">{p.name}</div>
                    <div className="patient-id">{p.patient_id} · {p.age}y</div>
                  </div>
                  <span className={`risk-badge ${rClass}`}>{p.risk_level}</span>
                </div>
              )
            })}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-footer-count">
          <Activity style={{ width: 12, height: 12, display: 'inline', marginRight: 4, color: 'var(--risk-low)', verticalAlign: 'middle' }} />
          {patients.length} virtual patients · synthetic data
        </div>
      </div>
    </div>
  )
}
