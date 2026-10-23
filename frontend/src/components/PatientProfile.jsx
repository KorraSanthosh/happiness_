import { getInitials, getAvatarColor, getRiskClass } from '../utils'

export default function PatientProfile({ twin }) {
  if (!twin?.patient_info) return null
  const p = twin.patient_info
  const [bg, fg] = getAvatarColor(p.patient_id)
  const rClass = getRiskClass(p.risk_level)

  return (
    <div className="fade-in">
      {/* Header card */}
      <div className="patient-profile-header">
        <div className="profile-avatar-large" style={{ background: bg, color: fg }}>
          {getInitials(p.name)}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div className="profile-name">{p.name}</div>
            <span className={`risk-badge ${rClass}`}>{p.risk_level} risk</span>
          </div>
          <div className="profile-id">{p.patient_id}</div>
          <div className="profile-tags" style={{ marginTop: 6 }}>
            {p.diabetes_type && <span className="tag">🩸 {p.diabetes_type}</span>}
            <span className="tag">{p.gender}</span>
            <span className="tag">{p.age} years</span>
            {p.bmi && <span className="tag">BMI {p.bmi?.toFixed(1)}</span>}
          </div>
        </div>
      </div>

      {/* Demographics */}
      <div style={{ fontSize: 'var(--fs-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 8 }}>Demographics</div>
      <div className="info-grid" style={{ marginBottom: 18 }}>
        {[
          { label: 'Age', value: `${p.age} years` },
          { label: 'Weight', value: p.weight_kg ? `${p.weight_kg} kg` : '—' },
          { label: 'Height', value: p.height_cm ? `${p.height_cm} cm` : '—' },
          { label: 'BMI', value: p.bmi ? p.bmi.toFixed(1) : '—' },
        ].map(item => (
          <div key={item.label} className="info-item">
            <div className="info-label">{item.label}</div>
            <div className="info-value">{item.value}</div>
          </div>
        ))}
      </div>

      {/* Lab results */}
      {p.lab_results?.length > 0 && (
        <>
          <div style={{ fontSize: 'var(--fs-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 8 }}>Lab Results</div>
          <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-glass)', borderRadius: 'var(--r-lg)', padding: '4px 16px', marginBottom: 18 }}>
            {p.lab_results.map((lab, i) => {
              const abnormal = lab.value > 5.7 && lab.test_name === 'HbA1c'
                || lab.value > 100 && lab.test_name === 'Fasting Glucose'
              return (
                <div key={i} className="lab-item">
                  <span className="lab-name">{lab.test_name}</span>
                  <div className="lab-value">
                    {abnormal && <div className="abnormal-dot" />}
                    <span className="lab-val" style={{ color: abnormal ? 'var(--risk-high)' : undefined }}>{lab.value}</span>
                    <span className="lab-unit">{lab.unit}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* Diagnoses */}
      {p.diagnoses?.length > 0 && (
        <>
          <div style={{ fontSize: 'var(--fs-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 8 }}>Active Diagnoses</div>
          <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-glass)', borderRadius: 'var(--r-lg)', padding: '4px 16px' }}>
            {p.diagnoses.map((dx, i) => (
              <div key={i} className="dx-item">
                <span className="dx-code">{dx.code}</span>
                <div>
                  <div className="dx-desc">{dx.description}</div>
                  {dx.diagnosed_date && <div className="dx-date">Since {dx.diagnosed_date}</div>}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
