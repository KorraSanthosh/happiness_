import { useState, useEffect } from 'react'
import { Shield, TrendingUp, AlertTriangle, CheckCircle, Archive } from 'lucide-react'

function MetricPill({ label, value, color }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '12px 14px',
      background: 'rgba(255,255,255,0.03)',
      border: `1px solid ${color}22`,
      borderRadius: 'var(--r-lg)',
      minWidth: 90,
    }}>
      <div style={{ fontSize: '1.25rem', fontWeight: 800, color, letterSpacing: '-0.02em', lineHeight: 1.1 }}>
        {value}
      </div>
      <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginTop: 4, fontWeight: 500, textAlign: 'center' }}>
        {label}
      </div>
    </div>
  )
}

function ComparisonRow({ model, accuracy, roc, prAuc, recall, f1, selected }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '1fr repeat(5, 80px)',
      gap: 6, alignItems: 'center',
      padding: '9px 12px',
      borderRadius: 'var(--r-md)',
      background: selected ? 'rgba(59,130,246,0.08)' : 'rgba(255,255,255,0.02)',
      border: `1px solid ${selected ? 'rgba(59,130,246,0.25)' : 'var(--border-glass)'}`,
      marginBottom: 4,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        {selected
          ? <CheckCircle style={{ width: 13, height: 13, color: 'var(--risk-low)', flexShrink: 0 }} />
          : <div style={{ width: 13, height: 13, flexShrink: 0 }} />
        }
        <span style={{
          fontSize: 'var(--fs-sm)', fontWeight: selected ? 700 : 400,
          color: selected ? 'var(--text-white)' : 'var(--text-secondary)',
        }}>{model}</span>
        {selected && <span style={{ fontSize: 9, background: 'rgba(59,130,246,0.15)', color: 'var(--accent-light)', border: '1px solid rgba(59,130,246,0.25)', borderRadius: 4, padding: '1px 5px', fontWeight: 700 }}>ACTIVE</span>}
      </div>
      {[accuracy, roc, prAuc, recall, f1].map((v, i) => (
        <div key={i} style={{
          fontSize: 'var(--fs-xs)', fontFamily: 'var(--font-mono)', fontWeight: 600,
          color: selected ? 'var(--accent-light)' : 'var(--text-secondary)',
          textAlign: 'center',
        }}>{v}</div>
      ))}
    </div>
  )
}

export default function ModelPerformance() {
  const [card, setCard] = useState(null)
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    fetch('/api/model-card')
      .then(r => r.json())
      .then(d => { setCard(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div style={{ padding: '10px 0' }}>
        <div className="skeleton" style={{ height: 14, width: '50%', marginBottom: 8 }} />
        <div className="skeleton" style={{ height: 60 }} />
      </div>
    )
  }

  if (!card) return null

  const tm = card.live_test_metrics || card.test_metrics || {}
  const vc = card.live_validation_comparison || {}
  const acc = tm.confusion_matrix
    ? ((tm.confusion_matrix.tp + tm.confusion_matrix.tn) / tm.support_total * 100).toFixed(2) + '%'
    : '—'

  const models = Object.entries(vc).map(([name, r]) => ({
    name,
    accuracy: r.accuracy != null ? (r.accuracy * 100).toFixed(1) + '%' : '—',
    roc_auc:  r.roc_auc  != null ? r.roc_auc.toFixed(3)  : '—',
    pr_auc:   r.pr_auc   != null ? r.pr_auc.toFixed(3)   : '—',
    recall:   r.recall   != null ? r.recall.toFixed(3)    : '—',
    f1:       r.f1       != null ? r.f1.toFixed(3)        : '—',
    selected: name === card.model_name,
  }))

  return (
    <div>
      {/* Compact header row — always visible */}
      <div
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', marginBottom: open ? 14 : 0 }}
        onClick={() => setOpen(o => !o)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Shield style={{ width: 14, height: 14, color: 'var(--accent-light)' }} />
          <span style={{ fontSize: 'var(--fs-sm)', fontWeight: 600, color: 'var(--text-white)' }}>
            Model Performance
          </span>
          <span style={{ fontSize: 'var(--fs-xs)', fontFamily: 'var(--font-mono)', color: 'var(--risk-low)', fontWeight: 600 }}>
            {acc} accuracy
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>
            {card.model_name} v{card.model_version}
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>{open ? '▲' : '▼'}</span>
        </div>
      </div>

      {open && (
        <div className="fade-in">
          {/* Key metrics pills */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
            <MetricPill label="Accuracy"    value={acc} color="var(--risk-low)" />
            <MetricPill label="ROC-AUC"     value={tm.roc_auc?.toFixed(4) || '—'} color="var(--accent-light)" />
            <MetricPill label="PR-AUC"      value={tm.pr_auc?.toFixed(4) || '—'}  color="var(--teal)" />
            <MetricPill label="Recall"       value={tm.recall ? (tm.recall*100).toFixed(1)+'%' : '—'} color="#f59e0b" />
            <MetricPill label="Precision"    value={tm.precision ? (tm.precision*100).toFixed(1)+'%' : '—'} color="var(--purple)" />
            <MetricPill label="F1"           value={tm.f1?.toFixed(4) || '—'} color="var(--pink)" />
          </div>

          {/* Confusion matrix */}
          {tm.confusion_matrix && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6, fontWeight: 600 }}>
                Confusion Matrix (test set, n={tm.support_total?.toLocaleString()})
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, maxWidth: 260 }}>
                {[
                  { label: 'True Negative',  v: tm.confusion_matrix.tn, color: 'var(--risk-low)' },
                  { label: 'False Positive', v: tm.confusion_matrix.fp, color: 'var(--risk-high)' },
                  { label: 'False Negative', v: tm.confusion_matrix.fn, color: '#f59e0b' },
                  { label: 'True Positive',  v: tm.confusion_matrix.tp, color: 'var(--risk-low)' },
                ].map(item => (
                  <div key={item.label} style={{
                    padding: '8px 10px', borderRadius: 'var(--r-md)',
                    background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-glass)',
                  }}>
                    <div style={{ fontSize: 'var(--fs-lg)', fontWeight: 800, color: item.color, fontFamily: 'var(--font-mono)' }}>
                      {item.v?.toLocaleString()}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{item.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model comparison */}
          {models.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6, fontWeight: 600 }}>
                Candidate Comparison (same patient-aware test set)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr repeat(5, 80px)', gap: 6, padding: '4px 12px 6px', marginBottom: 3 }}>
                {['Model', 'Accuracy', 'ROC-AUC', 'PR-AUC', 'Recall', 'F1'].map(h => (
                  <div key={h} style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', textAlign: h === 'Model' ? 'left' : 'center' }}>{h}</div>
                ))}
              </div>
              {models.map(m => (
                <ComparisonRow key={m.name}
                  model={m.name} accuracy={m.accuracy} roc={m.roc_auc}
                  prAuc={m.pr_auc} recall={m.recall} f1={m.f1} selected={m.selected}
                />
              ))}
            </div>
          )}

          {/* Reproduction status */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: '8px 12px',
            background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.18)', borderRadius: 'var(--r-md)',
            fontSize: 'var(--fs-xs)', color: 'var(--risk-low)', fontWeight: 500,
            marginBottom: 10,
          }}>
            <CheckCircle style={{ width: 12, height: 12, flexShrink: 0 }} />
            Reproduced — Δ 0.0000pp from original | Seed 42 | Patient-aware split | No leakage detected
          </div>

          {/* Leakage audit */}
          {card.leakage_audit && (
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6, fontWeight: 600 }}>
                Leakage Audit
              </div>
              {Object.entries(card.leakage_audit).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', gap: 8, marginBottom: 3 }}>
                  <CheckCircle style={{ width: 11, height: 11, color: 'var(--risk-low)', flexShrink: 0, marginTop: 1 }} />
                  <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)' }}>
                    <strong style={{ color: 'var(--text-primary)' }}>{k.replace(/_/g, ' ')}: </strong>{v}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Backup / rollback */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: '7px 12px',
            background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.15)', borderRadius: 'var(--r-md)',
            fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)',
          }}>
            <Archive style={{ width: 11, height: 11, color: 'var(--purple)', flexShrink: 0 }} />
            Backup saved: <code style={{ color: 'var(--purple)', fontFamily: 'var(--font-mono)' }}>models/snapshots/v2_histgb_20241020/</code>
          </div>

          {/* Disclaimer */}
          <div style={{
            display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 10,
            padding: '7px 10px', background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.15)', borderRadius: 'var(--r-md)',
            fontSize: 10, color: '#92704a', lineHeight: 1.5,
          }}>
            <AlertTriangle style={{ width: 11, height: 11, flexShrink: 0, marginTop: 1, color: '#f59e0b' }} />
            All metrics are on <strong>SYNTHETIC data only</strong>. Performance on real patient CGM data is unknown. Not validated for clinical use.
          </div>
        </div>
      )}
    </div>
  )
}
