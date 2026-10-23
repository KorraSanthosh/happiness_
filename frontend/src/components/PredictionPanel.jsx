import { Brain, AlertTriangle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import ModelPerformance from './ModelPerformance'
import { FeatureChart } from './Charts'
import { getRiskClass, getTrendClass } from '../utils'

export default function PredictionPanel({ patientId, prediction, loading, error, onPredict }) {
  const rClass = getRiskClass(prediction?.risk_category)
  const trend  = prediction?.recent_glucose_trend
  const TrendIcon = trend === 'rising' ? TrendingUp : trend === 'falling' ? TrendingDown : Minus

  return (
    <div className="fade-in">
      {/* CTA button */}
      <button
        id="predict-btn"
        className="predict-btn"
        onClick={onPredict}
        disabled={loading || !patientId}
      >
        {loading
          ? <><div className="spinner" /> Running model…</>
          : <><Brain style={{ width: 17, height: 17 }} /> Predict Next 2 Hours</>
        }
      </button>

      {/* Error */}
      {error && (
        <div className="error-box" style={{ marginBottom: 14 }}>
          <AlertTriangle style={{ width: 15, height: 15, flexShrink: 0 }} />
          {error}
        </div>
      )}

      {/* Results */}
      {prediction && !loading && (
        <div className="fade-in">
          {/* Risk circle */}
          <div className={`risk-display ${rClass}`}>
            <div className={`risk-circle ${rClass}`}>
              <div className="risk-pct" style={{ color: rClass === 'low' ? 'var(--risk-low)' : rClass === 'moderate' ? 'var(--risk-mod)' : 'var(--risk-high)' }}>
                {(prediction.spike_probability * 100).toFixed(0)}%
              </div>
              <div className="risk-label-small" style={{ color: rClass === 'low' ? 'var(--risk-low)' : rClass === 'moderate' ? 'var(--risk-mod)' : 'var(--risk-high)' }}>
                spike risk
              </div>
            </div>
            <div>
              <div className="risk-info-label">2-Hour Forecast</div>
              <div className={`risk-category-text ${rClass}`}>
                {(prediction.risk_category || '').toUpperCase()} RISK
              </div>
              <div className="prediction-glucose">
                Predicted: ~{prediction.predicted_glucose_mgdl} mg/dL
              </div>
              {trend && (
                <div style={{ marginTop: 6 }}>
                  <span className={`trend-badge ${getTrendClass(trend)}`}>
                    <TrendIcon style={{ width: 11, height: 11 }} />
                    {trend}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Feature importances */}
          {prediction.feature_importances?.length > 0 && (
            <>
              <div style={{ fontSize: 'var(--fs-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 10 }}>
                Contributing Factors
              </div>
              <FeatureChart features={prediction.feature_importances} />
            </>
          )}

          {/* Explanation */}
          {prediction.explanation && (
            <div className="explanation-box">
              💡 {prediction.explanation}
            </div>
          )}

          {/* Disclaimer */}
          <div style={{
            marginTop: 14,
            padding: '8px 10px',
            background: 'rgba(245,158,11,0.06)',
            border: '1px solid rgba(245,158,11,0.15)',
            borderRadius: 'var(--r-md)',
            fontSize: 10,
            color: '#92704a',
            lineHeight: 1.5,
            display: 'flex', gap: 6, alignItems: 'flex-start'
          }}>
            <AlertTriangle style={{ width: 11, height: 11, flexShrink: 0, marginTop: 2, color: '#f59e0b' }} />
            {prediction.experimental_status || 'EXPERIMENTAL — Not for clinical use'}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!prediction && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '32px 16px', color: 'var(--text-muted)' }}>
          <Brain style={{ width: 36, height: 36, margin: '0 auto 12px', opacity: 0.3 }} />
          <div style={{ fontSize: 'var(--fs-sm)' }}>Run the model to see a 2-hour glucose spike prediction and feature breakdown.</div>
        </div>
      )}

      {/* Model performance — always visible below prediction */}
      <div className="divider" style={{ margin: '16px 0 12px' }} />
      <ModelPerformance />
    </div>
  )
}
