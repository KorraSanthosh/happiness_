import { useEffect, useState } from 'react'
import {
  User, Calendar, Droplet, Activity, Database, ClipboardList, Radio, UtensilsCrossed, Info, LineChart, ChevronDown,
  HeartPulse, Footprints, Syringe, Pill, Moon, Dna, TrendingUp, TrendingDown, ArrowRight, Shield, CheckCircle2, XCircle,
  SkipBack, SkipForward, ChevronLeft, ChevronRight, Target, FlaskConical,
} from 'lucide-react'
import { getV7Stream } from '../../services/api'
import { useStudio, riskOf, contextToScenario } from '../store'
import { Card, Stat, Segmented, Gauge, BodyTwin, CgmChart, EventBars, Banner, Level, fmt, css } from '../ui'

const CONDITIONS = new Set(['Hypertension', 'Hyperlipidemia', 'Smoker', 'Alcohol use'])

export const initials = d => (d?.display_id ? `#${d.display_id}`.slice(0, 5) : '—')

export function labsMap(detail) {
  const m = {}
  for (const l of detail?.ehr?.labs || []) m[l.name] = l
  return m
}

/* ── moment selector (the replay "now") ───────────────────── */
export function MomentBar() {
  const { detail, bin, setBin, overview } = useStudio()
  if (!detail || bin == null) return null
  const go = d => setBin(b => Math.min(detail.n_bins - 1, Math.max(detail.min_bin, b + d)))
  return (
    <Card className="gt-noprint" style={{ padding: '12px 16px' }}>
      <div className="gt-row" style={{ flexWrap: 'wrap', gap: 8 }}>
        <div style={{ minWidth: 150 }}>
          <div className="gt-small gt-muted">Replay moment ("now")</div>
          <div style={{ fontWeight: 700 }}>{overview?.now ?? '…'}</div>
        </div>
        <button className="gt-btn small" onClick={() => go(-4)} aria-label="Back one hour"><SkipBack />1 h</button>
        <button className="gt-btn small" onClick={() => go(-1)} aria-label="Back 15 minutes"><ChevronLeft />15 min</button>
        <button className="gt-btn small" onClick={() => go(1)} aria-label="Forward 15 minutes">15 min<ChevronRight /></button>
        <button className="gt-btn small" onClick={() => go(4)} aria-label="Forward one hour">1 h<SkipForward /></button>
        {detail.default_is_example_warning && (
          <button className="gt-btn small" onClick={() => setBin(detail.default_bin)}><Target />Example early warning</button>
        )}
        <input className="gt-range" style={{ flex: 1, minWidth: 180 }} type="range" min={detail.min_bin} max={detail.n_bins - 1}
          value={bin} onChange={e => setBin(Number(e.target.value))} aria-label="Replay moment" />
      </div>
    </Card>
  )
}

/* ── patient selector ─────────────────────────────────────── */
function SelectPatient() {
  const { patients, pid, setPid, detail } = useStudio()
  const cur = patients.find(p => p.patient_id === pid)
  return (
    <Card title="Select patient" icon={User} style={{ padding: '16px 18px' }}>
      <div style={{ position: 'relative' }}>
        <div className="gt-row" style={{ padding: '10px 12px', borderRadius: 14, border: `1px solid ${css('border-strong')}`, background: css('surface-solid') }}>
          <div className="gt-user-av" style={{ width: 44, height: 44, fontSize: '0.7rem', flexShrink: 0 }}>{initials(detail)}</div>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 700, whiteSpace: 'nowrap' }}>{cur ? `Patient #${cur.display_id}` : 'Loading…'}</div>
            <div className="gt-small gt-muted" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{cur ? `${cur.diabetes_type === 'T2D' ? 'Type 2' : 'Type 1'} · ${cur.dataset}${cur.dataset === 'Your upload' ? '' : ' · real record'}` : ''}</div>
          </div>
          <ChevronDown style={{ marginLeft: 'auto', width: 18, color: css('text-2') }} />
        </div>
        <select aria-label="Select patient" value={pid ?? ''} onChange={e => setPid(e.target.value)}
          style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', width: '100%' }}>
          {patients.map(p => (
            <option key={p.patient_id} value={p.patient_id}>
              #{p.display_id} · {p.diabetes_type === 'T2D' ? 'Type 2' : 'Type 1'} · {p.dataset}{p.age ? ` · ${Math.round(p.age)} y` : ''}
            </option>
          ))}
        </select>
      </div>
    </Card>
  )
}

/* ── tabs ─────────────────────────────────────────────────── */
function HealthRecords() {
  const { detail, patients, pid } = useStudio()
  const L = labsMap(detail)
  const cur = patients.find(p => p.patient_id === pid)
  const e = detail?.ehr
  const profile = [
    ['Age', L.Age ? `${Math.round(L.Age.value)}` : '—'], ['Sex', e?.sex ?? '—'],
    ['Diabetes type', detail?.diabetes_type === 'T2D' ? 'Type 2' : 'Type 1'],
    ['HbA1c', L.HbA1c ? `${L.HbA1c.value}%` : '—'], ['BMI', L.BMI ? `${L.BMI.value}` : '—'],
    ['Duration', L['Diabetes duration'] ? `${L['Diabetes duration'].value} y` : '—'],
  ]
  const [all, setAll] = useState(false)
  const allLabs = (e?.labs || []).filter(l => !['Age', 'BMI', 'Diabetes duration'].includes(l.name))
  const labs = all ? allLabs : allLabs.slice(0, 6)
  return (
    <Card title="Patient profile" icon={User}>
      <div style={{ display: 'grid', gridTemplateColumns: '92px minmax(0,1fr)', gap: 16, alignItems: 'start' }}>
        <div className="gt-user-av" style={{ width: 92, height: 92, borderRadius: 22, fontSize: '1.05rem' }}>{initials(detail)}</div>
        <div className="gt-kv">{profile.map(([k, v]) => <div key={k}><div className="k">{k}</div><div className="v">{v}</div></div>)}</div>
      </div>
      {!e?.available && (
        <div style={{ marginTop: 14 }}><Banner>{detail?.uploaded ? 'No clinical details were given with this upload (add your age and HbA1c on the My data page to include them).'
          : `This dataset (${cur?.dataset}) has no clinical record.`} The twin knows the diabetes type and relies on the CGM, insulin and meal stream.</Banner></div>
      )}
      {e?.available && (
        <div className="gt-two" style={{ marginTop: 16 }}>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Medical history</div>
            <div className="gt-list">
              {L['Diabetes duration'] && <div><Calendar style={{ width: 16, color: css('brand') }} />Diabetes for {L['Diabetes duration'].value} years</div>}
              {(e.conditions_and_drugs || []).map(c => {
                const Icon = CONDITIONS.has(c) ? HeartPulse : Pill
                return <div key={c}><Icon style={{ width: 16, color: css('brand') }} />{CONDITIONS.has(c) ? c : `On ${c.toLowerCase()}`}</div>
              })}
              {e.complications != null && <div><Shield style={{ width: 16, color: css('brand') }} />{e.complications ? `${e.complications} diabetic complication${e.complications === 1 ? '' : 's'}` : 'No recorded complications'}</div>}
            </div>
          </div>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Lab results <span className="gt-small gt-muted">({detail?.uploaded ? 'as you entered them' : 'at enrolment'})</span></div>
            <div className="gt-list">
              {labs.map(l => <div key={l.name} className="gt-small"><span>{l.name}</span><span className="gt-right gt-mono" style={{ fontWeight: 700 }}>{l.value}</span><span className="gt-muted" style={{ width: 64 }}>{l.unit}</span></div>)}
            </div>
            {allLabs.length > 6 && <button className="gt-btn small" style={{ marginTop: 6 }} onClick={() => setAll(a => !a)}>{all ? 'Show fewer' : `Show all ${allLabs.length}`}</button>}
          </div>
        </div>
      )}
    </Card>
  )
}

const RANGES = [['24', '24H'], ['168', '7D'], ['336', '14D']]

export function GlucoseCard({ title = 'Recent glucose readings (CGM)', showFutureToggle = false, events = false, defaultHours = '24' }) {
  const { pid, bin, overview } = useStudio()
  const [hours, setHours] = useState(defaultHours)
  const [future, setFuture] = useState(false)
  const [stream, setStream] = useState(null)
  useEffect(() => {
    if (!pid || bin == null) return
    let live = true
    getV7Stream(pid, bin, Number(hours), future).then(s => live && setStream(s)).catch(() => live && setStream(null))
    return () => { live = false }
  }, [pid, bin, hours, future])
  const items = Object.fromEntries((overview?.completeness?.items || []).map(i => [i.name, i]))
  const chips = [['Glucose (CGM)', Droplet, 'Glucose'], ['Heart rate', HeartPulse, 'Heart rate'], ['Insulin', Syringe, 'Insulin'], ['Activity (steps)', Footprints, 'Activity']]
  return (
    <Card title={title} icon={LineChart} right={
      <div className="gt-row">
        {showFutureToggle && <button className="gt-btn small" onClick={() => setFuture(f => !f)}>{future ? 'Hide' : 'Show'} what happened next</button>}
        <Segmented label="Time range" value={hours} options={RANGES} onChange={setHours} />
      </div>}>
      {overview?.glucose_now != null && (
        <div className="gt-row gt-small" style={{ marginBottom: 4 }}>
          <span className="gt-muted">Current</span><b className="gt-mono">{Math.round(overview.glucose_now)} mg/dL</b>
          {overview.glucose_now > 180 ? <span className="gt-level high">Above range</span> : overview.glucose_now < 70 ? <span className="gt-level high">Below range</span> : <span className="gt-level low">In range</span>}
        </div>
      )}
      {stream ? <CgmChart points={stream.points} nowTime={stream.now} showFuture={future} height={events ? 220 : 230} />
        : <div className="gt-skel" style={{ height: 230 }} />}
      {events && stream && (
        <div className="gt-two" style={{ marginTop: 12 }}>
          <div><div className="gt-small gt-muted">Insulin (U per 15 min)</div><EventBars points={stream.points} dataKey="insulin" unit="U" color={css('brand-strong')} nowTime={stream.now} /></div>
          <div><div className="gt-small gt-muted">Logged carbs (g)</div><EventBars points={stream.points} dataKey="carbs" unit="g" color={css('above')} nowTime={stream.now} /></div>
        </div>
      )}
      {!events && (
        <div className="gt-four" style={{ marginTop: 12 }}>
          {chips.map(([k, Icon, label]) => {
            const it = items[k]
            return <div key={k} className={`gt-chip ${it?.available ? 'on' : 'off'}`}><Icon />{label}<small>{it?.available ? 'Available' : 'Not recorded'}</small></div>
          })}
        </div>
      )}
    </Card>
  )
}

function ContextExplainer() {
  const { detail } = useStudio()
  const sh = detail?.source?.startsWith('sh_')
  const rows = [
    [UtensilsCrossed, 'Meal & carbohydrates', 'Yes', sh ? 'Used by the 2-hour model (it learned meals from the Shanghai meal logs). The 30-min and 1-h models ignore meals for this dataset.' : 'Used by all three models, as a meal eaten now.'],
    [Syringe, 'Rapid insulin (units)', 'Yes', 'Added as a dose taken now; insulin on board is recomputed.'],
    [Pill, 'Oral tablets (metformin etc.)', sh ? 'Yes' : 'No', sh ? 'Used by the 2-hour model, which learned tablet timing from the Shanghai records.' : 'Ignored: the models only learned tablets from the Shanghai records.'],
    [Footprints, 'Activity', 'No', 'Kept as context. Wearable step features did not improve the model (+0.0 points), so they are not used.'],
    [Moon, 'Sleep', 'No', 'Kept as context. None of the training datasets recorded sleep.'],
    [ClipboardList, 'Notes', 'No', 'Kept with the prediction for the clinician.'],
  ]
  return (
    <Card title="What the model does with the daily context" icon={Info}>
      <div className="gt-list">
        {rows.map(([Icon, k, used, txt]) => (
          <div key={k} style={{ alignItems: 'flex-start' }}>
            <Icon style={{ width: 18, color: css('brand'), flexShrink: 0, marginTop: 2 }} />
            <div style={{ minWidth: 0 }}><b>{k}</b><div className="gt-small gt-muted">{txt}</div></div>
            <span className={`gt-right gt-level ${used === 'Yes' ? 'low' : 'medium'}`}>{used === 'Yes' ? 'Used' : 'Not used'}</span>
          </div>
        ))}
      </div>
      <div className="gt-note" style={{ marginTop: 10 }}>The context is applied as "what if this happens now". The models learned from logged meals and doses; these what-if answers are not separately validated.</div>
    </Card>
  )
}

function AdditionalInfo() {
  const { detail } = useStudio()
  return (
    <Card title="About this record and the model" icon={FlaskConical}>
      <div className="gt-list">
        <div><Database style={{ width: 16, color: css('brand') }} /><div><b>Dataset</b><div className="gt-small gt-muted">{detail?.dataset_citation}</div></div></div>
        <div><Shield style={{ width: 16, color: css('brand') }} /><div><b>Unseen patient</b><div className="gt-small gt-muted">The models that score this patient were trained without any of the 30 demo patients.</div></div></div>
        <div><Activity style={{ width: 16, color: css('brand') }} /><div><b>Replay</b><div className="gt-small gt-muted">Predictions use only data up to the chosen moment; what happened next is shown only for comparison.</div></div></div>
        <div><Info style={{ width: 16, color: css('brand') }} /><div><b>Research prototype</b><div className="gt-small gt-muted">Not a medical device and not for clinical use.</div></div></div>
      </div>
    </Card>
  )
}

/* ── bottom row ───────────────────────────────────────────── */
export function DailyContext({ compact = false }) {
  const { context: c, setContext, detail } = useStudio()
  const set = (k, v) => setContext({ ...c, [k]: v })
  const sh = detail?.source?.startsWith('sh_')
  const sc = contextToScenario(c)
  return (
    <Card title={<>Daily context <span className="gt-muted" style={{ fontWeight: 500 }}>(optional)</span></>} icon={UtensilsCrossed}>
      <div className="gt-three">
        <div className="gt-field">
          <label htmlFor="ctx-meal">Meal (now)</label>
          <select id="ctx-meal" className="gt-select" value={c.meal} onChange={e => set('meal', e.target.value)}>
            <option value="none">No meal</option><option value="planned">About to eat</option><option value="eating">Eating now</option>
          </select>
        </div>
        <div className="gt-field">
          <label htmlFor="ctx-carbs">Carbohydrates (g)</label>
          <input id="ctx-carbs" className="gt-input" type="number" min="0" max="300" disabled={c.meal === 'none'} value={c.carbs_g}
            placeholder={c.meal === 'none' ? '—' : 'e.g. 60'} onChange={e => set('carbs_g', e.target.value)} />
        </div>
        <div className="gt-field">
          <label htmlFor="ctx-med">Medication (now)</label>
          <select id="ctx-med" className="gt-select" value={c.medication} onChange={e => set('medication', e.target.value)}>
            <option value="none">None</option><option value="insulin">Rapid insulin</option><option value="tablet">Oral tablet(s)</option>
          </select>
          {c.medication === 'insulin' && <input aria-label="Insulin units" className="gt-input" style={{ marginTop: 6 }} type="number" min="0" max="50" placeholder="units" value={c.insulin_u} onChange={e => set('insulin_u', e.target.value)} />}
          {c.medication === 'tablet' && <>
            <input aria-label="Number of tablets" className="gt-input" style={{ marginTop: 6 }} type="number" min="1" max="5" value={c.oral_tablets} onChange={e => set('oral_tablets', e.target.value)} />
            {!sh && <div className="gt-hint">Not used for this dataset (no tablet records in training).</div>}
          </>}
        </div>
        {!compact && <>
          <div className="gt-field">
            <label htmlFor="ctx-act">Activity</label>
            <select id="ctx-act" className="gt-select" value={c.activity} onChange={e => set('activity', e.target.value)}>
              {['not set', 'rest', 'light activity', 'moderate exercise', 'intense exercise'].map(v => <option key={v}>{v}</option>)}
            </select>
            <div className="gt-hint">Context only, not used by the model.</div>
          </div>
          <div className="gt-field">
            <label htmlFor="ctx-sleep">Sleep (last night)</label>
            <select id="ctx-sleep" className="gt-select" value={c.sleep} onChange={e => set('sleep', e.target.value)}>
              {['not set', 'poor (< 5 h)', 'fair (5-7 h)', 'good (7-8 h)', 'long (> 8 h)'].map(v => <option key={v}>{v}</option>)}
            </select>
            <div className="gt-hint">Context only, not used by the model.</div>
          </div>
          <div className="gt-field">
            <label htmlFor="ctx-notes">Notes</label>
            <input id="ctx-notes" className="gt-input" placeholder="Any relevant information…" value={c.notes} onChange={e => set('notes', e.target.value)} />
          </div>
        </>}
      </div>
      <div className={`gt-hint ${sc ? 'used' : ''}`} style={{ marginTop: 10 }}>
        {sc ? `Will be sent to the model: ${Object.entries(sc).map(([k, v]) => `${k.replace('_', ' ')} ${v}`).join(', ')}. Press “Generate 2-hour prediction”.`
          : 'Nothing entered: the prediction uses the recorded data only.'}
      </div>
    </Card>
  )
}

export function Completeness() {
  const { overview } = useStudio()
  const c = overview?.completeness
  if (!c) return <Card title="Data completeness" icon={Database}><div className="gt-skel" style={{ height: 160 }} /></Card>
  const share = c.available / c.total
  return (
    <Card title="Data completeness" icon={Database} right={<b>{Math.round(100 * share)}%</b>}>
      <div className="gt-progress" style={{ marginBottom: 10 }}><div style={{ width: `${100 * share}%` }} /></div>
      <div className="gt-list">
        {c.items.map(i => (
          <div key={i.name}>
            {i.available ? <CheckCircle2 style={{ width: 17, color: css('good') }} /> : <XCircle style={{ width: 17, color: css('text-3') }} />}
            <span>{i.name}</span><span className="gt-right gt-small" style={{ color: i.available ? css('good') : css('text-3') }}>{i.detail}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}

/* ── right column: the twin ───────────────────────────────── */
export function TwinColumn() {
  const { overview, prediction, predError, modes, generate, loading, setPage } = useStudio()
  const h2 = prediction?.horizons?.['2h']
  const s2 = prediction?.scenario?.horizons?.['2h']
  const shown = s2 ?? h2
  const risk = riskOf(shown, modes.alert)
  const trend = overview?.trend_mgdl_per_min
  const TrendIcon = trend != null && trend < 0 ? TrendingDown : TrendingUp
  const ew = shown?.early_warning
  return (
    <div className="gt-grid gt-sticky">
      <div className="gt-card gt-twin-hero">
        <div className="gt-twin-hex"><Dna /></div>
        <div>
          <div style={{ fontSize: '1.2rem', fontWeight: 800 }}>Digital Twin</div>
          <div className="gt-small gt-muted">Health record + live CGM stream, replayed moment by moment, to estimate the risk of glucose above 180 mg/dL.</div>
        </div>
      </div>
      <div className="gt-twin-body">
        <BodyTwin glow={risk?.level ?? 'low'} />
        <div className="gt-twin-metric" style={{ top: 14 }}>
          <div className="gt-row" style={{ gap: 6 }}><Droplet style={{ width: 18, color: '#7ff0f0' }} /><span className="big">{fmt(overview?.glucose_now)}</span></div>
          <div className="lbl">mg/dL · current glucose</div>
        </div>
        <div className="gt-twin-metric" style={{ top: 112 }}>
          <div className="gt-row" style={{ gap: 6 }}><TrendIcon style={{ width: 18, color: '#7ff0f0' }} /><span className="big">{trend == null ? '—' : `${trend > 0 ? '+' : ''}${trend.toFixed(1)}`}</span></div>
          <div className="lbl">mg/dL per min · last 30 min</div>
        </div>
        <div className="gt-twin-metric" style={{ top: 210 }}>
          <div className="gt-row" style={{ gap: 6 }}><Shield style={{ width: 18, color: risk?.level === 'high' ? '#ff9c8a' : risk?.level === 'medium' ? '#ffd27a' : '#7ff0c8' }} />
            <span style={{ fontWeight: 800, fontSize: '1.05rem' }}>{risk ? risk.label : '—'}</span></div>
          <div className="lbl">next 2 hours{modes.alert === 'safety' ? ' · safety mode' : ''}</div>
        </div>
      </div>
      <div className="gt-ai">
        <div className="gt-row" style={{ fontWeight: 800, fontSize: '1.1rem', marginBottom: 6 }}><Activity style={{ width: 20 }} />AI prediction</div>
        {predError ? <div className="gt-small" style={{ color: '#ffd27a', margin: '10px 0' }}>{predError}</div> : (
          <div className="gt-row" style={{ gap: 6 }}>
            <Gauge p={shown?.probability} level={risk?.level} onDark size={190}
              ticks={shown ? [shown.threshold_safety_mode, shown.threshold_accuracy_mode] : []} />
            <div className="gt-small" style={{ color: '#bfe9ec' }}>
              Risk of glucose above 180 mg/dL in the next 2 hours
              {s2 && h2 && <div style={{ marginTop: 6, color: '#fff' }}>With today's context: <b>{Math.round(100 * s2.probability)}%</b><br />Recorded data only: {Math.round(100 * h2.probability)}%</div>}
            </div>
          </div>
        )}
        {ew?.active && <div className="gt-row gt-small" style={{ color: '#ffb4a6', fontWeight: 700, margin: '4px 0 10px' }}>⚠ Early warning: glucose is still in range — act now</div>}
        <button className="gt-btn primary" onClick={generate} disabled={loading}>
          Generate 2-hour prediction<ArrowRight />
        </button>
        <div className="gt-small" style={{ color: '#a9dfe3', marginTop: 6, textAlign: 'center' }}>
          {loading ? 'Updating…' : prediction ? `For ${overview?.now ?? 'this moment'}${prediction.scenario ? ' · with daily context' : ' · recorded data only'}` : ''}
        </div>
        <button className="gt-btn small" style={{ marginTop: 10, width: '100%', background: 'transparent', color: '#cdeff1', borderColor: 'rgba(205,239,241,0.3)' }} onClick={() => setPage('prediction')}>
          See all horizons and why
        </button>
        <div className="gt-small" style={{ color: '#8fc3c8', marginTop: 8 }}>Ticks mark the model's watch and alert thresholds. Research prototype — not for clinical use.</div>
      </div>
    </div>
  )
}

/* ── the page ─────────────────────────────────────────────── */
const TABS = [['records', 'Health records', ClipboardList], ['sensor', 'Sensor data', Radio], ['context', 'Daily context', UtensilsCrossed], ['info', 'Additional info', Info]]

export default function PatientPage() {
  const { detail, patients, pid } = useStudio()
  const [tab, setTab] = useState('records')
  const L = labsMap(detail)
  const cur = patients.find(p => p.patient_id === pid)
  return (
    <div className="gt-patient-grid">
      <div className="gt-grid" style={{ gap: 18 }}>
        <div className="gt-top-row">
          <SelectPatient />
          <Stat label="Age" icon={Calendar} value={L.Age ? Math.round(L.Age.value) : '—'} unit={L.Age ? 'years' : 'not in dataset'} />
          <Stat label="HbA1c" icon={Droplet} value={L.HbA1c ? `${L.HbA1c.value}%` : '—'} unit={L.HbA1c ? 'at enrolment' : 'not in dataset'} />
          <Stat label="Diabetes type" icon={Activity} value={detail ? (detail.diabetes_type === 'T2D' ? 'Type 2' : 'Type 1') : '—'} />
          <Stat label="Data source" icon={Database} value={cur?.dataset ?? '—'} unit={detail?.uploaded ? 'your file, in memory only' : 'open data (CC BY 4.0)'} />
        </div>
        <MomentBar />
        <div className="gt-tabs" role="tablist">
          {TABS.map(([k, label, Icon]) => (
            <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}><Icon />{label}</button>
          ))}
        </div>
        {tab === 'records' && <div className="gt-two"><HealthRecords /><GlucoseCard /></div>}
        {tab === 'sensor' && <GlucoseCard title="Sensor stream" showFutureToggle events />}
        {tab === 'context' && <ContextExplainer />}
        {tab === 'info' && <AdditionalInfo />}
        <div className="gt-wide">
          <DailyContext />
          <Completeness />
        </div>
      </div>
      <TwinColumn />
    </div>
  )
}
