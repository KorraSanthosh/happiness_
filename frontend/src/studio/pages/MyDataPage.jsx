/**
 * Upload your own CGM export and get a digital twin of your own data.
 * The file goes only to this app's own backend and is kept in its memory (never written to disk).
 */
import { useRef, useState } from 'react'
import { UploadCloud, FileText, User, ShieldCheck, Trash2, ArrowRight, Download, CheckCircle2, Info } from 'lucide-react'
import { uploadV7, deleteV7Upload, V7_EXAMPLE_CSV } from '../../services/api'
import { useStudio } from '../store'
import { Card, Banner, css } from '../ui'

const EMPTY = { diabetes_type: 'T2D', age: '', sex: '', hba1c: '', bmi: '', years_with_diabetes: '', on_insulin: false, on_metformin: false, consent: false }

export default function MyDataPage() {
  const { patients, reloadPatients, setPage, pid } = useStudio()
  const [file, setFile] = useState(null)
  const [f, setF] = useState(EMPTY)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [done, setDone] = useState(null)
  const [drag, setDrag] = useState(false)
  const input = useRef(null)
  const set = (k, v) => setF({ ...f, [k]: v })
  const mine = patients.filter(p => p.dataset === 'Your upload')

  const submit = () => {
    setBusy(true); setErr(null); setDone(null)
    const form = new FormData()
    form.append('file', file)
    for (const [k, v] of Object.entries(f)) if (v !== '' && v !== null) form.append(k, String(v))
    uploadV7(form).then(r => { setDone(r); return reloadPatients(r.patient_id) })
      .catch(e => setErr(e?.response?.data?.detail || e.message)).finally(() => setBusy(false))
  }
  const remove = id => deleteV7Upload(id).then(() => reloadPatients(id === pid ? null : undefined))

  return (
    <div className="gt-wide">
      <div className="gt-grid" style={{ gap: 18 }}>
        <Card title="1. Your CGM file" icon={UploadCloud}>
          <div onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]) }}
            onClick={() => input.current?.click()} role="button" tabIndex={0}
            onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
            style={{ border: `2px dashed ${drag ? css('brand') : css('border-strong')}`, borderRadius: 16, padding: '28px 16px', textAlign: 'center',
              cursor: 'pointer', background: drag ? css('brand-soft') : css('surface-2') }}>
            <UploadCloud style={{ width: 34, height: 34, color: css('brand') }} />
            <div style={{ fontWeight: 700, marginTop: 6 }}>{file ? file.name : 'Drop your CSV file here, or click to choose'}</div>
            <div className="gt-small gt-muted">{file ? `${(file.size / 1024).toFixed(0)} KB` : 'CSV, up to 5 MB'}</div>
            <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={e => setFile(e.target.files[0] || null)} />
          </div>
          <div className="gt-list" style={{ marginTop: 12 }}>
            <div><FileText style={{ width: 16, color: css('brand') }} /><div><b>Dexcom</b><div className="gt-small gt-muted">Clarity → Export → CSV</div></div></div>
            <div><FileText style={{ width: 16, color: css('brand') }} /><div><b>FreeStyle Libre</b><div className="gt-small gt-muted">LibreView → Glucose history → Download (CSV)</div></div></div>
            <div><FileText style={{ width: 16, color: css('brand') }} /><div><b>Any other CSV</b><div className="gt-small gt-muted">A time column and a glucose column (mg/dL or mmol/L); carbs and insulin columns are optional</div></div></div>
          </div>
          <a className="gt-btn small" style={{ marginTop: 10, textDecoration: 'none' }} href={V7_EXAMPLE_CSV} download><Download />Download an example file</a>
          <div className="gt-note" style={{ marginTop: 8 }}>Best results with 7–14 days of data; at least 24 hours recommended (6 hours minimum).</div>
        </Card>

        <Card title="2. About you (optional, but helps)" icon={User}>
          <div className="gt-three">
            <div className="gt-field"><label htmlFor="u-type">Diabetes type</label>
              <select id="u-type" className="gt-select" value={f.diabetes_type} onChange={e => set('diabetes_type', e.target.value)}>
                <option value="T2D">Type 2</option><option value="T1D">Type 1</option></select></div>
            <div className="gt-field"><label htmlFor="u-age">Age (years)</label><input id="u-age" className="gt-input" type="number" min="1" max="110" value={f.age} onChange={e => set('age', e.target.value)} /></div>
            <div className="gt-field"><label htmlFor="u-sex">Sex</label>
              <select id="u-sex" className="gt-select" value={f.sex} onChange={e => set('sex', e.target.value)}>
                <option value="">Prefer not to say</option><option value="F">Female</option><option value="M">Male</option></select></div>
            <div className="gt-field"><label htmlFor="u-a1c">HbA1c (%)</label><input id="u-a1c" className="gt-input" type="number" step="0.1" min="3" max="20" value={f.hba1c} onChange={e => set('hba1c', e.target.value)} /></div>
            <div className="gt-field"><label htmlFor="u-bmi">BMI</label><input id="u-bmi" className="gt-input" type="number" step="0.1" min="10" max="80" value={f.bmi} onChange={e => set('bmi', e.target.value)} /></div>
            <div className="gt-field"><label htmlFor="u-yrs">Years with diabetes</label><input id="u-yrs" className="gt-input" type="number" min="0" max="90" value={f.years_with_diabetes} onChange={e => set('years_with_diabetes', e.target.value)} /></div>
          </div>
          <div className="gt-row" style={{ marginTop: 12, gap: 18, flexWrap: 'wrap' }}>
            <label className="gt-row gt-small"><input type="checkbox" checked={f.on_insulin} onChange={e => set('on_insulin', e.target.checked)} />I take insulin</label>
            <label className="gt-row gt-small"><input type="checkbox" checked={f.on_metformin} onChange={e => set('on_metformin', e.target.checked)} />I take metformin</label>
          </div>
          <div className="gt-note" style={{ marginTop: 8 }}>Age is needed for the clinical details to be used; without it the twin uses your diabetes type and glucose data only.</div>
        </Card>

        <Card title="3. Create your twin" icon={ShieldCheck}>
          <label className="gt-row gt-small" style={{ alignItems: 'flex-start' }}>
            <input type="checkbox" style={{ marginTop: 3 }} checked={f.consent} onChange={e => set('consent', e.target.checked)} />
            <span>I understand this is a research prototype, not medical advice, and must not be used to change my treatment.
              My file is sent only to this app's own server, kept in its memory (not saved to disk) and deleted when the server stops or when I delete it.</span>
          </label>
          {err && <div style={{ marginTop: 10 }}><Banner kind="warn">{err}</Banner></div>}
          <button className="gt-btn primary" style={{ marginTop: 12, width: '100%' }} disabled={!file || !f.consent || busy} onClick={submit}>
            {busy ? 'Reading your file…' : 'Upload and create my twin'}<ArrowRight />
          </button>
          {done && (
            <div style={{ marginTop: 14 }}>
              <Banner kind="info"><b>Done.</b> Read as a {done.summary.format} in {done.summary.units}: {done.summary.readings.toLocaleString()} readings over {done.summary.days} days
                (every {done.summary.reading_interval_min} min, {done.summary.first.slice(0, 16)} → {done.summary.last.slice(0, 16)}).
                {done.summary.has_carbs ? ' Carbs found.' : ' No carbs in the file.'}{done.summary.has_insulin ? ' Insulin found.' : ' No insulin in the file.'}</Banner>
              {done.summary.notes.map(n => <div key={n} className="gt-hint" style={{ marginTop: 6 }}>• {n}</div>)}
              <button className="gt-btn" style={{ marginTop: 10 }} onClick={() => setPage('patient')}><CheckCircle2 />Open my twin</button>
            </div>
          )}
        </Card>
      </div>

      <div className="gt-grid gt-sticky" style={{ gap: 18 }}>
        <Card title="Your uploads" icon={FileText}>
          {mine.length === 0 ? <div className="gt-small gt-muted">Nothing uploaded yet.</div> : (
            <div className="gt-list">{mine.map(p => (
              <div key={p.patient_id}>
                <div><b>{p.patient_id.replace('upload_', 'Upload ')}</b><div className="gt-small gt-muted">{p.diabetes_type === 'T2D' ? 'Type 2' : 'Type 1'} · {p.days} days</div></div>
                <button className="gt-btn small gt-right" onClick={() => remove(p.patient_id)} aria-label={`Delete ${p.patient_id}`}><Trash2 />Delete</button>
              </div>))}</div>
          )}
        </Card>
        <Card title="What happens to your data" icon={Info}>
          <div className="gt-list gt-small">
            <div>Your readings are converted to the same 15-minute grid as the training data, using the same cleaning rules.</div>
            <div>The models score you at any moment you pick, using only data up to that moment — exactly as for the 30 demo patients.</div>
            <div>You were never in the training data, so this is a true “unseen person” test — but the models have not been validated for individual medical use.</div>
            <div>Nothing is saved to disk. Restarting the server or pressing Delete removes it.</div>
          </div>
        </Card>
      </div>
    </div>
  )
}
