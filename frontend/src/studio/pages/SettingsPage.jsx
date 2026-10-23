import { useEffect, useState } from 'react'
import { Database, Bell, Palette, LayoutDashboard, PaintBucket, RotateCcw, Check } from 'lucide-react'
import { useStudio } from '../store'
import { Card, Segmented } from '../ui'

const PRESETS = [
  ['Mint', '#dff3ee'], ['Sky', '#dcebfa'], ['Lavender', '#e8e3f7'], ['Peach', '#fbe6dc'], ['Sand', '#f3ecdc'],
  ['Rose', '#f7e1e7'], ['Slate', '#3b4a57'], ['Forest', '#16392f'], ['Navy', '#132743'], ['Midnight', '#0d1117'],
]
const toHex = ([r, g, b]) => `#${[r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('')}`
const toRgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16))

function BackgroundPicker({ value, onChange }) {
  const [rgb, setRgb] = useState(() => toRgb(value || '#e6f1f4'))
  useEffect(() => { if (value) setRgb(toRgb(value)) }, [value])
  const setChannel = (i, v) => { const n = [...rgb]; n[i] = Number(v); setRgb(n); onChange(toHex(n)) }
  const hex = toHex(rgb)
  return (
    <Card title="Background colour" icon={PaintBucket} right={
      <button className="gt-btn small" onClick={() => onChange(null)} disabled={!value}><RotateCcw />Theme default</button>}>
      <div className="gt-small" style={{ color: 'var(--gt-text-2)', marginBottom: 12 }}>
        Pick a preset or mix your own with the R / G / B sliders. Cards stay readable; the page title switches to light text on dark colours.
      </div>
      <div className="gt-row" style={{ flexWrap: 'wrap', gap: 10, marginBottom: 16 }}>
        {PRESETS.map(([name, c]) => (
          <button key={c} onClick={() => onChange(c)} aria-label={`${name} background`} title={name}
            style={{ width: 46, height: 46, borderRadius: 14, background: c, cursor: 'pointer', display: 'grid', placeItems: 'center',
              border: value === c ? '3px solid var(--gt-brand)' : '1px solid var(--gt-border-strong)' }}>
            {value === c && <Check style={{ width: 18, color: toRgb(c).reduce((a, b) => a + b) < 300 ? '#fff' : '#0f2a35' }} />}
          </button>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '92px minmax(0,1fr)', gap: 16, alignItems: 'center' }}>
        <label style={{ display: 'grid', gap: 6, justifyItems: 'center' }} className="gt-small">
          <input type="color" value={hex} onChange={e => onChange(e.target.value)} aria-label="Pick any colour"
            style={{ width: 76, height: 76, border: '1px solid var(--gt-border-strong)', borderRadius: 16, padding: 4, background: 'var(--gt-surface-solid)', cursor: 'pointer' }} />
          <span className="gt-mono">{hex.toUpperCase()}</span>
        </label>
        <div style={{ display: 'grid', gap: 8 }}>
          {['R', 'G', 'B'].map((ch, i) => (
            <div key={ch} className="gt-row">
              <b style={{ width: 16, color: ['#d03b3b', '#0b8a3e', '#2a78d6'][i] }}>{ch}</b>
              <input className="gt-range" type="range" min="0" max="255" value={rgb[i]} onChange={e => setChannel(i, e.target.value)}
                aria-label={`${['Red', 'Green', 'Blue'][i]} channel`} style={{ accentColor: ['#d03b3b', '#0b8a3e', '#2a78d6'][i] }} />
              <input className="gt-input gt-mono" type="number" min="0" max="255" value={rgb[i]} onChange={e => setChannel(i, e.target.value)}
                aria-label={`${['Red', 'Green', 'Blue'][i]} value`} style={{ width: 76, padding: '6px 8px' }} />
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}

function Mode({ icon, title, value, options, onChange, children }) {
  return (
    <Card title={title} icon={icon} right={<Segmented label={title} value={value} options={options} onChange={onChange} />}>
      <div className="gt-small" style={{ color: 'var(--gt-text-2)', lineHeight: 1.6 }}>{children}</div>
    </Card>
  )
}

export default function SettingsPage() {
  const { modes, setModes } = useStudio()
  const set = (k, v) => setModes({ ...modes, [k]: v })
  return (
    <div className="gt-grid" style={{ gap: 16, maxWidth: 980 }}>
      <Mode icon={Database} title="Data mode" value={modes.data} onChange={v => set('data', v)}
        options={[['open', 'Real patients (open data)'], ['synthetic', 'Synthetic demo']]}>
        <b>Real patients</b> replays 30 anonymised open-data recordings (Shanghai T2DM / T1DM, AZT1D, T1D-UOM, HUPA-UCM) through the
        validated models; none of these people were in the models' training data. <b>Synthetic demo</b> shows the original simulated
        patients and their separate model, for demonstration only.
      </Mode>
      <Mode icon={Bell} title="Alert mode" value={modes.alert} onChange={v => set('alert', v)}
        options={[['standard', 'Standard'], ['safety', 'Safety']]}>
        <b>Standard</b> raises an alert only at high risk (the threshold that gave the best accuracy on held-out patients): fewer false
        alarms, some spikes missed. <b>Safety</b> also alerts at medium risk (the threshold that balanced sensitivity and specificity):
        more spikes caught, more false alarms. The separate 2-hour <i>early warning</i> (fires only while glucose is still in range)
        is the same in both modes.
      </Mode>
      <Mode icon={Palette} title="Theme" value={modes.theme} onChange={v => set('theme', v)}
        options={[['light', 'Light'], ['dark', 'Dark']]}>
        Light is the clinic layout; dark is easier on the eyes at night. Chart colours were checked for colour-blind readers in both.
      </Mode>
      <BackgroundPicker value={modes.bg} onChange={v => set('bg', v)} />
      <Mode icon={LayoutDashboard} title="Layout" value={modes.layout} onChange={v => set('layout', v)}
        options={[['studio', 'New'], ['classic', 'Classic']]}>
        <b>New</b> is this layout. <b>Classic</b> is the original dark dashboard; both use the same backend and models.
      </Mode>
      <div className="gt-note">Modes are remembered in this browser only.</div>
    </div>
  )
}
