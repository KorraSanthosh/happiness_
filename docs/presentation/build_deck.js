/**
 * Builds docs/GlucoTwin_Presentation.pptx from the v7 result files.
 *   NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node docs/presentation/build_deck.js
 * Numbers are read from models/combined_v7/*.json, so the deck always matches the results.
 */
const path = require('path')
const fs = require('fs')
const pptxgen = require('pptxgenjs')
const React = require('react')
const ReactDOMServer = require('react-dom/server')
const sharp = require('sharp')
const fa = require('react-icons/fa')

const ROOT = path.resolve(__dirname, '..', '..')
const V7 = path.join(ROOT, 'models', 'combined_v7')
const OUT = path.join(ROOT, 'docs', 'GlucoTwin_Presentation.pptx')
const load = f => JSON.parse(fs.readFileSync(path.join(V7, f), 'utf8'))
const cv = load('stage_cv.json').horizons
const fin = load('stage_final.json')
const abl = load('stage_ablation.json')
const inv = JSON.parse(fs.readFileSync(path.join(ROOT, 'models', 'investigation', 'analysis.json'), 'utf8'))
const v9 = JSON.parse(fs.readFileSync(path.join(ROOT, 'backend', 'models', 'twin_v7', 'summary.json'), 'utf8')).v9
const pc = v => `${(100 * v).toFixed(1)}%`
const H = ['30m', '60m', '2h']
const HL = { '30m': '30 min', '60m': '1 hour', '2h': '2 hours' }

const THEME = {
  name: 'GlucoTwin', headFontFace: 'Cambria', bodyFontFace: 'Calibri',
  colors: {
    dk1: '1B2430', lt1: 'FFFFFF', dk2: '0B2545', lt2: 'EEF4F7',
    accent1: '0E7C86', accent2: 'E4572E', accent3: 'F3A712', accent4: '29A36A', accent5: '5B7DB1', accent6: '8A94A6',
    hlink: '0E7C86', folHlink: '5B7DB1',
  },
}
const HEX = THEME.colors

async function icon(Comp, color = '#FFFFFF') {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color, size: 256 }))
  const png = await sharp(Buffer.from(svg)).png().toBuffer()
  return 'image/png;base64,' + png.toString('base64')
}

async function main() {
  const pres = new pptxgen()
  pres.layout = 'LAYOUT_WIDE'                      // 13.333 x 7.5 in
  pres.title = 'GlucoTwin - Diabetes Digital Twin'
  pres.author = 'GlucoTwin team'
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace }
  const C = pres.SchemeColor

  // ── layouts ────────────────────────────────────────────────
  pres.defineSlideMaster({
    title: 'TITLE_DARK', background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.0, w: 11.7, h: 1.5, fontSize: 48, bold: true, color: C.background1, valign: 'bottom' }, text: '' } },
      { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 3.6, w: 11.7, h: 1.4, fontSize: 22, color: C.background2, valign: 'top' }, text: '' } },
    ],
  })
  pres.defineSlideMaster({
    title: 'CONTENT', background: { color: C.background1 },
    margin: [0.5, 0.6, 0.7, 0.6],
    objects: [
      { placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 30, bold: true, color: C.text2, valign: 'middle', align: 'left' }, text: '' } },
      { text: { text: 'GlucoTwin · Digital Twin Challenge 2026 · research prototype', options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent6 } } },
    ],
    slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6 },
  })

  const ic = {
    heart: await icon(fa.FaHeartbeat), notes: await icon(fa.FaNotesMedical), wave: await icon(fa.FaChartLine),
    brain: await icon(fa.FaBrain), doctor: await icon(fa.FaUserMd), db: await icon(fa.FaDatabase),
    clone: await icon(fa.FaClone), eye: await icon(fa.FaEyeSlash), ruler: await icon(fa.FaRulerCombined),
    q: await icon(fa.FaQuestion), clock: await icon(fa.FaClock), shield: await icon(fa.FaShieldAlt),
    users: await icon(fa.FaUsers), flask: await icon(fa.FaFlask), globe: await icon(fa.FaGlobeAsia),
    link: await icon(fa.FaLink), mobile: await icon(fa.FaMobileAlt), lock: await icon(fa.FaLock),
    check: await icon(fa.FaCheck), cross: await icon(fa.FaTimes),
  }
  const circleIcon = (s, data, x, y, d, fill, name) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name + '-circle' })
    s.addImage({ data, x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52, objectName: name + '-icon' })
  }
  const card = (s, x, y, w, h, name, fill = C.background2) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name })
  const text = (s, t, o) => s.addText(t, { isTextBox: true, margin: 0, color: C.text1, fontSize: 14, valign: 'top', ...o })
  // Bar chart drawn from shapes: Keynote does not import pptxgenjs chart objects (they come out blank),
  // and the team presents from Keynote/PDF. Shapes stay editable and render identically everywhere.
  const barChart = (s, { x, y, w, h, cats, series, min, max, step, title, name }) => {
    text(s, title, { x, y, w, h: 0.4, fontSize: 14, bold: true, color: C.text2 })
    const top = y + 0.55, legendH = 0.45, catH = 0.4
    const ph = h - 0.55 - catH - legendH, px = x + 0.5, pw = w - 0.6
    for (let v = min; v <= max + 1e-9; v += step) {
      const yy = top + ph * (1 - (v - min) / (max - min))
      s.addShape(pres.shapes.LINE, { x: px, y: yy, w: pw, h: 0, line: { color: HEX.lt2, width: 0.75 }, objectName: `${name}-grid-${v}` })
      text(s, String(v), { x: x, y: yy - 0.12, w: 0.42, h: 0.24, fontSize: 10, color: C.accent6, align: 'right' })
    }
    const gw = pw / cats.length, bw = (gw * 0.72) / series.length
    cats.forEach((c, i) => {
      series.forEach((sr, j) => {
        const v = sr.values[i], bh = ph * Math.max(0, (v - min) / (max - min))
        const bx = px + i * gw + gw * 0.14 + j * bw
        s.addShape(pres.shapes.RECTANGLE, { x: bx + 0.03, y: top + ph - bh, w: bw - 0.06, h: bh, fill: { color: sr.color }, line: { color: sr.color }, objectName: `${name}-bar-${i}-${j}` })
        text(s, v.toFixed(1), { x: bx - 0.1, y: top + ph - bh - 0.3, w: bw + 0.2, h: 0.26, fontSize: 11, color: C.text1, align: 'center' })
      })
      text(s, c, { x: px + i * gw, y: top + ph + 0.08, w: gw, h: 0.3, fontSize: 13, bold: true, color: C.text1, align: 'center' })
    })
    let lx = px
    series.forEach((sr, j) => {
      s.addShape(pres.shapes.RECTANGLE, { x: lx, y: y + h - 0.3, w: 0.18, h: 0.18, fill: { color: sr.color }, line: { color: sr.color }, objectName: `${name}-key-${j}` })
      text(s, sr.name, { x: lx + 0.25, y: y + h - 0.36, w: 2.9, h: 0.3, fontSize: 11, color: C.text1 })
      lx += 3.0
    })
  }
  const content = (title, section) => {
    const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: section })
    s.addText(title, { placeholder: 'title' })
    return s
  }

  // ── 1. title ───────────────────────────────────────────────
  pres.addSection({ title: 'Opening' })
  let s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: 'Opening' })
  s.addText('GlucoTwin', { placeholder: 'title' })
  s.addText('A diabetes digital twin that predicts glucose spikes 30 minutes to 2 hours before they happen', { placeholder: 'body' })
  circleIcon(s, ic.heart, 0.8, 0.8, 1.0, C.accent2, 'logo')
  text(s, 'Digital Twin Challenge 2026  ·  Team hackers_  ·  IIT (ISM) Dhanbad  ·  Team leader: Korra Santhosh', { x: 0.8, y: 6.3, w: 11.7, h: 0.4, fontSize: 16, color: C.background2 })
  s.addNotes('Introduce the team: hackers_ from IIT (ISM) Dhanbad, presented by team leader Korra Santhosh. GlucoTwin is a proof-of-concept digital twin for people with diabetes: it fuses the patient record with continuous glucose and wearable data, and warns 30 minutes to 2 hours before glucose goes above 180 mg/dL.')

  // ── 2. problem ─────────────────────────────────────────────
  pres.addSection({ title: 'Problem' })
  s = content('Diabetes care in India is still reactive', 'Problem')
  card(s, 0.6, 1.5, 3.9, 2.4, 'stat1', C.text2)
  text(s, '101 M', { x: 0.9, y: 1.75, w: 3.3, h: 1.1, fontSize: 60, bold: true, color: C.background1, fontFace: 'Cambria' })
  text(s, 'Indians living with diabetes', { x: 0.9, y: 2.95, w: 3.3, h: 0.7, fontSize: 16, color: C.background2 })
  card(s, 0.6, 4.1, 3.9, 2.4, 'stat2', C.accent1)
  text(s, '136 M', { x: 0.9, y: 4.35, w: 3.3, h: 1.1, fontSize: 60, bold: true, color: C.background1, fontFace: 'Cambria' })
  text(s, 'more with prediabetes (ICMR-INDIAB, 2023)', { x: 0.9, y: 5.55, w: 3.3, h: 0.7, fontSize: 16, color: C.background1 })
  text(s, [
    { text: 'Repeated high glucose (> 180 mg/dL) drives eye, kidney, nerve and heart damage.', options: { bullet: true, breakLine: true } },
    { text: 'A continuous glucose monitor alarms only after glucose is already high.', options: { bullet: true, breakLine: true } },
    { text: 'Doctors see past readings, not where this patient is heading next.', options: { bullet: true } },
  ], { x: 5.0, y: 1.6, w: 7.6, h: 2.4, fontSize: 18, paraSpaceAfter: 10 })
  card(s, 5.0, 4.4, 7.6, 2.0, 'question', C.background2)
  circleIcon(s, ic.q, 5.3, 4.85, 1.0, C.accent2, 'q')
  text(s, 'Our question, asked every 15 minutes for every patient:', { x: 6.6, y: 4.65, w: 5.8, h: 0.4, fontSize: 14, color: C.accent6 })
  text(s, 'Will glucose go above 180 mg/dL in the next 30 minutes, 1 hour or 2 hours?', { x: 6.6, y: 5.1, w: 5.8, h: 1.1, fontSize: 20, bold: true, color: C.text2 })
  s.addNotes('India has about 101 million people with diabetes and 136 million with prediabetes (ICMR-INDIAB, Lancet Diabetes & Endocrinology 2023). Hyperglycaemia causes long-term complications, yet monitoring is reactive. We want a forward-looking view.')

  // ── 3. solution ────────────────────────────────────────────
  pres.addSection({ title: 'Solution' })
  s = content('Our digital twin fuses two data streams and looks ahead', 'Solution')
  const steps = [
    [ic.notes, 'Static stream: EHR', 'Age, BMI, diabetes type and duration, complications, medicines, HbA1c and other labs', C.accent5],
    [ic.wave, 'Dynamic stream', 'CGM glucose, insulin, meals, heart rate, steps — every 5 to 15 minutes', C.accent1],
    [ic.brain, 'The twin', '111 fused features → three models (30 min, 1 h, 2 h), each with accuracy and safety alert modes', C.text2],
    [ic.doctor, 'Doctor dashboard', 'Live timeline, risk cards, patient record and a replay of what really happened', C.accent2],
  ]
  steps.forEach(([im, head, body, col], i) => {
    const x = 0.6 + i * 3.15
    card(s, x, 1.9, 2.85, 3.9, 'step' + i, C.background2)
    circleIcon(s, im, x + 0.92, 2.2, 1.0, col, 'step' + i)
    text(s, head, { x: x + 0.2, y: 3.4, w: 2.45, h: 0.6, fontSize: 19, bold: true, color: C.text2, align: 'center' })
    text(s, body, { x: x + 0.2, y: 4.05, w: 2.45, h: 1.6, fontSize: 15, align: 'center' })
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.89, y: 3.7, w: 0.22, h: 0.3, fill: { color: C.accent6 }, line: { color: C.accent6 }, objectName: 'arrow' + i })
  })
  s.addNotes('The twin has two inputs, exactly as the challenge asks: a static clinical record and a dynamic sensor stream. They are fused into one feature vector; three models give the risk for three horizons; the doctor sees it on a dashboard.')

  // ── 4. data ────────────────────────────────────────────────
  pres.addSection({ title: 'Data' })
  s = content('Built only on open, anonymised data: 187 people, Type 1 and Type 2', 'Data')
  const rows = [
    ['Dataset', 'People', 'Diabetes', 'What it adds'],
    ['ShanghaiT2DM', '100', 'Type 2', 'Real clinical record (labs, drugs, history)'],
    ['ShanghaiT1DM', '12', 'Type 1', 'Real clinical record'],
    ['HUPA-UCM (Spain)', '24', 'Type 1', 'Fitbit heart rate, steps, calories'],
    ['AZT1D (USA)', '25', 'Type 1', 'Insulin-pump data, carbohydrates'],
    ['T1D-UOM (UK)', '17', 'Type 1', 'Insulin, meals, heart rate, steps'],
    ['BrisT1D (UK)', '9', 'Type 1', 'Insulin, meals, smartwatch'],
    ['OhioT1DM (USA)', '12', 'Type 1', 'External test only — never trained on'],
  ].map((r, i) => r.map(c => ({ text: c, options: i === 0 ? { bold: true, color: C.background1, fill: { color: C.text2 } } : {} })))
  s.addTable(rows, { x: 0.6, y: 1.5, w: 8.0, colW: [2.2, 0.9, 1.1, 3.8], fontSize: 13, color: C.text1, border: { type: 'solid', pt: 0.5, color: HEX.lt2 }, rowH: 0.48, objectName: 'datasets' })
  const facts = [['187', 'people for training & testing'], ['≈ 530 k', 'prediction moments per horizon'], ['Open data', 'CC BY 4.0 research datasets*']]
  facts.forEach(([big, small], i) => {
    card(s, 9.0, 1.5 + i * 1.6, 3.7, 1.4, 'fact' + i, i === 2 ? C.accent1 : C.background2)
    text(s, big, { x: 9.25, y: 1.6 + i * 1.6, w: 3.3, h: 0.7, fontSize: 32, bold: true, color: i === 2 ? C.background1 : C.text2, fontFace: 'Cambria' })
    text(s, small, { x: 9.25, y: 2.3 + i * 1.6, w: 3.3, h: 0.5, fontSize: 14, color: i === 2 ? C.background1 : C.text1 })
  })
  text(s, '* BrisT1D is published under CC BY 4.0; we used its Kaggle competition copy, whose rules we have not separately verified.', { x: 0.6, y: 6.55, w: 12.1, h: 0.3, fontSize: 10, color: C.accent6 })
  s.addNotes('Every training dataset is openly available and anonymised, as the challenge rules require. The 100 Type 2 patients from Shanghai are the closest open match to the Indian population and come with a real clinical summary. OhioT1DM is under a data-use agreement, so we only use it locally as an unseen external test.')

  // ── 5. data engineering ────────────────────────────────────
  s = content('We found and fixed six data problems before any modelling', 'Data')
  const probs = [
    [ic.clone, 'Duplicated readings', 'BrisT1D stored each reading up to 72 times; stitched into one timeline (0 conflicts in 11.4 M values)'],
    [ic.eye, 'Hidden future leak', 'HUPA-UCM gaps were filled by straight lines toward the NEXT reading; removed'],
    [ic.ruler, 'Mixed units', 'mmol/L vs mg/dL, carbs in grams vs 10-g portions, basal U/h vs units'],
    [ic.db, 'Missing ≠ zero', '"Not recorded" kept separate from "none"; long-acting insulin spread over 24 h'],
    [ic.clock, 'Different sensor speeds', '5-min and 15-min CGMs put on one common 15-minute grid'],
    [ic.shield, 'Leak-proof by test', '100+ automated tests: changing future data never changes a past feature'],
  ]
  probs.forEach(([im, head, body], i) => {
    const x = 0.6 + (i % 3) * 4.15, y = 1.55 + Math.floor(i / 3) * 2.65
    card(s, x, y, 3.9, 2.4, 'prob' + i, C.background2)
    circleIcon(s, im, x + 0.25, y + 0.3, 0.8, i === 5 ? C.accent4 : C.accent2, 'prob' + i)
    text(s, head, { x: x + 1.2, y: y + 0.4, w: 2.55, h: 0.6, fontSize: 17, bold: true, color: C.text2, valign: 'middle' })
    text(s, body, { x: x + 0.25, y: y + 1.25, w: 3.45, h: 1.05, fontSize: 13 })
  })
  s.addNotes('Most of the real work was data engineering. Each of these problems would silently inflate or corrupt accuracy; we caught them with sanity checks and lock them down with tests.')

  // ── 6. features & model ────────────────────────────────────
  pres.addSection({ title: 'Model' })
  s = content('111 fused features feed three horizon-specific models (123 in the v9 2-hour model)', 'Model')
  card(s, 0.6, 1.5, 4.0, 5.0, 'stream', C.background2)
  text(s, 'Stream features · 80', { x: 0.85, y: 1.7, w: 3.5, h: 0.5, fontSize: 20, bold: true, color: C.accent1 })
  text(s, [
    { text: 'Glucose level, change, slope, acceleration, 30-min projection', options: { bullet: true, breakLine: true } },
    { text: 'Rolling mean, spread, min, max (30 min – 4 h)', options: { bullet: true, breakLine: true } },
    { text: 'Insulin on board, carbs on board, last meal', options: { bullet: true, breakLine: true } },
    { text: 'Heart rate, steps, energy', options: { bullet: true, breakLine: true } },
    { text: 'Personal memory: spikes at this time on the last 7 days', options: { bullet: true } },
  ], { x: 0.85, y: 2.3, w: 3.55, h: 4.0, fontSize: 14, paraSpaceAfter: 8 })
  card(s, 4.8, 1.5, 4.0, 5.0, 'ehr', C.background2)
  text(s, 'EHR features · 31', { x: 5.05, y: 1.7, w: 3.5, h: 0.5, fontSize: 20, bold: true, color: C.accent5 })
  text(s, [
    { text: 'Age, sex, BMI, smoking, alcohol', options: { bullet: true, breakLine: true } },
    { text: 'Diabetes type and duration, complications', options: { bullet: true, breakLine: true } },
    { text: 'HbA1c, glycated albumin, fasting and post-meal glucose, C-peptide', options: { bullet: true, breakLine: true } },
    { text: 'Lipids, creatinine, eGFR', options: { bullet: true, breakLine: true } },
    { text: 'Drug classes: insulin, metformin, SGLT2, DPP-4, GLP-1 …', options: { bullet: true } },
  ], { x: 5.05, y: 2.3, w: 3.55, h: 4.0, fontSize: 14, paraSpaceAfter: 8 })
  card(s, 9.0, 1.5, 3.7, 5.0, 'model', C.text2)
  circleIcon(s, ic.brain, 10.35, 1.75, 1.0, C.accent1, 'model')
  text(s, 'HistGradientBoosting × 3', { x: 9.25, y: 2.95, w: 3.2, h: 0.5, fontSize: 17, bold: true, color: C.background1, align: 'center' })
  text(s, [
    { text: 'One model per horizon: 30 min, 1 h, 2 h', options: { bullet: true, breakLine: true } },
    { text: 'Accuracy mode and safety mode thresholds', options: { bullet: true, breakLine: true } },
    { text: 'Each person weighted equally', options: { bullet: true, breakLine: true } },
    { text: '2-hour model v9: +12 history and trend features', options: { bullet: true, breakLine: true } },
    { text: 'Runs live in under a second', options: { bullet: true } },
  ], { x: 9.3, y: 3.55, w: 3.2, h: 2.8, fontSize: 14, color: C.background2, paraSpaceAfter: 8 })
  s.addNotes('Stream features describe the last minutes to days of glucose, insulin, meals and activity, and the person\'s own history at this time of day. EHR features come from the clinical record. Gradient-boosted trees handle missing values natively, which matters because not every dataset has every signal.')

  // ── 7. validation ──────────────────────────────────────────
  s = content('Every number is measured on people the model never saw', 'Model')
  const val = [
    ['1', 'Person-grouped 7-fold CV', 'A person is never in training and testing at once'],
    ['2', 'Nested thresholds', 'Alert thresholds chosen inside the training folds only'],
    ['3', 'Leave-one-dataset-out', 'Train on 4 datasets, test on a 5th'],
    ['4', 'External test: OhioT1DM', 'New hospital, devices and patients'],
    ['5', 'Bias & fit audit', 'Sex, age, BMI, diabetes type, time of day, calibration, train-vs-test gap'],
  ]
  val.forEach(([n, head, body], i) => {
    const y = 1.55 + i * 1.03
    s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.75, h: 0.75, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: 'n' + i })
    text(s, n, { x: 0.8, y, w: 0.75, h: 0.75, fontSize: 22, bold: true, color: C.background1, align: 'center', valign: 'middle' })
    text(s, head, { x: 1.85, y: y + 0.02, w: 4.6, h: 0.75, fontSize: 19, bold: true, color: C.text2, valign: 'middle' })
    text(s, body, { x: 6.3, y: y + 0.02, w: 6.4, h: 0.75, fontSize: 15, valign: 'middle' })
  })
  s.addNotes('The most common mistake in glucose prediction is testing on the same people you trained on. We never do that: all numbers come from people held out of training, with thresholds tuned without seeing them, plus a fully external dataset.')

  // ── 8. headline results ────────────────────────────────────
  pres.addSection({ title: 'Results' })
  s = content(`Early warning that works: ${pc(cv['30m'].accuracy_mode.all_open.accuracy)} at 30 minutes, ${pc(cv['2h'].accuracy_mode.all_open.accuracy)} at 2 hours`, 'Results')
  H.forEach((h, i) => {
    const r = cv[h].accuracy_mode.all_open, o = fin.ohio_external_validation[h].accuracy_mode
    const x = 0.6 + i * 4.15, fill = [C.accent2, C.accent3, C.accent1][i]
    card(s, x, 1.6, 3.9, 4.9, 'res' + i, C.background2)
    text(s, `Next ${HL[h]}`, { x: x + 0.3, y: 1.85, w: 3.3, h: 0.5, fontSize: 18, bold: true, color: C.text2 })
    text(s, pc(r.accuracy), { x: x + 0.3, y: 2.4, w: 3.3, h: 1.2, fontSize: 60, bold: true, color: fill, fontFace: 'Cambria' })
    text(s, 'accuracy on unseen people', { x: x + 0.3, y: 3.6, w: 3.3, h: 0.4, fontSize: 14, color: C.accent6 })
    text(s, [
      { text: `ROC-AUC ${r.roc_auc.toFixed(3)}`, options: { breakLine: true } },
      { text: `Spikes caught ${pc(r.recall)} (safety mode ${pc(cv[h].safety_mode.all_open.recall)})`, options: { breakLine: true } },
      { text: `OhioT1DM external: ${pc(o.accuracy)}` },
    ], { x: x + 0.3, y: 4.2, w: 3.4, h: 2.0, fontSize: 15, paraSpaceAfter: 6 })
  })
  s.addNotes('These are the headline results on 187 held-out people. The 95% confidence intervals are about ±0.4 to ±0.9 points. On the external Ohio dataset, never used in training, performance is essentially the same, which shows the model generalises.')

  // ── 9. chart vs baseline ───────────────────────────────────
  s = content('GlucoTwin beats the "already high" rule, and the margin grows with the horizon', 'Results')
  const labels = H.map(h => HL[h])
  barChart(s, { x: 0.6, y: 1.5, w: 8.2, h: 5.2, cats: labels, min: 75, max: 100, step: 5, name: 'baseline',
    title: 'Accuracy on people the model never saw (%)', series: [
      { name: 'Simple rule (glucose > 180 now)', color: HEX.accent6, values: H.map(h => 100 * cv[h].accuracy_mode.all_open.rule_accuracy) },
      { name: 'GlucoTwin (cross-validation)', color: HEX.accent1, values: H.map(h => 100 * cv[h].accuracy_mode.all_open.accuracy) },
      { name: 'GlucoTwin on OhioT1DM (external)', color: HEX.accent2, values: H.map(h => 100 * fin.ohio_external_validation[h].accuracy_mode.accuracy) },
    ] })
  text(s, [
    { text: 'Why a baseline matters', options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
    { text: 'If glucose is already above 180, a spike "in the next 30 minutes" is almost certain. That rule alone is right most of the time.', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: `The twin adds the hard part: spotting rises from normal glucose. Its ranking quality (AUC ${cv['2h'].accuracy_mode.all_open.roc_auc.toFixed(2)} at 2 h) is far above the rule's.`, options: {} },
  ], { x: 9.1, y: 1.7, w: 3.6, h: 4.8, fontSize: 15 })
  s.addNotes('We always compare with the simplest possible rule. Short horizons are easy for both; the further ahead we look, the more the model adds. Ohio, the external test, tracks the cross-validation closely.')

  // ── 10. type 2 & fairness ──────────────────────────────────
  s = content('Strongest for Type 2 diabetes, and fair across patient groups', 'Results')
  barChart(s, { x: 0.6, y: 1.5, w: 6.4, h: 5.2, cats: labels, min: 75, max: 100, step: 5, name: 'type',
    title: 'Accuracy by diabetes type (%)', series: [
      { name: 'Type 1 (87 people)', color: HEX.accent5, values: H.map(h => 100 * cv[h].accuracy_mode.type1.accuracy) },
      { name: 'Type 2 (100 people)', color: HEX.accent1, values: H.map(h => 100 * cv[h].accuracy_mode.type2.accuracy) },
    ] })
  const g = (k, grp) => fin.bias[k].find(r => r[k.split(' (')[0]] === grp)
  const fair = [['Group (Type 2 & Type 1, Shanghai)', '2-h accuracy', 'Calibration gap']].concat([
    ['Women', 'sex (Shanghai, real EHR)', 'female'], ['Men', 'sex (Shanghai, real EHR)', 'male'],
    ['Age < 40', 'age_group (Shanghai, real EHR)', '<40'], ['Age 40–60', 'age_group (Shanghai, real EHR)', '40-60'],
    ['Age 60+', 'age_group (Shanghai, real EHR)', '60+'], ['BMI < 23', 'bmi_group (Shanghai, real EHR)', '<23'],
    ['BMI 23–27.5', 'bmi_group (Shanghai, real EHR)', '23-27.5'], ['BMI 27.5+', 'bmi_group (Shanghai, real EHR)', '27.5+'],
  ].map(([lab, k, grp]) => { const r = g(k, grp); return [lab, pc(r.accuracy), `${(100 * r.calibration_gap).toFixed(1)} pts`] }))
  s.addTable(fair.map((r, i) => r.map(c => ({ text: c, options: i === 0 ? { bold: true, color: C.background1, fill: { color: C.text2 } } : {} }))),
    { x: 7.3, y: 1.6, w: 5.4, colW: [2.6, 1.4, 1.4], fontSize: 13, color: C.text1, border: { type: 'solid', pt: 0.5, color: HEX.lt2 }, rowH: 0.5, objectName: 'fairness' })
  s.addNotes('Type 2 patients, the main group in India, get the most accurate predictions. Accuracy is similar for women and men and across age and BMI groups, and predicted risk matches the observed rate within about one point in every group.')

  // ── 11. honest findings ────────────────────────────────────
  s = content('What we learned by testing every idea', 'Results')
  const first = abl.cold_start_shanghai['first 24 h of a recording']
  const coldGain = (100 * first.fusion_minus_cgm.accuracy_diff_mean).toFixed(1)
  card(s, 0.6, 1.5, 6.0, 5.0, 'helped', C.background2)
  circleIcon(s, ic.check, 0.85, 1.7, 0.7, C.accent4, 'helped')
  text(s, 'Helped', { x: 1.75, y: 1.75, w: 4.5, h: 0.6, fontSize: 22, bold: true, color: C.accent4, valign: 'middle' })
  text(s, [
    { text: 'More diverse patients: 21 → 187 people from 4 countries', options: { bullet: true, breakLine: true } },
    { text: 'Personal memory: "spikes at this time in the last 7 days"', options: { bullet: true, breakLine: true } },
    { text: 'Cleaning the data (removing a future leak, fixing units)', options: { bullet: true, breakLine: true } },
    { text: 'Shorter horizons for urgent alerts; safety-mode thresholds', options: { bullet: true, breakLine: true } },
    { text: `Type 2 data: the most accurate group (${pc(cv['2h'].accuracy_mode.type2.accuracy)} at 2 h)`, options: { bullet: true } },
  ], { x: 0.85, y: 2.55, w: 5.5, h: 3.8, fontSize: 17, paraSpaceAfter: 12 })
  card(s, 6.8, 1.5, 5.9, 5.0, 'nothelp', C.background2)
  circleIcon(s, ic.cross, 7.05, 1.7, 0.7, C.accent2, 'nothelp')
  text(s, 'Did not help (measured, not guessed)', { x: 7.95, y: 1.75, w: 4.6, h: 0.6, fontSize: 20, bold: true, color: C.accent2, valign: 'middle' })
  text(s, [
    { text: 'Bigger / deeper models: they only memorise training patients*', options: { bullet: true, breakLine: true } },
    { text: 'Even perfect knowledge of future meals: +0.2 points*', options: { bullet: true, breakLine: true } },
    { text: 'Per-patient recalibration on day 1–3: no gain*', options: { bullet: true, breakLine: true } },
    { text: `EHR on top of CGM: +${coldGain} points in a patient's first day (not significant), ~0 after`, options: { bullet: true, breakLine: true } },
    { text: 'Extra wearable features (steps, calories): +0.0 points*', options: { bullet: true } },
  ], { x: 7.05, y: 2.55, w: 5.4, h: 3.8, fontSize: 17, paraSpaceAfter: 12 })
  text(s, '* measured in the earlier 87-person study (v6); the other results are from the 187-person v7 evaluation', { x: 0.6, y: 6.6, w: 12.1, h: 0.3, fontSize: 11, color: C.accent6 })
  s.addNotes('We ran an experiment for every idea instead of assuming. Algorithm and tuning changes moved accuracy by about a tenth of a point, so the remaining gains have to come from new information that is known at prediction time.')

  // ── 11b. verified error analysis ───────────────────────────
  s = content('Where the remaining error comes from', 'Results')
  {
    const A = inv.A_label_reliability.all_open, E = inv.E_cgm_vs_fingerstick.overall, ep = inv.B_missed_episodes.serious
    const cal = H.map(h => (100 * inv.C_calibration[h].all.ece).toFixed(2)).join(' / ')
    const cards = [
      [pc(A.uncertain_share_of_errors), 'of 2-hour errors happen where the actual peak was 162–198 mg/dL, i.e. within ±10 % of 180',
       `Accuracy ${pc(A.accuracy_clear)} on clear outcomes vs ${pc(A.accuracy_uncertain)} near 180 (${A.rows.toLocaleString('en-US')} moments, 187 people)`, C.accent2],
      [`${E.mard_pct.toFixed(1)}%`, 'mean difference between CGM and fingerstick readings (MARD): the reference itself is noisy',
       `${E.pairs.toLocaleString('en-US')} pairs from ${E.people} Shanghai patients; median ${E.median_ard_pct.toFixed(1)}%; CGM reads ${Math.abs(E.median_bias_pct).toFixed(1)}% low`, C.accent5],
      [pc(ep.miss_rate), 'of serious spike episodes get no warning 30–120 min ahead',
       `${ep.missed.toLocaleString('en-US')} of ${ep.episodes.toLocaleString('en-US')} episodes (peak ≥ 250 or ≥ 3 consecutive 15-min bins > 180)`, C.accent1],
    ]
    cards.forEach(([big, what, basis, col], i) => {
      const x = 0.6 + i * 4.15
      card(s, x, 1.55, 3.9, 4.35, 'err' + i, C.background2)
      text(s, big, { x: x + 0.3, y: 1.8, w: 3.3, h: 1.0, fontSize: 48, bold: true, color: col, fontFace: 'Cambria' })
      text(s, what, { x: x + 0.3, y: 2.9, w: 3.3, h: 1.4, fontSize: 16, color: C.text2 })
      text(s, basis, { x: x + 0.3, y: 4.45, w: 3.3, h: 1.3, fontSize: 12, color: C.accent6 })
    })
    text(s, `Calibration on held-out people is good (ECE ${cal} % at 30 min / 1 h / 2 h). "Near 180" uses the actual outcome, so it splits the error rather than selecting cases in advance. Fingerstick meters have their own error.`,
      { x: 0.6, y: 6.1, w: 12.1, h: 0.6, fontSize: 12, color: C.text1 })
  }
  s.addNotes('Every number here is re-derived from the held-out predictions by an automated verifier. About half of the remaining 2-hour errors sit right at the 180 line, where even the sensor and a fingerstick disagree by over 10 percent. The rest are mostly spikes that start later from normal glucose, often after a meal eaten after the prediction. That is why the next gains need new information, such as meal announcements, rather than a new algorithm.')

  // ── 11c. v9: new information, tested once on locked-away people ──
  if (v9) {
    s = content(`New information, tested once on ${v9.test_people} people locked away in advance`, 'Results')
    const t7 = v9.test.v7_config, t9 = v9.test.v9, vs = v9.test_v9_vs_v7_config
    const A = v9.alert_policy.arms, a7 = A['baseline + v7 policy'], a9 = A['candidate + v9 policy']
    const apd = v9.alert_policy.alerts_per_patient_day
    const pc2 = v => `${(100 * v).toFixed(2)}%`
    const fad = a => (a.false_alerts / (v9.test_rows / 96)).toFixed(2)     // from counts: no double rounding
    const blocks = [
      ['Prediction (2 h)', C.accent1, [
        `Accuracy ${pc2(t7.accuracy)} → ${pc2(t9.accuracy)} (change within noise)`,
        `Ranking (AUC) ${t7.roc_auc.toFixed(4)} → ${t9.roc_auc.toFixed(4)}: +${vs.roc_auc_diff_mean.toFixed(4)}, significant`,
        `OhioT1DM external ${pc2(t7.ohio_accuracy)} → ${pc2(t9.ohio_accuracy)}`,
        `Type 2 ${pc2(t7.type2_accuracy)} → ${pc2(t9.type2_accuracy)}`]],
      ['Alerts (serious spikes)', C.accent4, [
        `Warned 30–120 min ahead: ${a7.serious_warned.toLocaleString('en-US')} → ${a9.serious_warned.toLocaleString('en-US')} of ${a9.serious_episodes.toLocaleString('en-US')} episodes`,
        `False alerts per patient-day ${fad(a7)} → ${fad(a9)}`,
        `All alerts per day ${apd['baseline + v7 policy'].toFixed(1)} → ${apd['candidate + v9 policy'].toFixed(1)}: warns only while glucose is still in range`]],
    ]
    blocks.forEach(([head, col, lines], i) => {
      const x = 0.6 + i * 6.2
      card(s, x, 1.5, 5.9, 3.2, 'v9-' + i, C.background2)
      text(s, head, { x: x + 0.3, y: 1.7, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: col })
      text(s, lines.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < lines.length - 1 } })),
        { x: x + 0.3, y: 2.35, w: 5.4, h: 2.2, fontSize: 16, paraSpaceAfter: 10 })
    })
    text(s, [
      { text: 'Tested: ', options: { bold: true, color: C.text2 } },
      { text: 'longer personal history, recorded meals and tablets, multi-window trends (all known at prediction time). ' },
      { text: 'Every choice was made on 150 development people; the 37 test people were scored once. ', options: {} },
      { text: 'Served: v9 2-hour model + early-warning policy; 30 min / 1 h unchanged. No open Indian Type 2 CGM dataset was available.', options: { color: C.accent6 } },
    ], { x: 0.6, y: 5.0, w: 12.1, h: 1.1, fontSize: 14 })
    s.addNotes('We tested new information instead of new algorithms, with a test set locked away before any experiment. The new features improve ranking slightly but not accuracy in a measurable way. The bigger practical gain is the alert policy: warning only while glucose is still in range halves the number of alerts and warns of slightly more serious spikes, without more false alerts.')
  }

  // ── 12. dashboard ──────────────────────────────────────────
  pres.addSection({ title: 'Prototype' })
  s = content('What the doctor sees', 'Prototype')
  const shot = path.join(ROOT, 'docs', 'dashboard.png')
  if (fs.existsSync(shot)) {
    s.addImage({ path: shot, x: 0.6, y: 1.45, w: 8.4, h: 5.25, sizing: { type: 'contain', w: 8.4, h: 5.25 }, objectName: 'dashboard' })
  }
  text(s, [
    { text: 'Patient input', options: { bold: true, color: C.text2, breakLine: true } },
    { text: 'EHR, labs, CGM chart, data completeness', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: 'Digital twin & AI prediction', options: { bold: true, color: C.text2, breakLine: true } },
    { text: 'Live probability from FastAPI, early warning', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: 'Daily context (what-if)', options: { bold: true, color: C.text2, breakLine: true } },
    { text: 'Meal, insulin or tablets taken now', options: { breakLine: true } },
    { text: ' ', options: { breakLine: true } },
    { text: 'Monitor · Trends · Reports · My data', options: { bold: true, color: C.text2, breakLine: true } },
    { text: 'Replay, time in range, printable report, upload your own CGM file' },
  ], { x: 9.3, y: 1.6, w: 3.4, h: 5.0, fontSize: 14 })
  s.addNotes('The dashboard is a React app talking to the FastAPI backend. Every number on it comes from the API: the probability, the thresholds, the data completeness. Demo patients are real open-data recordings replayed moment by moment, and none of them was in the training data.')

  // ── 12b. demo workflow ─────────────────────────────────────
  s = content('Demo workflow: from patient to prediction in five steps', 'Prototype')
  const flow = [
    ['1', 'Select a patient', '30 open-data demo patients (Type 1 and Type 2) or your own uploaded CGM file'],
    ['2', 'Pick the moment', 'Replay slider sets "now"; the model only sees data up to that moment'],
    ['3', 'Review the record', 'EHR and labs, CGM history, sensor completeness'],
    ['4', 'Add daily context', 'Optional what-if: meal carbs, rapid insulin, oral tablets'],
    ['5', 'Generate & compare', 'Probabilities for 30 min, 1 h, 2 h; then the replay shows what really happened'],
  ]
  flow.forEach(([n, head, body], i) => {
    const y = 1.55 + i * 1.03
    s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.75, h: 0.75, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: 'flow' + i })
    text(s, n, { x: 0.8, y, w: 0.75, h: 0.75, fontSize: 22, bold: true, color: C.background1, align: 'center', valign: 'middle' })
    text(s, head, { x: 1.85, y: y + 0.02, w: 3.6, h: 0.75, fontSize: 19, bold: true, color: C.text2, valign: 'middle' })
    text(s, body, { x: 5.4, y: y + 0.02, w: 7.3, h: 0.75, fontSize: 15, valign: 'middle' })
  })
  s.addNotes('This is exactly what we will do live. If the API is unavailable, we show the recorded screenshots and the verified numbers instead, and say so.')

  // ── 13. architecture ───────────────────────────────────────
  s = content('Architecture', 'Prototype')
  s.addImage({ path: path.join(ROOT, 'docs', 'architecture.png'), x: 0.6, y: 1.35, w: 12.1, h: 5.45, sizing: { type: 'contain', w: 12.1, h: 5.45 }, objectName: 'architecture' })
  s.addNotes('Data sources, harmonisation, feature fusion, three models with validation, and serving through FastAPI to a React dashboard. Stack: Python, scikit-learn, FastAPI, React.')

  // ── 13b. limitations ───────────────────────────────────────
  s = content('Limitations and clinical safety', 'Next steps')
  {
    const ep = inv.B_missed_episodes.serious, A = inv.A_label_reliability.all_open, E = inv.E_cgm_vs_fingerstick.overall
    const lim = [
      ['Not clinically validated', 'A research prototype trained on open research data; not a medical device and not for treatment decisions.'],
      ['Not an Indian cohort', 'The Type 2 data is from Shanghai, China. Validation with Indian Type 2 patients is future work.'],
      ['Missed serious spikes', `${pc(ep.miss_rate)} of ${ep.episodes.toLocaleString('en-US')} serious episodes got no warning 30–120 min ahead (all 187 people, event level).`],
      ['Errors near the 180 line', `${pc(A.uncertain_share_of_errors)} of 2-hour errors have a peak within ±10% of 180 mg/dL; CGM and fingerstick differ by MARD ${E.mard_pct}%.`],
      ['False alarms', 'Even with the in-range early-warning rule, about 1.3 false alerts per patient-day on the locked test set.'],
      ['What-if and uploads', 'Daily-context scenarios and individual uploads are not separately validated.'],
    ]
    lim.forEach(([head, body], i) => {
      const x = 0.6 + (i % 2) * 6.2, y = 1.5 + Math.floor(i / 2) * 1.75
      card(s, x, y, 5.9, 1.55, 'lim' + i, C.background2)
      text(s, head, { x: x + 0.3, y: y + 0.15, w: 5.3, h: 0.45, fontSize: 17, bold: true, color: C.accent2 })
      text(s, body, { x: x + 0.3, y: y + 0.65, w: 5.3, h: 0.85, fontSize: 13 })
    })
  }
  s.addNotes('We are explicit about what the system cannot do. Accuracy, recall and calibration are different measures; none of them means clinical benefit, which has not been tested.')

  // ── 14. impact ─────────────────────────────────────────────
  pres.addSection({ title: 'Next steps' })
  s = content('From prototype to Indian clinics', 'Next steps')
  const next = [
    [ic.flask, 'Clinical validation', 'Prospective study with Indian Type 2 patients and clinicians before any real use'],
    [ic.link, 'Health-record integration', 'Connect to ABHA-linked records (ABDM) for labs and prescriptions'],
    [ic.mobile, 'Affordable sensing', 'Low-cost CGMs and phone alerts for patients and care teams'],
    [ic.lock, 'Privacy by design', 'DPDP-Act consent, encrypted processing, clinician-in-the-loop alerts'],
  ]
  next.forEach(([im, head, body], i) => {
    const x = 0.6 + (i % 2) * 6.2, y = 1.55 + Math.floor(i / 2) * 2.6
    card(s, x, y, 5.9, 2.35, 'next' + i, C.background2)
    circleIcon(s, im, x + 0.3, y + 0.4, 0.95, C.accent1, 'next' + i)
    text(s, head, { x: x + 1.5, y: y + 0.4, w: 4.2, h: 0.55, fontSize: 19, bold: true, color: C.text2 })
    text(s, body, { x: x + 1.5, y: y + 1.0, w: 4.2, h: 1.2, fontSize: 15 })
  })
  s.addNotes('The prototype is not a medical device. The path to impact is clinical validation in India, integration with the national digital health stack, affordable sensors, and privacy-first deployment.')

  // ── 15. closing ────────────────────────────────────────────
  s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: 'Next steps' })
  s.addText('See the spike before it happens', { placeholder: 'title' })
  s.addText(`${pc(cv['30m'].accuracy_mode.all_open.accuracy)} at 30 min · ${pc(cv['60m'].accuracy_mode.all_open.accuracy)} at 1 h · ${pc(cv['2h'].accuracy_mode.all_open.accuracy)} at 2 h — on 187 unseen people, confirmed on an external dataset`, { placeholder: 'body' })
  circleIcon(s, ic.heart, 0.8, 0.8, 1.0, C.accent2, 'logo-end')
  text(s, 'hackers_ · IIT (ISM) Dhanbad  ·  Code (MIT): github.com/KorraSanthosh/happiness_   ·   Thank you', { x: 0.8, y: 6.3, w: 11.7, h: 0.4, fontSize: 16, color: C.background2 })
  s.addNotes('Thank the jury. Summarise: two-stream fusion, honest validation, working dashboard, and a clear path to clinical use in India.')

  await pres.writeFile({ fileName: OUT })
  const { applyTheme } = require(process.env.APPLY_THEME)
  await applyTheme(OUT, THEME)
  console.log('wrote', OUT)
}

main().catch(e => { console.error(e); process.exit(1) })
