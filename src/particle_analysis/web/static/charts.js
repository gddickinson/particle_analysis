// Small SVG charts (no library): histogram and line/point chart. Values are drawn as given; non-finite values are
// skipped and counted in the caption.
import { esc } from './core.js';

export const fmt = (v, d = 3) => v === null || v === undefined || !isFinite(v) ? '–'
  : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v).toFixed(d);

function ticks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}

const W = 420, H = 230, L = 48, R = 12, T = 12, B = 38;

export function histogram(values, { xlabel = '', bins = 20, color = 'var(--accent3)', title = '' } = {}) {
  const v = values.filter(x => x !== null && isFinite(x));
  const skipped = values.length - v.length;
  if (!v.length) return `<p class="mute small">No finite values${skipped ? ` (${skipped} skipped)` : ''}.</p>`;
  let lo = Math.min(...v), hi = Math.max(...v);
  if (hi === lo) { lo -= 0.5; hi += 0.5; }
  const nb = Math.min(bins, Math.max(5, Math.ceil(Math.sqrt(v.length)) + 2)), w = (hi - lo) / nb;
  const counts = new Array(nb).fill(0);
  for (const x of v) counts[Math.min(nb - 1, Math.floor((x - lo) / w))]++;
  const cmax = Math.max(...counts), sx = x => L + (x - lo) / (hi - lo) * (W - L - R), sy = c => H - B - c / cmax * (H - T - B);
  const bars = counts.map((c, i) => `<rect class="bar" style="fill:${color}" x="${sx(lo + i * w) + 0.5}" y="${sy(c)}" width="${Math.max(0.5, sx(lo + (i + 1) * w) - sx(lo + i * w) - 1)}" height="${H - B - sy(c)}"><title>${fmt(lo + i * w)} – ${fmt(lo + (i + 1) * w)}: ${c}</title></rect>`).join('');
  const xt = ticks(lo, hi).map(t => `<text x="${sx(t)}" y="${H - B + 14}" text-anchor="middle">${fmt(t, 2)}</text>`).join('');
  const yt = ticks(0, cmax, 4).filter(t => Number.isInteger(t)).map(t => `<text x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title || 'Histogram of ' + xlabel)}">
    <line class="axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}"/>
    ${bars}${xt}${yt}<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(xlabel)}</text></svg>
    <p class="mute small" style="margin:2px 0 0">n = ${v.length}${skipped ? `, ${skipped} not finite (skipped)` : ''}</p>`;
}

// series: [{x:[], y:[], color, label, dash, points}]
export function lineChart(series, { xlabel = '', ylabel = '', title = '' } = {}) {
  const pts = series.flatMap(s => s.x.map((x, i) => [x, s.y[i]])).filter(([x, y]) => isFinite(x) && isFinite(y) && x !== null && y !== null);
  if (!pts.length) return '<p class="mute small">Nothing to plot.</p>';
  let x0 = Math.min(0, ...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  let y0 = Math.min(0, ...pts.map(p => p[1])), y1 = Math.max(...pts.map(p => p[1]));
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) y1 = y0 + 1;
  const sx = x => L + (x - x0) / (x1 - x0) * (W - L - R), sy = y => H - B - (y - y0) / (y1 - y0) * (H - T - B);
  const body = series.map(s => {
    const p = s.x.map((x, i) => [x, s.y[i]]).filter(([x, y]) => x !== null && y !== null && isFinite(x) && isFinite(y));
    const line = p.length > 1 && !s.points ? `<polyline fill="none" stroke="${s.color}" stroke-width="1.6" ${s.dash ? 'stroke-dasharray="5 3"' : ''} points="${p.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')}"/>` : '';
    const dots = s.points ? p.map(([x, y]) => `<circle cx="${sx(x)}" cy="${sy(y)}" r="3" fill="${s.color}"/>`).join('') : '';
    return line + dots;
  }).join('');
  const xt = ticks(x0, x1).map(t => `<text x="${sx(t)}" y="${H - B + 14}" text-anchor="middle">${fmt(t, 2)}</text>`).join('');
  const yt = ticks(y0, y1, 4).map(t => `<text x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t, 2)}</text>`).join('');
  const legend = series.filter(s => s.label).map(s => `<span style="--c:${s.color}">${esc(s.label)}</span>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title || ylabel + ' against ' + xlabel)}">
    <line class="axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}"/>
    ${body}${xt}${yt}<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(xlabel)}</text>
    <text transform="translate(11 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text></svg>
    ${legend ? `<div class="legend">${legend}</div>` : ''}`;
}

// Sortable table: cols [{key, label, fmt?}], rows objects. Re-renders itself on header click; onRow(row) on click/Enter.
export function sortable(host, cols, rows, { onRow, selected } = {}) {
  let key = cols[0].key, dir = 1;
  const draw = () => {
    const sorted = [...rows].sort((a, b) => {
      const x = a[key], y = b[key];
      if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
      return (x > y ? 1 : -1) * dir;
    });
    host.innerHTML = `<div class="scroll"><table class="tbl"><thead><tr>${cols.map(c => `<th><button type="button" data-k="${c.key}">${esc(c.label)}${c.key === key ? (dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`).join('')}</tr></thead>
      <tbody>${sorted.map((r, i) => `<tr data-i="${rows.indexOf(r)}" ${onRow ? 'class="click" tabindex="0"' : ''} ${selected && selected(r) ? 'class="click sel" tabindex="0"' : ''}>${cols.map(c => `<td>${c.fmt ? c.fmt(r[c.key], r) : esc(r[c.key])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    host.querySelectorAll('th button').forEach(b => b.onclick = () => { if (key === b.dataset.k) dir = -dir; else { key = b.dataset.k; dir = 1; } draw(); });
    if (onRow) host.querySelectorAll('tbody tr').forEach(tr => {
      const go = () => onRow(rows[+tr.dataset.i]);
      tr.onclick = go; tr.onkeydown = e => { if (e.key === 'Enter') go(); };
    });
  };
  draw();
  return { redraw: draw };
}
