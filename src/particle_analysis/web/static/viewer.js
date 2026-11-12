// Movie viewer: draws the 8-bit display copy of the recording on a canvas with a frame slider and play button, and
// lets the page draw overlays (detections, tracks, ground truth) in recording pixel coordinates.
// Coordinates are pixel centres (the detector's convention): raw pixel (x, y) is drawn at the centre of its pixel.
import { animate, esc } from './core.js';

const LUT = (() => { // green "fluorescence" lookup table
  const t = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) { const v = i / 255; t[i * 3] = 255 * v ** 2.2 * 0.7; t[i * 3 + 1] = 255 * v ** 0.9; t[i * 3 + 2] = 255 * v ** 2 * 0.5; }
  return t;
})();

export const COLORS = ['#7cff6b', '#ff5ad1', '#5ab8ff', '#ffd23f', '#ff8a3d', '#b48cff', '#3dffe0', '#ff5c7a', '#c6ff3d', '#ff9ee6'];
export const MOTION_COLORS = { diffusive: '#5ab8ff', normal_diffusion: '#5ab8ff', confined: '#ffd23f', directed: '#ff5ad1',
  immobile: '#94a3c6', anomalous: '#ff8a3d' };

// host: element; data: dataset with .movie (or null: blank field sized to the detections); draw(ctx, frame, toCanvas)
export function createViewer(host, data, draw, { label = 'Recording' } = {}) {
  const mv = data.movie;
  const nFrames = mv ? mv.n : Math.max(1, ...data.detections.frame.map(f => f + 1));
  const W = mv ? mv.width : Math.ceil(Math.max(16, ...data.detections.x) + 4);
  const H = mv ? mv.height : Math.ceil(Math.max(16, ...data.detections.y) + 4);
  const bin = mv ? mv.bin : 1;
  const zoom = Math.max(1, Math.ceil(640 / Math.max(W, H)));
  host.innerHTML = `<div class="viewer">
    <div class="stage"><canvas width="${W * zoom}" height="${H * zoom}" role="img" aria-label="${esc(label)}: frame view with overlays"></canvas></div>
    <div class="vbar"><button type="button" class="alt" data-play>Play</button>
      <label class="sr" for="vf-${host.id}" style="margin:0">Frame</label>
      <input type="range" id="vf-${host.id}" min="0" max="${nFrames - 1}" value="0">
      <span class="fr"></span></div></div>`;
  const cv = host.querySelector('canvas'), g = cv.getContext('2d');
  const slider = host.querySelector('input'), fr = host.querySelector('.fr'), play = host.querySelector('[data-play]');
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const og = off.getContext('2d'), img = og.createImageData(W, H);
  // raw pixel-centre coordinate -> canvas coordinate
  const toCanvas = (x, y) => [((x - (bin - 1) / 2) / bin + 0.5) * zoom, ((y - (bin - 1) / 2) / bin + 0.5) * zoom];
  toCanvas.scale = zoom / bin;
  let frame = 0, playing = false, last = 0;
  function render() {
    if (mv) {
      const px = mv.frames[frame], d = img.data;
      for (let i = 0; i < px.length; i++) { const v = px[i] * 3; d[i * 4] = LUT[v]; d[i * 4 + 1] = LUT[v + 1]; d[i * 4 + 2] = LUT[v + 2]; d[i * 4 + 3] = 255; }
      og.putImageData(img, 0, 0);
      g.imageSmoothingEnabled = false;
      g.drawImage(off, 0, 0, W * zoom, H * zoom);
    } else {
      g.fillStyle = '#020308'; g.fillRect(0, 0, cv.width, cv.height);
    }
    draw(g, frame, toCanvas);
    fr.textContent = `${frame + 1} / ${nFrames}`;
  }
  const api = {
    get frame() { return frame; },
    set(f) { frame = Math.max(0, Math.min(nFrames - 1, f)); slider.value = frame; render(); },
    render,
  };
  slider.addEventListener('input', () => api.set(+slider.value));
  play.addEventListener('click', () => { playing = !playing; play.textContent = playing ? 'Pause' : 'Play'; });
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const stop = animate(t => {
    if (!playing || t - last < (reduce ? 400 : 110)) return;
    last = t; api.set((frame + 1) % nFrames);
  });
  render();
  return api;
}

export function ring(g, [x, y], r, color, width = 1.5) {
  g.strokeStyle = color; g.lineWidth = width; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.stroke();
}

export function cross(g, [x, y], r, color) {
  g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
  g.moveTo(x - r, y - r); g.lineTo(x + r, y + r); g.moveTo(x + r, y - r); g.lineTo(x - r, y + r); g.stroke();
}
