// Entry point: lazy page modules, the mode label, a subtle background of blinking diffraction-limited spots, router.
import { $, app, lazy, route, start } from './core.js';
import { modeNote } from './state.js';

const PAGES = {
  'home.js': ['home'], 'recording.js': ['recording'], 'tracks.js': ['tracks'], 'featurespage.js': ['features'],
  'motion.js': ['motion'], 'data.js': ['data'], 'about.js': ['about'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = modeNote();

// Background: a few soft green spots drifting by Brownian steps. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const ps = Array.from({ length: 18 }, () => ({ x: Math.random() * innerWidth, y: Math.random() * innerHeight, a: Math.random() }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 80) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of ps) {
      p.x = (p.x + (Math.random() - 0.5) * 5 + canvas.width) % canvas.width;
      p.y = (p.y + (Math.random() - 0.5) * 5 + canvas.height) % canvas.height;
      p.a = Math.min(1, Math.max(0.15, p.a + (Math.random() - 0.5) * 0.2));
      const grd = g.createRadialGradient(p.x, p.y, 0, p.x, p.y, 9);
      grd.addColorStop(0, `rgba(124,255,107,${0.55 * p.a})`); grd.addColorStop(1, 'rgba(124,255,107,0)');
      g.fillStyle = grd; g.fillRect(p.x - 9, p.y - 9, 18, 18);
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
