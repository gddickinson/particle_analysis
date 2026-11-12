// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py. Tracking and track
// features run in the browser on any site (tracker.js, features.js); detection on new recordings and the motion
// analysis of tracks other than the precomputed demo runs need the Python server.
const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export const NEEDS_SERVER = 'needs the full app running locally (Python server): see the README, "Run the web app"';

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'detect') throw new Error('Detecting particles in your own recording ' + NEEDS_SERVER + '.');
  if (clean === 'motion') throw new Error('The motion analysis of these tracks ' + NEEDS_SERVER + '.');
  return file(clean);
}
