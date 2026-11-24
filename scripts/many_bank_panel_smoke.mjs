// Exercise the released local HTML at one known analysis pixel in a disposable offline Chrome profile.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [browser, filename, px, py] = process.argv.slice(2);
const x = Number(px), y = Number(py);
if (!(Number.isInteger(x) && x >= 0 && x < 378 && Number.isInteger(y) && y >= 0 && y < 252)) throw Error('Expected analysis pixel');
const dir = await mkdtemp(join(tmpdir(), 'evidence-bank-browser-'));
const chrome = spawn(browser, [
  '--headless=new', '--remote-debugging-pipe', '--disable-background-networking', '--disable-component-update',
  '--disable-default-apps', '--disable-sync', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
  `--user-data-dir=${dir}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe'] });
const exited = once(chrome, 'exit');
const [write, read] = chrome.stdio.slice(3);
let number = 0, buffer = '', session = ''; const calls = new Map(), listeners = new Map(), remote = [];
read.setEncoding('utf8');
read.on('data', chunk => {
  buffer += chunk;
  for (let end; (end = buffer.indexOf('\0')) >= 0;) {
    const message = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1);
    if (message.id && calls.has(message.id)) {
      const { resolve, reject } = calls.get(message.id); calls.delete(message.id);
      if (message.error) reject(Error(message.error.message)); else resolve(message.result);
    }
    if (message.sessionId === session) {
      if (message.method === 'Network.requestWillBeSent' && /^https?:/i.test(message.params.request.url)) remote.push(new URL(message.params.request.url).origin);
      listeners.get(message.method)?.(message.params);
    }
  }
});
const send = (method, params = {}, targetSession = session) => new Promise((resolve, reject) => {
  const id = ++number; calls.set(id, { resolve, reject });
  write.write(JSON.stringify({ id, method, params, ...(targetSession ? { sessionId: targetSession } : {}) }) + '\0');
});
async function js(expression) {
  const { result, exceptionDetails } = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (exceptionDetails) throw Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
  return result.value;
}
async function exercise() {
  const version = await send('Browser.getVersion');
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  ({ sessionId: session } = await send('Target.attachToTarget', { targetId, flatten: true }));
  await send('Page.enable'); await send('Network.enable');
  await send('Network.emulateNetworkConditions', { offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0 });
  await send('Emulation.setDeviceMetricsOverride', { width: 460, height: 1000, deviceScaleFactor: 1, mobile: false });
  const loaded = new Promise(resolve => listeners.set('Page.loadEventFired', resolve));
  await send('Page.navigate', { url: pathToFileURL(resolve(filename)).href }); await loaded;
  const layers = await js('Promise.all([...document.images].map(async i=>{await i.decode(); return [i.alt,i.naturalWidth,i.naturalHeight]}))');
  if (layers.length !== 6 || layers.some(a => a[1] !== 378 || a[2] !== 252)) throw Error('Browser did not decode six analysis-grid images');
  const position = await js(`(()=>{const i=document.images[0];i.scrollIntoView({block:'center'}); const b=i.getBoundingClientRect();return {x:b.left+(${x}+.5)*b.width/378,y:b.top+(${y}+.5)*b.height/252}})()`);
  await send('Input.dispatchMouseEvent', {type: 'mousePressed', ...position, button: 'left', clickCount: 1});
  await send('Input.dispatchMouseEvent', {type: 'mouseReleased', ...position, button: 'left', clickCount: 1});
  const text = await js('document.querySelector("#pixel").textContent');
  if (!text.startsWith(`Pixel (${x}, ${y}) ·`) || remote.length) throw Error('Offline source-ID interaction failed');
  return { browser: version.product, viewport_css: [460,1000], network_emulation: 'offline', external_http_requests_observed: remote.length, decoded_maps: layers, pixel: [x,y], displayed_text: text };
}
let timer;
try {
  const deadline = new Promise((_, reject) => { timer = setTimeout(() => reject(Error('Offline browser check timed out')), 25000); });
  const result = await Promise.race([exercise(), deadline]);
  process.stdout.write(JSON.stringify(result) + '\n');
} finally {
  clearTimeout(timer); chrome.kill('SIGTERM'); await exited;
  await rm(dir, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
}
