// Verify the actual standalone report using keyboard and pointer events in an offline browser target.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';

const [program, report, screenshot] = process.argv.slice(2);
const dir = await mkdtemp(join(tmpdir(), 'evidence-summary-'));
const chrome = spawn(program, ['--headless=new','--remote-debugging-port=0',`--user-data-dir=${dir}`,'--disable-background-networking','--disable-component-update','--disable-extensions','--disable-sync','--no-first-run','--no-default-browser-check','about:blank'], {stdio:'ignore'});
const closed = once(chrome, 'exit');
function check(v,m){if(!v)throw Error(m)}
let socket, watchdog;
async function validate(){
  let port;
  for(let k=0;k<150 && !port;k++){try{port=(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0]}catch{await delay(50)}}
  check(port, 'Missing loopback browser control port');
  const target=(await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t=>t.type==='page');
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok,no)=>{socket.onopen=ok;socket.onerror=no});
  let id=0,load;const pending=new Map(), external=[],errors=[];
  socket.onmessage=({data})=>{const v=JSON.parse(data);if(v.id&&pending.has(v.id)){const [ok,no]=pending.get(v.id);pending.delete(v.id);v.error?no(Error(v.error.message)):ok(v.result)}if(v.method==='Page.loadEventFired'&&load)load();if(v.method==='Network.requestWillBeSent'&& /^https?:/i.test(v.params.request.url))external.push(v.params.request.url);if(v.method==='Runtime.exceptionThrown')errors.push(v.params.exceptionDetails.text)};
  function send(method,params={}){return new Promise((ok,no)=>{const n=++id;pending.set(n,[ok,no]);socket.send(JSON.stringify({id:n,method,params}))})}
  async function js(expression){const v=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});check(!v.exceptionDetails,v.exceptionDetails?.exception?.description??'Browser evaluation failed');return v.result.value}
  async function click(selector){const pos=await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);for(const type of ['mousePressed','mouseReleased'])await send('Input.dispatchMouseEvent',{type,...pos,button:'left',clickCount:1})}
  async function key(name,code){for(const type of ['keyDown','keyUp'])await send('Input.dispatchKeyEvent',{type,key:name,code:name,windowsVirtualKeyCode:code,nativeVirtualKeyCode:code})}
  async function viewport(width,height){await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false})}
  const version=await send('Browser.getVersion');await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
  await send('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:0,uploadThroughput:0});
  await viewport(1280,850);const loaded=new Promise(ok=>{load=ok});await send('Page.navigate',{url:pathToFileURL(resolve(report)).href});await loaded;
  const base=await js(`({data:JSON.parse(document.getElementById('study-data').textContent),profiles:[...document.querySelectorAll('#pairs article h3')].map(e=>e.textContent),images:document.images.length})`);
  check(base.images===0 && base.profiles.join(',')==='Living room,Bridge','Missing summary cases or unexpected images');
  const verifies=[];
  for(const scene of ['living_room','bridge']){
    await click('#choose-'+scene);
    check(await js(`document.getElementById('choose-${scene}').getAttribute('aria-pressed')==='true' && document.getElementById('case-details').textContent.includes(${JSON.stringify(scene==='bridge'?'Bridge':'Living room')})`),'Scene switch failed');
    if(!await js('document.getElementById("target-details").open'))await click('#target-details > summary');
    const shown=await js(`document.querySelectorAll('#view-table tbody tr').length`);check(shown===base.data.external[scene].planned,'Incomplete target-view table');verifies.push(scene);
  }
  await click('#choose-bridge');await key('Home',36);check(await js(`document.activeElement.id==='choose-living_room' && document.getElementById('choose-living_room').getAttribute('aria-pressed')==='true'`),'Keyboard scene control failed');
  const legacyText=base.data.external.living_room.methods['coverage-4 MS-L1'].reasons?'no_comparable_source':'no measured rank arrays';
  await click('#reason-details > summary');check(await js(`document.getElementById('reason-details').open && document.getElementById('reason-table').textContent.includes(${JSON.stringify(legacyText)})`),'Legacy reason dictionary failed');
  await js(`(()=>{const s=document.getElementById('method');s.focus();s.value='coverage-16 B-CPU-LR';s.dispatchEvent(new Event('change',{bubbles:true}));return s.value})()`);
  const newText=base.data.external.living_room.methods['coverage-16 B-CPU-LR'].reasons?'queried_material_unavailable':'no measured rank arrays';
  check(await js(`document.getElementById('reason-table').textContent.includes(${JSON.stringify(newText)}) && document.getElementById('method').value.includes('coverage-16')`),'Method control or new reason dictionary failed');
  await click('a[href="#geometry"]');check(await js(`location.hash==='#geometry' && document.getElementById('geometry').getBoundingClientRect().top>=-25 && document.getElementById('geometry').getBoundingClientRect().top<innerHeight-45`),'Physical visibility link failed');
  await viewport(375,812);const mobile=await js('({viewport:innerWidth,content:document.documentElement.scrollWidth})');check(mobile.content<=mobile.viewport+1,'Broken mobile reflow');
  await viewport(1280,850);await js('scrollTo(0,0)');if(screenshot){const img=await send('Page.captureScreenshot',{format:'png'});await writeFile(screenshot,Buffer.from(img.data,'base64'))}
  check(!external.length&&!errors.length,'External HTTP request or page script error');
  return {browser:version.product,network:'offline',external_http_requests_observed:external.length,images:base.images,verified_scenes:verifies,planned_rows:Object.fromEntries(verifies.map(k=>[k,base.data.external[k].planned])),keyboard_scene:'Home switched to Living room',method_selection:'Changed the native selector to coverage-16 B-CPU-LR',checked_details:['per-target denominators','legacy and many-source reason dictionaries'],anchor:'#geometry',mobile,desktop_viewport_css:[1280,850],script_errors:errors};
}
try{const limit=new Promise((_,reject)=>{watchdog=setTimeout(()=>reject(Error('Browser acceptance timed out')),30000)});process.stdout.write(JSON.stringify(await Promise.race([validate(),limit]))+'\n')}
finally{clearTimeout(watchdog);socket?.close();chrome.kill('SIGTERM');await closed;await rm(dir,{recursive:true,force:true,maxRetries:3,retryDelay:100})}
