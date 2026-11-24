// Exercise registered source, condition and view controls in a native offline Chrome target.
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {mkdtemp, readFile, writeFile, mkdir, rm} from 'node:fs/promises';
import {dirname, join, relative, resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';

const [program, page, screenshotDir] = process.argv.slice(2);
const root = dirname(resolve(page));
const profile = await mkdtemp(join(tmpdir(), 'evidence-foundation-browser-'));
const chrome = spawn(program, ['--headless=new', '--remote-debugging-port=0', '--user-data-dir='+profile,
  '--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions',
  '--no-first-run','--no-default-browser-check','about:blank'], {stdio:'ignore'});
const closed = once(chrome, 'exit');
let socket, watchdog;
function must(value, message) { if (!value) throw Error(message); }
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

async function verify() {
  let port;
  for (let i = 0; i < 150 && !port; i++) {
    try { port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; }
    catch { await delay(50); }
  }
  must(port, 'Missing isolated Chrome control port');
  const target = (await (await fetch('http://127.0.0.1:'+port+'/json/list')).json()).find(t => t.type === 'page');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok, no) => { socket.onopen = ok; socket.onerror = no; });
  const pending = new Map(), fileRequests = new Set(), external = [], errors = [];
  let serial = 0, load;
  socket.onmessage = ({data}) => {
    const v = JSON.parse(data);
    if (v.id && pending.has(v.id)) {
      const [ok,no] = pending.get(v.id);
      pending.delete(v.id);
      v.error ? no(Error(v.error.message)) : ok(v.result);
    }
    if (v.method === 'Page.loadEventFired') load?.();
    if (v.method === 'Runtime.exceptionThrown') errors.push(v.params.exceptionDetails.exception?.description??v.params.exceptionDetails.text);
    if (v.method === 'Network.requestWillBeSent') {
      const url = v.params.request.url;
      if (url.startsWith('file:')) fileRequests.add(relative(root,fileURLToPath(url)));
      else if (url !== 'about:blank') external.push(url);
    }
  };
  function send(method, params={}) {
    return new Promise((ok,no) => {
      const n = ++serial; pending.set(n,[ok,no]);
      socket.send(JSON.stringify({id:n,method,params}));
    });
  }
  async function js(expression) {
    const v = await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    must(!v.exceptionDetails, v.exceptionDetails?.exception?.description??'Page evaluation failed');
    return v.result.value;
  }
  async function click(selector) {
    const xy = await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});
      if(!e||e.disabled) throw Error('Unavailable control: '+${JSON.stringify(selector)});
      e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect();
      return {x:r.left+r.width/2,y:r.top+r.height/2}})()`);
    for (const type of ['mousePressed','mouseReleased']) {
      await send('Input.dispatchMouseEvent',{type,...xy,button:'left',clickCount:1});
    }
  }
  async function select(id, value) {
    await js(`(()=>{const el=document.getElementById(${JSON.stringify(id)});el.value=${JSON.stringify(String(value))};
      el.dispatchEvent(new Event('change',{bubbles:true}));return el.value})()`);
  }
  async function key(name, keyCode) {
    for (const type of ['keyDown','keyUp']) {
      await send('Input.dispatchKeyEvent',{type,key:name,code:name,windowsVirtualKeyCode:keyCode,nativeVirtualKeyCode:keyCode});
    }
  }
  async function image(id, spec) {
    const result = await js(`(async()=>{const i=document.getElementById(${JSON.stringify(id)});
      await i.decode(); return [i.getAttribute('src'),i.naturalWidth,i.naturalHeight,!!i.getClientRects().length]})()`);
    must(eq(result, [spec.path,...spec.size,true]), id+' failed to decode or display '+spec.path+' '+JSON.stringify(result));
  }
  async function gateClosed() {
    must(await js(`document.querySelector('#evaluation-body').hidden && !document.querySelector('#reference').hasAttribute('src') &&
      !document.querySelector('#error').hasAttribute('src') && document.querySelector('#open-evaluation').getAttribute('aria-expanded')==='false'`),
      'Independent evaluation unexpectedly open');
  }
  async function viewport(width, height, mobile = false) {
    await send('Emulation.setDeviceMetricsOverride', {width,height,deviceScaleFactor:1,mobile});
  }
  const screenshots = {};
  async function capture(name, area, condition) {
    if (area) await js(`document.querySelector(${JSON.stringify(area)}).scrollIntoView({block:'start'})`);
    else await js('scrollTo(0,0)');
    await js('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
    const dims = await js('({width:innerWidth,height:innerHeight,doc:document.documentElement.scrollWidth,scroll_y:Math.round(scrollY)})');
    must(dims.doc <= dims.width + 1, 'Horizontal page overflow in '+name);
    const shot = await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    await writeFile(join(screenshotDir, name+'.png'),Buffer.from(shot.data,'base64'));
    screenshots[name] = {path:'verification/'+name+'.png', ...dims, ...condition};
  }
  await mkdir(screenshotDir,{recursive:true});
  const version = await send('Browser.getVersion');
  for (const domain of ['Page','Runtime','Network']) await send(domain+'.enable');
  await send('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:0,uploadThroughput:0});
  await viewport(1280,850);
  const loaded = new Promise(ok=>{load=ok;});
  await send('Page.navigate',{url:pathToFileURL(resolve(page)).href});
  await loaded;
  const data = JSON.parse(await readFile(join(root,'case-index.json'),'utf8'));
  const inline = await js("JSON.parse(document.getElementById('case-data').textContent)");
  // Standalone native JSON is sorted and the inline payload has the same canonical key order.
  must(eq(inline,data),'Standalone JSON and actual browser input differ');
  await image('prediction',data.cases[0].views[0].target);
  await gateClosed();
  must(await js("document.getElementById('source-pane').hidden && document.getElementById('overlay-pane').hidden"),'Missing plain picture view');
  const initial = [...fileRequests].sort();
  must(eq(initial, ['index.html',data.cases[0].views[0].target.path].sort()), 'Unexpected initial request '+initial);
  await capture('picture',null,{case:data.cases[0].id,view:'v01',layer:'picture'});
  await click('#layer-picture'); await key('ArrowRight',39);
  must(await js("document.activeElement.id==='layer-assessment' && document.activeElement.getAttribute('aria-selected')==='true'"), 'ArrowRight failed');
  await key('End',35);
  must(await js("document.activeElement.id==='layer-sources' && !document.getElementById('source-pane').hidden"), 'End failed');
  await key('Home',36);
  must(await js("document.activeElement.id==='layer-picture' && document.getElementById('source-pane').hidden"), 'Home failed');
  await click('#next-view');
  must(await js("document.getElementById('view').value==='1'"),'Next camera failed');
  await click('#prev-view');
  must(await js("document.getElementById('view').value==='0'"),'Previous camera failed');
  const expected = new Set(['index.html']), tests = [], openedReferences = [], sourcesDecoded = {};
  const nm = x => new Intl.NumberFormat('en-GB').format(x);
  for (const [ci,c] of data.cases.entries()) {
    await select('case',ci);
    const viewed = new Set();
    const byId = new Map(c.sources.map(s=>[s.id,s]));
    c.sources.forEach(s=>expected.add(s.image.path));
    for (const [vi,v] of c.views.entries()) {
      await select('view',vi);
      await image('prediction',v.target);
      expected.add(v.target.path);
      await gateClosed();
      for (const [qi,q] of v.conditions.entries()) {
        await click('[data-condition="'+q.id+'"]');
        await gateClosed();
        await image('prediction',v.target);
        must(await js("document.querySelector('[data-condition][aria-pressed=true]').dataset.condition")===q.id,'Condition switch failed');
        await click('#layer-assessment');
        for (const name of ['rank','state','reason']) {
          await click('[data-map="'+name+'"]');
          await image('map',q[name]);
          expected.add(q[name].path);
          const labels = await js("Array.from(document.querySelectorAll('#legend .label'),x=>[x.dataset.code,x.textContent])");
          must(eq(labels,q.legends[name].map(x=>[x.label,x.label.replaceAll('_',' ')])),'Colour key differs from actual '+name+' record');
          if (name!=='rank') {
            const nums = await js("Array.from(document.querySelectorAll('#legend strong'),x=>x.textContent)");
            must(eq(nums,q.legends[name].map(x=>nm(q.counts[name+'_counts'][x.label]))),'Category counts differ');
          }
          if (ci===0 && vi===0 && qi===0 && name==='state')
            await capture('assessment','#reader',{case:c.id,view:v.id,condition:q.id,layer:'assessment',map:name});
        }
        await click('#layer-sources');
        must(await js("document.getElementById('source-list').textContent")===q.query_ids.join('  '),'Ordered query differs '+q.id);
        const all = ci===0 || q.k===128;
        const indexes = all ? q.query_ids.map((_,i)=>i) : [0,q.k-1];
        for (const i of indexes) {
          await select('source-choice',i);
          const source = byId.get(q.query_ids[i]);
          await image('source-image',source.image);
          const shown = await js("document.getElementById('source-info').textContent");
          must(shown.includes(source.id) && shown.includes(source.analysis.sha256) && shown.includes(source.original.sha256),'Source attribution lost');
          viewed.add(source.id);
        }
        if (ci===1 && q.k===16) {
          await select('source-choice',0); await click('#next-source');
          must(await js("document.getElementById('source-choice').value==='1'"),'Next queried source failed');
          await click('#prev-source'); await image('source-image',byId.get(q.query_ids[0]).image);
          await click('#query-roster>summary');
          must(await js("document.getElementById('query-roster').open"),'Query disclosure failed');
          await capture('sources','#source-pane',{case:c.id,view:v.id,condition:q.id,source:q.query_ids[0],layer:'sources'});
          await click('#query-roster>summary');
        }
        if (ci===1) {
          const row = await js("Array.from(document.querySelectorAll('#budget-table .selected-row td'),x=>x.textContent)");
          must(eq(row,[q.k,q.counts.query_has_comparable,q.counts.selected_gap,q.counts.library_no_comparable,
            q.counts.new_comparable_since_previous,q.counts.new_comparable_since_q4].map(nm)),'Budget audit differs');
          must(await js("document.getElementById('identities').textContent").then(x=>x.includes(v.target_prediction_sha256)),'Full target identity missing');
          if (qi===0) {
            await click('#open-evaluation');
            await image('reference',v.evaluation.reference);
            await click('[data-condition="'+v.conditions[1].id+'"]');
            await gateClosed();
          }
        }
        tests.push({case:c.id,view:v.id,condition:q.id,query_count:q.k,assessment_maps:['rank','state','reason']});
      }
      await click('#open-evaluation');
      await image('reference',v.evaluation.reference);
      await image('error',v.evaluation.error);
      expected.add(v.evaluation.reference.path); expected.add(v.evaluation.error.path);
      const counts = await js("document.getElementById('evaluation-counts').textContent");
      const e = v.evaluation.counts;
      must(counts.includes(nm(e.errors)+' of '+nm(e.labelled)) && counts.includes(nm(e.reference)) && counts.includes('0.05'), 'Evaluation denominator or criterion missing');
      const legends = await js("Array.from(document.querySelectorAll('#error-key .label'),x=>x.textContent)");
      must(eq(legends,v.evaluation.legend.map(x=>x.label)),'Error graphic key differs');
      openedReferences.push({case:c.id,view:v.id,errors:e.errors,labelled:e.labelled,reference:e.reference});
      if(ci===1) await capture('evaluation','#evaluation',{case:c.id,view:v.id,condition:v.conditions.at(-1).id,layer:'independent evaluation'});
      await click('#open-evaluation'); await gateClosed();
    }
    sourcesDecoded[c.id] = viewed.size;
    must(viewed.size===c.sources.length,'Incomplete source traversal');
  }
  await viewport(375,812,true);
  await select('case',1); await click('#layer-assessment'); await click('[data-map="state"]');
  await image('map',data.cases[1].views[0].conditions[0].state);
  await capture('mobile',null,{case:data.cases[1].id,view:'h00',condition:'coverage-4',layer:'assessment',map:'state'});
  await click('#layer-sources');
  const mobile = await js('({width:innerWidth,document_width:document.documentElement.scrollWidth})');
  must(mobile.width===375 && mobile.width>=mobile.document_width,'Mobile source rows overflow');
  await click('#open-evaluation'); await image('error',data.cases[1].views[0].evaluation.error);
  must(await js('document.documentElement.scrollWidth <= innerWidth'), 'Mobile evaluation overflow');
  must(!errors.length && !external.length,'Unexpected page error or external request: '+JSON.stringify([errors,external]));
  must(eq([...fileRequests].sort(),[...expected].sort()),'Actual image/request roster differs: '+JSON.stringify([...expected].filter(k=>!fileRequests.has(k))));
  return {schema:'el-foundation-browser-readback-1',browser:version.product,protocol:version.protocolVersion,
    page_transport:'file://',network:'offline',external_requests:external.length,script_errors:errors,initial_files:initial,
    png_paths_read:[...fileRequests].filter(x=>x.endsWith('.png')).sort(),verified_conditions:tests,
    rendered_source_views_decoded:sourcesDecoded,independent_references_opened:openedReferences,screenshots,
    keyboard:['ArrowRight selected assessment','End selected sources and procedure','Home restored picture'],
    target_navigation:'Previous and next camera buttons matched native view order',
    gate:'Picture load requested just its prediction. Evaluation had no source URLs until an explicit open. Selecting a new condition closed it.',
    mobile};
}
try {
  const limit = new Promise((_,no)=>{watchdog=setTimeout(()=>no(Error('Browser test timed out')),140000);});
  process.stdout.write(JSON.stringify(await Promise.race([verify(),limit]))+'\n');
} finally {
  clearTimeout(watchdog); socket?.close(); chrome.kill('SIGTERM'); await closed;
  await rm(profile,{recursive:true,force:true,maxRetries:3,retryDelay:100});
}
