"""Open the finished local report in an isolated offline Chromium process and exercise its view picker."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from bundle_original import check, put, sha

CONTROL = r"""
const {spawn}=require('node:child_process'),fs=require('node:fs'),{pathToFileURL}=require('node:url');
const [chrome,page,profile,png]=process.argv.slice(1);
const proc=spawn(chrome,['--headless=new','--disable-gpu','--disable-background-networking','--disable-component-update','--disable-sync','--no-first-run','--disable-default-apps','--no-default-browser-check','--host-resolver-rules=MAP * ~NOTFOUND','--user-data-dir='+profile,'--remote-debugging-pipe'],{stdio:['ignore','ignore','ignore','pipe','pipe']});
let seq=0, buf=Buffer.alloc(0);const wait=new Map(),requests=[];
function call(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++seq;wait.set(id,[resolve,reject]);proc.stdio[3].write(JSON.stringify({id,method,params,sessionId})+'\0');});}
proc.stdio[4].on('data',chunk=>{buf=Buffer.concat([buf,chunk]);for(let pos;(pos=buf.indexOf(0))>=0;){const m=JSON.parse(buf.subarray(0,pos).toString());buf=buf.subarray(pos+1);if(m.id && wait.has(m.id)){const [resolve,reject]=wait.get(m.id);wait.delete(m.id);if(m.error)reject(Error(m.error.message));else resolve(m.result);}else if(m.method==='Network.requestWillBeSent'){requests.push(m.params.request.url.split(':')[0]);}}});
(async()=>{const {targetId}=await call('Target.createTarget',{url:'about:blank'});const {sessionId:s}=await call('Target.attachToTarget',{targetId,flatten:true});
await call('Network.enable',{},s);await call('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:0,uploadThroughput:0},s);
await call('Page.enable',{},s);await call('Emulation.setDeviceMetricsOverride',{width:1240,height:960,deviceScaleFactor:1,mobile:false},s);
await call('Page.navigate',{url:pathToFileURL(page).href},s);
async function value(expression){const q=await call('Runtime.evaluate',{expression,returnByValue:true},s);if(q.exceptionDetails)throw Error('runtime exception');return q.result.value;}
for(let i=0;i<150;i++){if(await value("document.readyState==='complete' && !!document.querySelector('#pick') && [...document.images].every(x=>x.complete && x.naturalWidth===256)"))break;await new Promise(r=>setTimeout(r,100));}
const inspect=await value("(()=>{const q=document.querySelector('#pick'),sections=[...document.querySelectorAll('.observation')],out=[]; for(const opt of q.options){q.value=opt.value;q.dispatchEvent(new Event('change'));out.push({value:opt.text,visible:sections.filter(s=>!s.hidden).map(s=>s.querySelector('h3').textContent)});}q.value='0';q.dispatchEvent(new Event('change'));return {views:out,images:[...document.images].map(i=>({width:i.naturalWidth,height:i.naturalHeight})),receipt:document.querySelector('a[download]').getAttribute('href').startsWith('data:application/json;base64,')};})()");
const capture=await call('Page.captureScreenshot',{format:'png'},s);fs.writeFileSync(png,Buffer.from(capture.data,'base64'));
const version=await call('Browser.getVersion');console.log(JSON.stringify({product:version.product,...inspect,request_schemes:[...new Set(requests)].sort(),offline:true}));await call('Browser.close');})().catch(err=>{console.error(String(err));proc.kill();process.exit(1);});
"""


def inspect(a):
    with tempfile.TemporaryDirectory(dir=a.out.parent) as temp:
        out = a.out / "offline-browser.png"
        p = subprocess.run(
            [shutil.which("node"), "-e", CONTROL, str(a.chrome), str(a.html.resolve()), temp, str(out)],
            capture_output=True,
            text=True,
            timeout=55,
        )
        check(p.returncode == 0, p.stderr[-1300:])
        r = json.loads(p.stdout)
        check(
            r["request_schemes"] == ["data", "file"]
            and r["receipt"]
            and all(i == {"width": 256, "height": 256} for i in r["images"])
            and len(r["images"]) == 28,
            "unexpected document resources or broken images",
        )
        check(
            all(item["visible"] == [item["value"]] for item in r["views"][:-1])
            and len(r["views"][-1]["visible"]) == 6,
            "view picker did not reveal its named target",
        )
        put(
            a.out / "html-readback.json",
            {
                "schema": "el-bundle-offline-browser-1",
                "html_sha256": sha(a.html.read_bytes()),
                "screenshot_sha256": sha(out.read_bytes()),
                **r,
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("chrome", "html", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    inspect(p.parse_args())
