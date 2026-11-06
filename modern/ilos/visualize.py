"""Export a zero-dependency, self-contained HTML playback of the planned fleet."""
from __future__ import annotations

import json
from pathlib import Path
from .planner import Result

_TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ILOS | Fleet simulation</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,Arial;color:#edf5ff;background:#070e1a}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 15% 5%,#12294b,#070e1a 55%);min-height:100vh}main{max-width:1300px;margin:auto;padding:42px 22px}
.top{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}.logo{font-size:27px;letter-spacing:.13em;font-weight:850}.logo span{color:#54e8bc}.tag{border:1px solid #27425a;border-radius:100px;padding:7px 13px;color:#9db3c9;font-size:12px}
h1{font-size:clamp(30px,5vw,55px);margin:44px 0 9px;letter-spacing:-.04em}p{color:#9fb2c9;line-height:1.65}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:30px 0}.stat,.panel{background:rgba(12,24,42,.92);border:1px solid #20334c;border-radius:20px}.stat{padding:21px}.stat b{display:block;font-size:29px}.stat small{font-size:12px;color:#8ba8c3}
.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(230px,1fr);gap:16px}.panel{padding:22px}svg{width:100%;height:auto;display:block}.fleet{display:grid;gap:10px}.vehicle{padding:12px;border:1px solid #20334c;border-radius:11px;display:flex;gap:10px;align-items:center}.dot{width:11px;height:11px;border-radius:50%;flex-shrink:0}.vehicle span{color:#92a6bc;font-size:12px;display:block;margin-top:4px}button{white-space:nowrap;flex-shrink:0;cursor:pointer;background:#55edbb;color:#09131f;border:0;border-radius:10px;padding:12px 20px;font-weight:800}button:hover{filter:brightness(1.1)}input[type=range]{min-width:0;flex:1;width:100%;accent-color:#55edbb}.controls{display:flex;align-items:center;gap:16px;margin-top:16px}.status{color:#9db1cc;margin-left:auto;font-variant-numeric:tabular-nums}.hint{font-size:12px}footer{font-size:12px;color:#607f9a;margin:26px 2px}
@media(max-width:790px){.layout{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,minmax(0,1fr))}.stat{padding:13px}.stat b{font-size:22px}}
</style></head><body><main>
<div class="top"><div class="logo">I<span>L</span>OS <span style="font-size:10px;vertical-align:middle;font-weight:500;letter-spacing:.12em">INTELLIGENT LOGISTICS</span></div><div class="tag">● REPRODUCIBLE SIMULATION</div></div>
<h1 id="headline">One coordinated fleet.</h1><p>Time-aware fleet routing. Visualize each discrete movement, review congested areas, and inspect collision-free trajectories.</p>
<div class="stats"><div class="stat"><small>ACTIVE VEHICLES</small><b id="active"></b></div><div class="stat"><small>MAKESPAN</small><b id="makespan"></b></div><div class="stat"><small>SAFETY CHECK</small><b style="color:#55edbb">PASS</b></div></div>
<div class="layout"><section class="panel"><svg id="map" role="img" aria-label="ILOS warehouse grid and AGV positions" viewBox="0 0 800 550"></svg><div class="controls"><button id="play">▶ Play</button><button id="restart" aria-label="Restart playback" style="background:#20334c;color:#edf5ff;padding:12px">↺</button><input id="step" type="range" min="0" value="0" aria-label="Simulation step"><strong id="counter" style="min-width:75px;text-align:right;font-size:13px">0 / 0</strong></div><p class="hint">One step represents one unit of discrete time. Vehicles remain at their destinations after arrival.</p></section><aside class="panel"><h3 style="margin-top:2px">Fleet overview</h3><div id="fleet" class="fleet"></div><p class="hint">Colors identify individual AGVs. Orange cells represent static congestion penalties; dark blocks are obstacles.</p></aside></div>
<footer>ILOS | Demonstration only. This simulation does not control physical vehicles.</footer>
<script>
const data = __DATA__;
const colors=['#60a5fa','#fcad68','#70e3c4','#db9bff','#ffd66d','#fa869d','#9cddf1','#a8ea80'];
const S=data.scenario, paths=data.paths, ids=Object.keys(paths), max=data.makespan;
const svg=document.getElementById('map'), fleet=document.getElementById('fleet'), step=document.getElementById('step'), counter=document.getElementById('counter');
let tick=0, timer=null;
document.getElementById('headline').textContent=ids.length+' vehicles. One coordinated plan.';
document.querySelector('.tag').textContent=data.algorithm;
document.getElementById('active').textContent=ids.length;
document.getElementById('makespan').textContent=max+' steps';step.max=max;
const walls=new Set(S.obstacles.map(p=>p.join(',')));
const congested=new Map(S.congestion.map(e=>[e.cell.join(','), e.cost]));
const W=800,H=550,cs=Math.min((W-45)/S.width,(H-45)/S.height),left=(W-cs*S.width)/2,gridTop=(H-cs*S.height)/2;
function el(tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attrs||{}))n.setAttribute(key,value);if(text!==undefined)n.textContent=text;return n}
function center(p){return [left+(p[0]+.5)*cs,gridTop+(p[1]+.5)*cs]}
function draw(){svg.replaceChildren();svg.append(el('rect',{x:0,y:0,width:W,height:H,rx:15,fill:'#0a1729'}));
for(let y=0;y<S.height;y++)for(let x=0;x<S.width;x++){const k=x+','+y,wall=walls.has(k),traffic=congested.get(k)||0;svg.append(el('rect',{x:left+x*cs+1,y:gridTop+y*cs+1,width:cs-2,height:cs-2,rx:Math.max(2,cs*.11),fill:wall?'#34435b':traffic?'#48362b':'#112239',stroke:'#19304a','stroke-width':'.8'}));if(traffic&&!wall){const [cx,cy]=center([x,y]);svg.append(el('circle',{cx,cy,r:cs*.075,fill:'#ed995e',opacity:.75}))}}
for(const [i,id] of ids.entries()){const path=paths[id],p=path[Math.min(tick,path.length-1)],[x,y]=center(p),c=colors[i%colors.length];const tx=path.map(v=>center(v).join(',')).join(' ');svg.append(el('polyline',{points:tx,fill:'none',stroke:c,'stroke-width':Math.max(1.2,cs*.045),opacity:.35,'stroke-linecap':'round','stroke-linejoin':'round'}));svg.append(el('circle',{cx:x,cy:y,r:cs*.28,fill:c,stroke:'#edf9ff','stroke-width':1.8}));svg.append(el('text',{x,y:y+4,'font-size':Math.max(8,cs*.22),'text-anchor':'middle',fill:'#09111b','font-weight':'800'},String(i+1)))}
step.value=tick;counter.textContent=tick+' / '+max;fleet.replaceChildren();ids.forEach((id,i)=>{const p=paths[id][Math.min(tick,paths[id].length-1)],item=document.createElement('div');item.className='vehicle';const dot=document.createElement('div');dot.className='dot';dot.style.background=colors[i%colors.length];const t=document.createElement('div');const b=document.createElement('strong');b.textContent=id;const s=document.createElement('span');s.textContent='('+p.join(', ')+') · '+(tick>=paths[id].length-1?'Arrived':'En route');t.append(b,s);item.append(dot,t);fleet.append(item)})}
function stop(){if(timer!==null)clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ Play'}
document.getElementById('play').onclick=()=>{if(timer!==null){stop();return}if(tick>=max)tick=0;document.getElementById('play').textContent='Ⅱ Pause';timer=setInterval(()=>{tick=Math.min(tick+1,max);draw();if(tick>=max)stop()},420)};
document.getElementById('restart').onclick=()=>{stop();tick=0;draw()};
step.oninput=()=>{tick=Number(step.value);stop();draw()};draw();
</script></main></body></html>'''


def save_html(result: Result, filename: str | Path) -> None:
    result.validate()
    scenario = result.scenario
    obj = {"scenario": {"name": scenario.name, "width": scenario.width,
                         "height": scenario.height,
                         "obstacles": [list(p) for p in sorted(scenario.obstacles)],
                         "congestion": [{"cell": list(k), "cost": v} for k, v in scenario.congestion.items()]},
           "paths": {k: [list(p) for p in value] for k, value in result.paths.items()},
           "makespan": result.makespan, "algorithm": result.algorithm}
    # Avoid breaking out of the script element through untrusted names/IDs.
    safe_json = json.dumps(obj, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = _TEMPLATE.replace('__DATA__', safe_json)
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding='utf-8')
