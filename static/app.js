// SIZZLE front-end — free-form audio-reactive timeline.
// Images own time: each block's WIDTH = render duration. Drag to place, drag the
// right edge to resize, click to edit its prompt. Beats are soft snap guides.
// A "continue" block grows the video from the previous block's last frame.

const state = {
  audioId: null,
  audioUrl: null,
  duration: 0,
  peaks: [],
  beats: [],
  downbeats: [],
  variant: null,
  maxClip: 6.0,
  fps: 24,
  format: null,          // {width,height} chosen output resolution
  formats: [],           // presets from server (each with megapixels)
  minDim: 256, maxDim: 1920,
  autoprompt: false,
  zoom: 1,
  baseFit: 100,          // px/sec at zoom 1 (set on layout)
  blocks: [],            // {id,kind,imageId,url,prompt,promptStatus,start,dur}
  tray: {},              // imageId -> {url, prompt, status}
  selectedId: null,
};

const MIN_BLOCK = 1.0;
const DEFAULT_BLOCK = 4.0;
const SNAP_PX = 8;
const MAGNET_PX = 26;    // stronger, wider pull to clip edges / timeline start
const GAP_MIN = 0.05;    // gaps larger than this (s) are flagged before generate
let _idc = 0;
const uid = () => "b" + (++_idc);

const $ = (s) => document.querySelector(s);
const api = async (url, opts) => (await fetch(url, opts)).json();
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
function fmt(t){ if(!isFinite(t))t=0; const m=Math.floor(t/60), s=(t%60); return `${m}:${s.toFixed(1).padStart(4,"0")}`; }

// "NVIDIA RTX PRO 6000 Blackwell Server Edition" -> "RTX PRO 6000 Blackwell".
// Drops the vendor prefix and the SKU tail so the badge stays a badge; the full
// name is still in the tooltip. No renaming — just trimming.
function shortGpu(name){
  return String(name)
    .replace(/^NVIDIA\s+/i, "")
    .replace(/\s+(Server|Workstation)\s+Edition$/i, "")
    .slice(0, 26);
}

const pxPerSec = () => state.baseFit * state.zoom;
const xOf = (t) => t * pxPerSec();
const tOf = (x) => x / pxPerSec();

// ---- status ----------------------------------------------------------------
async function loadStatus(){
  const st = await api("/api/status");
  const r = $("#badge-render");
  const ready = st.gpu_ready;
  // Distinguish the three states that matter on a self-hosted box: rendering
  // locally and ready, GPU fine but the ~100GB of weights not downloaded yet,
  // and no usable CUDA device at all. "offline" told you none of that.
  const gpu = st.gpu || {};
  const missing = st.weights_missing || [];
  if(ready && st.backend==="local"){
    r.textContent = `render: local · ${gpu.name ? shortGpu(gpu.name) : "cuda"}`;
    r.title = gpu.vram_gb ? `${gpu.name} · ${gpu.vram_gb} GB VRAM` : "local GPU inference";
  } else if(st.backend==="local" && missing.length){
    r.textContent = `render: local (${missing.length} weight${missing.length>1?"s":""} missing)`;
    r.title = "missing: " + missing.join(", ") + "\nrun: python scripts/download_models.py";
  } else if(st.backend==="local"){
    r.textContent = "render: local (no CUDA)";
    r.title = "no usable CUDA device found";
  } else {
    r.textContent = `render: ${st.backend}${ready?"":" (offline)"}`;
    r.title = "mock backend: ffmpeg synthesizes clips, no model runs";
  }
  r.className = "badge " + (ready ? "ok" : "bad");
  const p = $("#badge-prompt");
  state.autoprompt = !!st.autoprompt;
  // Reflect live reachability, not just key-presence: a present-but-unreachable
  // key (TLS-MITM / offline) shows a distinct amber "unreachable" instead of a
  // false green "on" (§6.2 — would have caught the Avast incident on sight).
  if(st.autoprompt && st.autoprompt_reachable===false){
    p.textContent = "auto-prompt: unreachable";
    p.className = "badge bad";
    p.title = st.autoprompt_error || "cannot reach the vision API";
  } else {
    p.textContent = "auto-prompt: " + (st.autoprompt ? "on" : "off");
    p.className = "badge " + (st.autoprompt ? "ok" : "warn");
    p.title = "";
  }
  const f = $("#badge-ffmpeg");
  f.textContent = "ffmpeg: " + (st.ffmpeg ? "ok" : "missing");
  f.className = "badge " + (st.ffmpeg ? "ok" : "bad");

  state.maxClip = st.max_clip_seconds || 6.0;
  state.fps = st.fps || 24;

  const sel = $("#variant"); sel.innerHTML = "";
  for (const [k,label] of Object.entries(st.variants||{})){
    const o=document.createElement("option"); o.value=k; o.textContent=label;
    if (k===st.default_variant) o.selected=true; sel.appendChild(o);
  }
  state.variant = st.default_variant;
  sel.onchange = () => state.variant = sel.value;

  setupFormats(st);
}

// ---- format / resolution (Phase E) -----------------------------------------
function setupFormats(st){
  state.formats = st.formats || [];
  state.fps = st.fps || state.fps;
  state.minDim = st.min_dim || 256;
  state.maxDim = st.max_dim || 1920;
  // The two-stage pipeline denoises at half size then upscales 2x, so both
  // dimensions must land on a 64px grid. The server snaps too; doing it here as
  // well means the box shows you the size you'll actually get as you type.
  state.dimAlign = st.dim_align || 64;
  const dw = st.default_width, dh = st.default_height;
  const fsel = $("#format"); if(!fsel) return;
  fsel.innerHTML = "";
  let defIdx = 0;
  state.formats.forEach((f,i)=>{
    const o=document.createElement("option");
    o.value=String(i); o.textContent=`${f.label} · ${f.width}×${f.height}`;
    if(f.width===dw && f.height===dh){ o.selected=true; defIdx=i; }
    fsel.appendChild(o);
  });
  const cust=document.createElement("option"); cust.value="custom"; cust.textContent="Custom size…";
  fsel.appendChild(cust);
  // default selection → state.format
  const d = state.formats[defIdx] || {width:dw||768,height:dh||1280};
  state.format = {width:d.width, height:d.height};
  $("#fmt-w").value=d.width; $("#fmt-h").value=d.height;
  fsel.onchange = onFormatChange;
  $("#fmt-w").oninput = $("#fmt-h").oninput = onCustomDims;
  updateFormatCost();
}
function onFormatChange(){
  const v=$("#format").value;
  const customBox=$("#fmt-custom");
  if(v==="custom"){
    customBox.classList.remove("hidden");
    onCustomDims();
  } else {
    customBox.classList.add("hidden");
    const f=state.formats[parseInt(v)];
    if(f){ state.format={width:f.width,height:f.height}; $("#fmt-w").value=f.width; $("#fmt-h").value=f.height; }
  }
  updateFormatCost(); refreshAspectBadges();
}
function onCustomDims(){
  let w=parseInt($("#fmt-w").value), h=parseInt($("#fmt-h").value);
  if(!w||!h) return;
  const a = state.dimAlign||64;
  const snap = (v)=>clamp(Math.round(v/a)*a, state.minDim, state.maxDim);
  w=snap(w); h=snap(h);
  state.format={width:w,height:h};
  updateFormatCost(); refreshAspectBadges();
}
// Local renders cost no money — they cost TIME, and time scales with pixels.
// So the readout that used to show $/sec now shows the frame size driving it.
function updateFormatCost(){
  const c=$("#fmt-cost"); if(!c||!state.format) return;
  const mp = (state.format.width*state.format.height/1e6).toFixed(2);
  c.textContent = `${state.format.width}×${state.format.height} · ${mp} MP/frame`;
}

// ---- audio upload ----------------------------------------------------------
const audioDrop = $("#audio-drop"), audioInput = $("#audio-input");
audioDrop.addEventListener("dragover", e=>{e.preventDefault();audioDrop.classList.add("hover");});
audioDrop.addEventListener("dragleave", ()=>audioDrop.classList.remove("hover"));
audioDrop.addEventListener("drop", e=>{e.preventDefault();audioDrop.classList.remove("hover");
  if (e.dataTransfer.files[0]) uploadAudio(e.dataTransfer.files[0]);});
audioInput.addEventListener("change", e=>{ if(e.target.files[0]) uploadAudio(e.target.files[0]); });

async function uploadAudio(file){
  $(".drop-label").textContent = "analyzing " + file.name + " ...";
  const fd = new FormData(); fd.append("file", file); fd.append("mode","beat");
  const res = await fetch("/api/audio", {method:"POST", body:fd}).then(r=>r.json());
  state.audioId = res.audio_id;
  state.duration = res.duration;
  state.peaks = res.peaks || [];
  state.beats = res.beats || [];
  state.downbeats = res.downbeats || [];
  state.blocks = [];
  state.audioUrl = URL.createObjectURL(file);
  $("#preview-audio").src = state.audioUrl;

  $(".drop-label").textContent = file.name + "  //  reload page to change track";
  $("#audio-meta").textContent =
    `${fmt(res.duration)} // ${res.tempo} bpm // ${state.beats.length} beats detected`;
  $("#seg-controls").classList.remove("hidden");

  $("#step-timeline").classList.remove("locked");
  fitZoom();
  // start zoomed to ~2x the fitted size so the first pass is a detailed,
  // leveled edit. The "fit" button drops back to 1x (fitted) whenever wanted.
  zoomTo(2);
  renderTimeline();
  updateGating();
  // reset the transport for the new track
  previewAudio.currentTime=0; playhead.style.left="0px"; playhead.style.display="block";
  updateTimeReadout(0); setPlayIcon();
}

// ---- timeline layout -------------------------------------------------------
const scrollEl = $("#timeline-scroll");
const innerEl = $("#timeline-inner");
const trackEl = $("#tl-track");
const rulerEl = $("#tl-ruler");
const ticksEl = $("#tl-ticks");
const waveCanvas = $("#waveform");

function fitZoom(){
  const w = scrollEl.clientWidth || 1100;
  state.baseFit = state.duration ? (w - 2) / state.duration : 100;
  state.zoom = 1;
  $("#zoom").value = "1";
}

// Set the zoom multiplier, clamped so we never zoom out past the fitted size
// (zoom 1 == the whole track fits the viewport) or past the slider's max.
function zoomTo(z){
  state.zoom = Math.min(10, Math.max(1, z));
  $("#zoom").value = String(state.zoom);
}

function niceStep(targetSec){
  const steps=[0.5,1,2,5,10,15,30,60];
  for(const s of steps){ if(s>=targetSec) return s; }
  return 120;
}

function layout(){
  if(!state.duration) return;
  const innerW = Math.max(scrollEl.clientWidth, xOf(state.duration));
  innerEl.style.width = innerW + "px";

  // waveform
  waveCanvas.width = innerW; const h = waveCanvas.height;
  const ctx = waveCanvas.getContext("2d");
  ctx.clearRect(0,0,innerW,h);
  const n = state.peaks.length;
  if(n){
    ctx.strokeStyle="#3bffd0"; ctx.globalAlpha=.75; ctx.beginPath();
    for(let i=0;i<n;i++){ const x=(i/n)*innerW, amp=state.peaks[i]*(h*0.46);
      ctx.moveTo(x,h/2-amp); ctx.lineTo(x,h/2+amp); }
    ctx.stroke(); ctx.globalAlpha=1;
  }

  // ruler labels
  rulerEl.innerHTML="";
  const step = niceStep(80 / pxPerSec());
  for(let t=0;t<=state.duration+1e-6;t+=step){
    const l=document.createElement("div"); l.className="rlabel";
    l.style.left=xOf(t)+"px"; l.textContent=fmt(t); rulerEl.appendChild(l);
  }

  // beat ticks (thin=beat, bright=downbeat). Skip minor beats when too dense.
  ticksEl.innerHTML="";
  const beatPx = state.beats.length>1 ? xOf(state.beats[1]-state.beats[0]) : 999;
  const dl = new Set(state.downbeats.map(t=>t.toFixed(3)));
  for(const t of state.beats){
    const down = dl.has(t.toFixed(3));
    if(!down && beatPx<5) continue;      // too dense: keep only downbeats
    const d=document.createElement("div"); d.className="tick"+(down?" down":"");
    d.style.left=xOf(t)+"px"; ticksEl.appendChild(d);
  }

  // position blocks
  for(const b of state.blocks){
    const el = b._el; if(!el) continue;
    el.style.left = xOf(b.start)+"px";
    el.style.width = Math.max(6, xOf(b.dur))+"px";
    updateBlockLabels(b);
  }
}

function renderTimeline(){
  // (re)build block DOM from state
  trackEl.querySelectorAll(".block").forEach(e=>e.remove());
  for(const b of state.blocks) trackEl.appendChild(makeBlockEl(b));
  layout();
  // re-apply the "now playing" highlight (fresh DOM lost the class)
  if(typeof previewAudio!=="undefined") updatePlayingBlock(previewAudio.currentTime);
}

// mark whichever block the playhead is currently over
function updatePlayingBlock(t){
  const playing=typeof previewAudio!=="undefined" && !previewAudio.paused && !previewAudio.ended;
  for(const b of state.blocks){
    if(!b._el) continue;
    const over = playing && t>=b.start && t<b.start+b.dur;
    b._el.classList.toggle("playing", over);
  }
}

// ---- snapping --------------------------------------------------------------
function snapPoints(exceptId){
  const pts=[0, state.duration, ...state.beats, ...state.downbeats];
  for(const b of state.blocks){ if(b.id===exceptId) continue; pts.push(b.start, b.start+b.dur); }
  return pts;
}
function snap(t, exceptId){
  const tol = SNAP_PX / pxPerSec();
  let best=t, bd=tol;
  for(const p of snapPoints(exceptId)){ const d=Math.abs(p-t); if(d<bd){bd=d;best=p;} }
  return best;
}

// Magnetism: structural edges (timeline start + neighbouring clip edges) pull
// harder and over a wider range than beats, so a clip dropped a few ticks off a
// neighbour snaps flush instead of leaving an accidental micro-gap. Falls back
// to the softer beat snap when nothing structural is within range.
function magnetPoints(exceptId){
  const pts=[0];
  for(const b of state.blocks){ if(b.id===exceptId) continue; pts.push(b.start, b.start+b.dur); }
  return pts;
}
function magnet(t, exceptId){
  const tol = MAGNET_PX / pxPerSec();
  let best=null, bd=tol;
  for(const p of magnetPoints(exceptId)){ const d=Math.abs(p-t); if(d<bd){bd=d;best=p;} }
  return best!=null ? best : snap(t, exceptId);
}

// neighbors in time order (excluding a block id) for non-overlap clamping
function neighborBounds(b){
  let leftEnd=0, rightStart=state.duration;
  for(const o of state.blocks){
    if(o.id===b.id) continue;
    if(o.start+o.dur<=b.start+1e-6) leftEnd=Math.max(leftEnd,o.start+o.dur);
    if(o.start>=b.start+b.dur-1e-6) rightStart=Math.min(rightStart,o.start);
  }
  return {leftEnd,rightStart};
}

// ---- block element ---------------------------------------------------------
function makeBlockEl(b){
  const el=document.createElement("div");
  el.className="block kind-"+b.kind;
  el.dataset.id=b.id;
  // the image rides on a dedicated .b-img layer (below .b-veil) so the render
  // shimmer on the veil always stays on top of it — see style.css.
  if(b.kind==="image" && b.url) el.style.setProperty("--bimg",`url("${b.url}")`);
  el.innerHTML = b.kind==="image"
    ? `<div class="b-img"></div>
       <div class="b-veil"></div>
       <div class="b-top"><span class="b-time"></span><span class="b-dur"></span></div>
       <div class="b-bottom"></div>
       <div class="b-resize" title="drag to resize"></div>`
    : `<div class="b-top"><span class="b-time"></span><span class="b-dur"></span></div>
       <div class="b-hint">⟳</div><div class="b-sub">CONTINUE</div>
       <div class="b-resize" title="drag to resize"></div>`;
  b._el=el;
  el.addEventListener("pointerdown", (e)=>onBlockPointerDown(e,b));
  el.addEventListener("click",(e)=>{ if(el._moved){el._moved=false;return;} openModal(b); });
  applyBlockRisk(b);
  return el;
}

function updateBlockLabels(b){
  const el=b._el; if(!el) return;
  el.querySelector(".b-time").textContent = fmt(b.start);
  const dtxt = b.dur.toFixed(1)+"s";
  el.querySelector(".b-dur").textContent = dtxt;
  el.classList.toggle("overcap", b.dur>state.maxClip+1e-3);
  if(b.kind==="image"){
    const bot=el.querySelector(".b-bottom");
    bot.textContent = b.promptStatus==="thinking" ? "auto-prompt…"
      : (b.prompt ? b.prompt : "click to add a prompt");
  }
}

// ---- block drag + resize ---------------------------------------------------
function onBlockPointerDown(e,b){
  e.preventDefault();
  const resizing = e.target.classList.contains("b-resize");
  const el=b._el; el._moved=false;
  const startX=e.clientX, origStart=b.start, origDur=b.dur;
  el.setPointerCapture(e.pointerId);
  el.classList.add("dragging");
  selectBlock(b);

  function move(ev){
    const dx=ev.clientX-startX;
    if(Math.abs(dx)>3){ el._moved=true; clearGapHighlights(); }
    const {leftEnd,rightStart}=neighborBounds(b);
    if(resizing){
      let nd = origDur + tOf(dx);
      // snap right edge (magnetise flush to the next clip / timeline end)
      let rightEdge = magnet(origStart+nd, b.id); nd = rightEdge-origStart;
      const maxRoom = rightStart - origStart;
      nd = clamp(nd, MIN_BLOCK, maxRoom);
      b.dur = nd;
    } else {
      let ns = origStart + tOf(dx);
      // magnetise whichever edge is closer (flush to a neighbour or the start)
      const sLeft=magnet(ns,b.id), sRight=magnet(ns+b.dur,b.id)-b.dur;
      ns = Math.abs(sLeft-ns) <= Math.abs(sRight-ns) ? sLeft : sRight;
      ns = clamp(ns, leftEnd, rightStart-b.dur);
      b.start = ns;
    }
    el.style.left=xOf(b.start)+"px";
    el.style.width=Math.max(6,xOf(b.dur))+"px";
    updateBlockLabels(b);
  }
  function up(ev){
    el.releasePointerCapture(e.pointerId);
    el.classList.remove("dragging");
    window.removeEventListener("pointermove",move);
    window.removeEventListener("pointerup",up);
    if(resizing && b.kind==="image" && b.dur>state.maxClip+1e-3) autoChainContinues(b);
    layout(); updateGating();
  }
  window.addEventListener("pointermove",move);
  window.addEventListener("pointerup",up);
}

// hybrid: stretching an image past the model cap -> keep it at cap and append
// continue blocks that carry the video onward, filling the requested length.
function autoChainContinues(b){
  const requested=b.dur;
  const {rightStart}=neighborBounds(b);
  b.dur=state.maxClip;
  let remaining=requested-state.maxClip;
  let cursor=b.start+b.dur;
  const order=state.blocks;
  let insertAt=order.indexOf(b)+1;
  while(remaining>0.05 && cursor<rightStart-0.05){
    const d=Math.min(state.maxClip, remaining, rightStart-cursor);
    if(d<MIN_BLOCK) break;
    const c={id:uid(),kind:"continue",prompt:"",promptStatus:"none",start:cursor,dur:d};
    order.splice(insertAt,0,c); trackEl.appendChild(makeBlockEl(c));
    insertAt++; cursor+=d; remaining-=d;
  }
  flashHelp(`stretched past ${state.maxClip}s → auto-added continue${insertAt>order.indexOf(b)+2?"s":""} to fill it`);
  renderTimeline();
}

function selectBlock(b){
  state.selectedId=b.id;
  trackEl.querySelectorAll(".block").forEach(e=>e.classList.toggle("selected", e.dataset.id===b.id));
}

// ---- tray: add images + auto-prompt ----------------------------------------
const imageInput=$("#image-input");
$("#tray-add").addEventListener("click",()=>imageInput.click());
imageInput.addEventListener("change", async e=>{ for(const f of e.target.files){ await addTrayImage(f); } e.target.value=""; });

async function addTrayImage(file){
  const fd=new FormData(); fd.append("file",file);
  const res=await fetch("/api/image",{method:"POST",body:fd}).then(r=>r.json());
  state.tray[res.image_id]={url:res.url, prompt:"", status:state.autoprompt?"thinking":"off"};

  const wrap=document.createElement("div"); wrap.className="tray-img"; wrap.dataset.imageId=res.image_id;
  wrap.innerHTML=`<img src="${res.url}" draggable="true">
    <span class="pstatus ${state.autoprompt?"thinking":"off"}">${state.autoprompt?"…":"no key"}</span>`;
  const img=wrap.querySelector("img");
  img.addEventListener("load",()=>{
    const it=state.tray[res.image_id];
    if(it){ it.ar=img.naturalWidth/img.naturalHeight; updateAspectBadge(res.image_id); }
  });
  img.addEventListener("dragstart",ev=>{
    wrap.classList.add("dragging"); draggingKind="image";
    ev.dataTransfer.setData("kind","image");
    ev.dataTransfer.setData("imageId",res.image_id);
    ev.dataTransfer.setData("url",res.url);
  });
  img.addEventListener("dragend",()=>{wrap.classList.remove("dragging"); draggingKind=null;});
  $("#tray-scroll").appendChild(wrap);

  // fire auto-prompt immediately so a prompt is ready by the time it's dropped
  if(state.autoprompt) fetchAutoPrompt(res.image_id);
}

async function fetchAutoPrompt(imageId){
  const item=state.tray[imageId]; if(!item) return;
  item.status="thinking"; setTrayStatus(imageId,"thinking","…");
  try{
    const res=await api("/api/autoprompt",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image_id:imageId})});
    item.prompt=res.prompt||""; item.source=res.source||"vision";
    item.risk=res.risk||"unknown"; setTrayRisk(imageId,item.risk);
    // A "heuristic" source means the vision call didn't run (no key / no net):
    // every image would get the SAME generic prompt, so flag it rather than
    // pretending it's a real per-image suggestion.
    if(res.source==="heuristic"){
      item.status="fallback"; setTrayStatus(imageId,"warn","generic");
      flashHelp("auto-prompt offline ("+(res.error||"no vision")+") — using a generic prompt; edit it per clip");
    } else {
      item.status="ready"; setTrayStatus(imageId,"ready","ready");
    }
    // push into any placed blocks that inherited empty/thinking from this image
    for(const b of state.blocks){
      if(b.kind==="image" && b.imageId===imageId && b.promptStatus!=="edited" && !b._userPrompt){
        b.prompt=item.prompt; b.promptStatus=item.status==="ready"?"ready":"fallback"; updateBlockLabels(b);
      }
    }
  }catch(e){ item.status="off"; setTrayStatus(imageId,"off","err"); }
}
function setTrayStatus(imageId,cls,txt){
  const w=$(`.tray-img[data-image-id="${imageId}"] .pstatus`);
  if(w){ w.className="pstatus "+cls; w.textContent=txt; }
}

// Pre-flight NSFW risk badge (Phase D Tier 3). Only maybe/likely show
// a badge — 'none'/'unknown' stay clean (no false alarms, graceful no-key path).
// Aspect-mismatch badge (Phase E6): warn (passively) when a tray image's aspect
// ratio differs from the chosen output format by more than ~15%. The whole image
// is kept and letterboxed (SIZZLE_IMAGE_FIT=contain), so the cost of a mismatch
// is black bars rather than lost picture — but it is still worth flagging, since
// matching the format uses more of the frame. Re-evaluated when format changes.
function formatAR(){ return state.format ? state.format.width/state.format.height : null; }
function updateAspectBadge(imageId){
  const it=state.tray[imageId], wrap=$(`.tray-img[data-image-id="${imageId}"]`);
  if(!it||!wrap||!it.ar) return;
  const far=formatAR(); if(!far) return;
  const delta=Math.abs(it.ar-far)/far;
  let b=wrap.querySelector(".ar-badge");
  if(delta>0.15){
    if(!b){ b=document.createElement("span"); b.className="ar-badge"; wrap.appendChild(b); }
    b.textContent="⤢";
    b.title=`this image is ${it.ar>far?"wider":"taller"} than the chosen `
      +`${state.format.width}×${state.format.height} format — the whole image is kept, `
      +`with black bars filling the rest. Pick a matching format to use the full frame.`;
  } else if(b){ b.remove(); }
}
function refreshAspectBadges(){ for(const id in state.tray) updateAspectBadge(id); }

// Renders run on your own GPU now, so nothing is refused at render time. What
// this flags is what happens AFTER: NSFW-leaning footage tends to get age-gated
// or pulled once the finished video is posted.
function riskTip(risk){
  return risk==="likely"
    ? "Auto-prompt thinks this shot is NSFW-leaning enough to get age-gated or taken down if you post the video. It will still render fine."
    : "Auto-prompt thinks this shot is borderline NSFW — it may get age-gated on YouTube/TikTok/Instagram. It will still render fine.";
}
function setTrayRisk(imageId, risk){
  const wrap=$(`.tray-img[data-image-id="${imageId}"]`);
  if(wrap){
    let b=wrap.querySelector(".risk");
    if(risk==="maybe" || risk==="likely"){
      if(!b){ b=document.createElement("span"); b.className="risk"; wrap.appendChild(b); }
      b.className="risk "+risk; b.textContent="⚠"; b.title=riskTip(risk);
    } else if(b){ b.remove(); }
  }
  // mirror onto any placed blocks that came from this image
  for(const bl of state.blocks){
    if(bl.kind==="image" && bl.imageId===imageId){ bl._risk=risk; applyBlockRisk(bl); }
  }
  updateRiskNote();
}
// The NSFW heads-up copy only shows once at least one image is actually flagged.
function updateRiskNote(){
  const note=$("#tray-note");
  if(note){ note.hidden = !$(".tray-scroll .risk"); }
}
function applyBlockRisk(b){
  const el=b._el; if(!el || b.kind!=="image") return;
  const risky = b._risk==="maybe" || b._risk==="likely";
  el.classList.toggle("risky", risky);
  const top=el.querySelector(".b-top");
  let r=top && top.querySelector(".b-risk");
  if(risky){
    if(top && !r){ r=document.createElement("span"); r.className="b-risk"; top.insertBefore(r, top.firstChild); }
    if(r){ r.textContent="⚠"; r.title=riskTip(b._risk); }
  } else if(r){ r.remove(); }
}

// ---- continue token drag ---------------------------------------------------
const contTok=$("#tray-continue");
contTok.addEventListener("dragstart",ev=>{ contTok.classList.add("dragging"); draggingKind="continue"; ev.dataTransfer.setData("kind","continue"); });
contTok.addEventListener("dragend",()=>{ contTok.classList.remove("dragging"); draggingKind=null; });

// ---- drop onto the track ---------------------------------------------------
let ghost=null, draggingKind=null;
function dropTime(ev){
  const rect=innerEl.getBoundingClientRect();
  return clamp(tOf(ev.clientX-rect.left), 0, state.duration);
}
// Where a dropped IMAGE should sit and how long it should run. An image owns as
// much time as the single-clip model cap allows (state.maxClip ≈ 19s), or less
// when the audio / neighbouring clips leave less room:
//   • open space to the right  -> grow from the drop point up to maxClip or the
//     end of audio, leaving the remaining audio to the right free.
//   • gap between two blocks    -> butt flush against the LEFT neighbour and
//     fill up to maxClip; a gap wider than the cap keeps its right side open.
function imagePlacement(t){
  let rightStart=state.duration, leftEnd=0;
  const inside=state.blocks.find(o=>t>o.start && t<o.start+o.dur);
  if(inside) t=inside.start+inside.dur;          // dropped on a clip -> its right edge
  for(const o of state.blocks){
    if(o.start>=t-1e-6) rightStart=Math.min(rightStart,o.start);
    if(o.start+o.dur<=t+1e-6) leftEnd=Math.max(leftEnd,o.start+o.dur);
  }
  const hasRight=rightStart<state.duration-1e-6;  // a real gap between two clips
  const start=hasRight ? leftEnd : t;             // gap -> snap to left neighbour
  const room=rightStart-start;
  return {start, dur:Math.min(state.maxClip, room), room};
}
scrollEl.addEventListener("dragover", ev=>{
  ev.preventDefault(); trackEl.classList.add("drop-hover");
  const t=magnet(dropTime(ev));
  if(!ghost){ ghost=document.createElement("div"); ghost.className="drop-ghost"; trackEl.appendChild(ghost); }
  let left, width;
  if(draggingKind==="continue"){
    left=xOf(t); width=xOf(Math.min(DEFAULT_BLOCK, Math.max(0,state.duration-t)));
  } else {
    const p=imagePlacement(t); left=xOf(p.start); width=xOf(Math.max(0,p.dur));
  }
  ghost.style.left=left+"px"; ghost.style.width=width+"px";
});
scrollEl.addEventListener("dragleave", ev=>{ if(ev.target===scrollEl){trackEl.classList.remove("drop-hover"); if(ghost){ghost.remove();ghost=null;}} });
scrollEl.addEventListener("drop", ev=>{
  ev.preventDefault(); trackEl.classList.remove("drop-hover");
  if(ghost){ghost.remove();ghost=null;}
  const kind=ev.dataTransfer.getData("kind");
  let t=magnet(dropTime(ev));
  placeBlock(kind, t, ev);
});

function placeBlock(kind, t, ev){
  if(kind==="image"){
    // size it to the single-clip cap (or whatever room is left), butting against
    // the left neighbour when it lands in a gap between clips.
    const imageId=ev.dataTransfer.getData("imageId");
    if(!imageId){ return; }
    const place=imagePlacement(t);
    if(place.room<MIN_BLOCK){ flashHelp("no room there — try a gap on the track"); return; }
    const item=state.tray[imageId]||{};
    // pull the url from the tray by id, not the drag payload: the DnD spec
    // remaps the "url" key to text/uri-list and a root-relative path like
    // /api/image/xyz doesn't survive the round-trip, so getData("url") comes
    // back empty and the block would render with no --bimg (blank).
    const url=item.url||ev.dataTransfer.getData("url");
    const b={id:uid(),kind:"image",imageId,url,
      prompt:item.prompt||"", promptStatus:item.status==="ready"?"ready":(state.autoprompt?"thinking":"none"),
      _risk:item.risk||"unknown",
      start:place.start,dur:place.dur};
    state.blocks.push(b); trackEl.appendChild(makeBlockEl(b));
    sortBlocks(); layout(); updateGating(); selectBlock(b);
    // if the tray prompt isn't ready yet, make sure it lands when it arrives
    if(state.autoprompt && (!item.prompt) ) { b.promptStatus="thinking"; updateBlockLabels(b); }
    return;
  }
  // find room: from t up to next block start (or duration), min MIN_BLOCK
  let rightStart=state.duration, leftEnd=0;
  for(const o of state.blocks){
    if(o.start>=t) rightStart=Math.min(rightStart,o.start);
    if(o.start+o.dur<=t) leftEnd=Math.max(leftEnd,o.start+o.dur);
  }
  // if dropped inside a block, snap start to nearest free edge
  const inside=state.blocks.find(o=>t>o.start && t<o.start+o.dur);
  if(inside){ t=inside.start+inside.dur; for(const o of state.blocks){ if(o.start>=t) rightStart=Math.min(rightStart,o.start);} }
  const room=rightStart-t;
  if(room<MIN_BLOCK){ flashHelp("no room there — try a gap on the track"); return; }
  const dur=Math.min(DEFAULT_BLOCK, room);

  if(kind==="continue"){
    // must have a renderable block somewhere before it
    const hasLeft=state.blocks.some(o=>o.kind==="image" && o.start<t+1e-6);
    if(!hasLeft){ flashHelp("drop a Continue AFTER an image — it grows from that image’s last frame"); return; }
    // butt it against the nearest block to the left if close
    let butt=leftEnd>0?leftEnd:t;
    const startT = (t-butt)<1.5 ? butt : t;
    const b={id:uid(),kind:"continue",prompt:"",promptStatus:"none",start:startT,dur:Math.min(dur,rightStart-startT)};
    state.blocks.push(b); trackEl.appendChild(makeBlockEl(b));
    sortBlocks(); layout(); updateGating(); selectBlock(b);
    return;
  }
}

function sortBlocks(){ state.blocks.sort((a,b)=>a.start-b.start); }

// ---- prompt modal ----------------------------------------------------------
const modal=$("#modal");
let modalBlock=null;
function openModal(b){
  modalBlock=b; selectBlock(b);
  $("#modal-title").textContent = b.kind==="image" ? "edit prompt" : "continue — optional guidance";
  const img=$("#modal-img");
  if(b.kind==="image" && b.url){ img.src=b.url; img.style.display=""; }
  else { img.style.display="none"; }
  $("#modal-sub").textContent = `${fmt(b.start)} → ${fmt(b.start+b.dur)}  ·  ${b.dur.toFixed(1)}s  ·  ${Math.round(b.dur*state.fps)} frames`
    + (b.kind==="continue" ? "  ·  first frame comes from the previous clip" : "");
  $("#modal-prompt").value = b.prompt||"";
  $("#modal-prompt").placeholder = b.kind==="image"
    ? "describe how this image should move to the music..."
    : "optional: how should the continued motion feel? (leave blank to just flow on)";
  $("#modal-resuggest").style.display = (b.kind==="image" && state.autoprompt) ? "" : "none";
  modal.classList.remove("hidden");
  $("#modal-prompt").focus();
}
function closeModal(){ modal.classList.add("hidden"); modalBlock=null; }
$("#modal-close").addEventListener("click",closeModal);
$("#modal").addEventListener("click",e=>{ if(e.target===modal) closeModal(); });
$("#modal-save").addEventListener("click",()=>{
  if(!modalBlock) return;
  modalBlock.prompt=$("#modal-prompt").value;
  modalBlock.promptStatus="edited"; modalBlock._userPrompt=true;
  updateBlockLabels(modalBlock); closeModal();
});
$("#modal-remove").addEventListener("click",()=>{
  if(!modalBlock) return;
  state.blocks=state.blocks.filter(x=>x.id!==modalBlock.id);
  renderTimeline(); updateGating(); closeModal();
});
$("#modal-resuggest").addEventListener("click",async()=>{
  if(!modalBlock||modalBlock.kind!=="image") return;
  const btn=$("#modal-resuggest"); btn.classList.add("thinking"); btn.textContent="⟳ thinking…";
  try{
    const res=await api("/api/autoprompt",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image_id:modalBlock.imageId})});
    $("#modal-prompt").value=res.prompt||"";
    if(res.source==="heuristic") flashHelp("auto-prompt offline ("+(res.error||"no vision")+") — generic prompt; edit below");
  }catch(e){}
  btn.classList.remove("thinking"); btn.textContent="⟳ re-suggest (auto-prompt)";
});
document.addEventListener("keydown",e=>{ if(e.key==="Escape"&&!modal.classList.contains("hidden")) closeModal(); });

// ---- zoom + scrub ----------------------------------------------------------
$("#zoom").addEventListener("input",e=>{ zoomTo(parseFloat(e.target.value)); layout(); repositionPlayhead(); });
$("#zoom-fit").addEventListener("click",()=>{ fitZoom(); layout(); repositionPlayhead(); });
function repositionPlayhead(){ if(state.duration) playhead.style.left=xOf(previewAudio.currentTime)+"px"; }
window.addEventListener("resize",()=>{ if(state.duration){ layout(); } });

// ---- playback / transport --------------------------------------------------
// Lets you play the track while watching the timeline, so you can hear where the
// vocals / chorus land and size + place image blocks to match. No rendering is
// involved; this is purely the preview <audio> element driving the playhead.
const previewAudio=$("#preview-audio"), playhead=$("#tl-playhead");
const playBtn=$("#tl-play"), playIcon=playBtn.querySelector(".ph-icon"), timeEl=$("#tl-time");

function setPlayIcon(){
  const playing=!previewAudio.paused && !previewAudio.ended;
  playIcon.textContent=playing ? "⏸" : "▶";
  playBtn.classList.toggle("playing",playing);
  playBtn.setAttribute("aria-label",playing ? "pause" : "play");
}
function updateTimeReadout(t){
  timeEl.innerHTML=`<span class="cur">${fmt(t)}</span> / ${fmt(state.duration||0)}`;
}
function movePlayhead(t){
  playhead.style.display="block";
  playhead.style.left=xOf(t)+"px";
  // keep the playhead in view when zoomed in: nudge scroll if it nears an edge
  const px=xOf(t), left=scrollEl.scrollLeft, vw=scrollEl.clientWidth;
  if(px>left+vw-80 || px<left+40) scrollEl.scrollLeft=clamp(px-vw*0.4,0,innerEl.clientWidth-vw);
  updateTimeReadout(t);
  updatePlayingBlock(t);
}

// smooth rAF playhead while playing (timeupdate alone is only ~4fps and jerky)
let _raf=null;
function tick(){
  if(previewAudio.paused||previewAudio.ended){ _raf=null; return; }
  movePlayhead(previewAudio.currentTime);
  _raf=requestAnimationFrame(tick);
}
function startTick(){ if(_raf==null) _raf=requestAnimationFrame(tick); }

function togglePlay(){
  if(!state.duration) return;                 // nothing loaded yet
  if(previewAudio.ended) previewAudio.currentTime=0;
  previewAudio.paused ? previewAudio.play() : previewAudio.pause();
}
playBtn.addEventListener("click",togglePlay);

previewAudio.addEventListener("play",()=>{ setPlayIcon(); startTick(); });
previewAudio.addEventListener("pause",()=>{ setPlayIcon(); movePlayhead(previewAudio.currentTime); });
previewAudio.addEventListener("ended",()=>{ setPlayIcon(); updatePlayingBlock(previewAudio.currentTime); });
previewAudio.addEventListener("loadedmetadata",()=>{ updateTimeReadout(0); setPlayIcon(); });
// fallback for browsers between rAF frames / when paused
previewAudio.addEventListener("timeupdate",()=>{ if(previewAudio.paused) movePlayhead(previewAudio.currentTime); });

// click ruler/waveform to scrub; keeps playing if it was playing
function seekFromEvent(ev){
  const rect=innerEl.getBoundingClientRect();
  const t=clamp(tOf(ev.clientX-rect.left),0,state.duration);
  previewAudio.currentTime=t; movePlayhead(t);
}
rulerEl.addEventListener("click",seekFromEvent);
waveCanvas.addEventListener("click",ev=>{ seekFromEvent(ev); if(previewAudio.paused) previewAudio.play(); });

// drag the playhead grip to reset where playback starts. Pauses while dragging
// so the rAF tick doesn't fight the pointer, then resumes if it was playing.
const playheadGrip=$("#tl-playhead-grip");
let _drag=null;
function playheadDragMove(ev){
  if(!_drag) return;
  const rect=innerEl.getBoundingClientRect();
  const t=clamp(tOf(ev.clientX-rect.left),0,state.duration);
  previewAudio.currentTime=t; movePlayhead(t);
}
function playheadDragEnd(ev){
  if(!_drag) return;
  const wasPlaying=_drag.wasPlaying; _drag=null;
  playhead.classList.remove("dragging");
  try{ playheadGrip.releasePointerCapture(ev.pointerId); }catch(_){}
  if(wasPlaying) previewAudio.play();
}
playheadGrip.addEventListener("pointerdown",ev=>{
  if(!state.duration) return;
  ev.preventDefault(); ev.stopPropagation();
  _drag={wasPlaying:!previewAudio.paused && !previewAudio.ended};
  if(_drag.wasPlaying) previewAudio.pause();
  playhead.classList.add("dragging");
  try{ playheadGrip.setPointerCapture(ev.pointerId); }catch(_){}
  playheadDragMove(ev);
});
playheadGrip.addEventListener("pointermove",playheadDragMove);
playheadGrip.addEventListener("pointerup",playheadDragEnd);
playheadGrip.addEventListener("pointercancel",playheadDragEnd);

// spacebar = play/pause (unless typing in the prompt modal)
document.addEventListener("keydown",e=>{
  if(e.code!=="Space") return;
  const t=e.target, typing=t && (t.tagName==="TEXTAREA"||t.tagName==="INPUT");
  if(typing || !modal.classList.contains("hidden")) return;
  e.preventDefault(); togglePlay();
});

function flashHelp(msg){ const h=$("#tl-help"); h.textContent=msg; h.style.color="var(--hot-2)";
  clearTimeout(flashHelp._t); flashHelp._t=setTimeout(()=>{h.style.color="";
    h.textContent="tip: click a clip to edit its prompt · drag the right edge to resize · beats are the faint ticks";},4200); }

// ---- toasts (Phase D Tier 2) -----------------------------------------------
// A transient popup over the render panel so image substitutions are not
// buried in the log. Distinct copy for an OOM vs a generic render error
// (they need different reactions: shrink the render vs. just wait/retry).
function toastHost(){
  let h=$("#toast-host");
  if(!h){ h=document.createElement("div"); h.id="toast-host"; document.body.appendChild(h); }
  return h;
}
function toast(msg, cls="", ttl=6500){
  const t=document.createElement("div"); t.className="toast "+cls;
  t.textContent=msg;
  toastHost().appendChild(t);
  requestAnimationFrame(()=>t.classList.add("show"));
  setTimeout(()=>{ t.classList.remove("show"); setTimeout(()=>t.remove(),300); }, ttl);
}
function toastOOM(index){
  toast(`Block ${index+1} ran the GPU out of memory. Sizzle is substituting `
      + `another image so the timing holds — if it keeps happening, drop to a `
      + `smaller output format or a shorter block.`, "warn", 8000);
}

// ---- gating + generate -----------------------------------------------------
// GENERATE button truth lives here. It is disabled unless there is ≥1 image AND
// no render is in flight — either this session's (state.activeJobId) or someone
// else's on the tunnel (state.othersRendering, from the /api/status poll). This
// is the client half of the global single-render lock (decision 3).
function applyGenerateButton(){
  const gen=$("#generate");
  const imgs=state.blocks.filter(b=>b.kind==="image").length;
  if(state.activeJobId){
    gen.disabled=true; gen.textContent="RENDERING…"; gen.classList.add("rendering"); return;
  }
  gen.classList.remove("rendering");
  if(state.othersRendering){
    gen.disabled=true; gen.textContent="someone is rendering…"; return;
  }
  gen.textContent="GENERATE";
  gen.disabled = imgs<1;
}

function updateGating(){
  const imgs=state.blocks.filter(b=>b.kind==="image").length;
  const cont=state.blocks.filter(b=>b.kind==="continue").length;
  const note=$("#fill-note"), stepGen=$("#step-generate");
  if(imgs>=1){
    stepGen.classList.remove("locked");
    const s=Math.min(...state.blocks.map(b=>b.start));
    const e=Math.max(...state.blocks.map(b=>b.start+b.dur));
    note.textContent=`${imgs} image${imgs>1?"s":""}${cont?` + ${cont} continue${cont>1?"s":""}`:""} · renders ${(e-s).toFixed(1)}s (${fmt(s)}–${fmt(e)})`;
  } else {
    stepGen.classList.add("locked"); note.textContent="";
  }
  applyGenerateButton();
  const dense = state.blocks.length;
  $("#tl-meta").textContent = dense ? `${dense} block${dense>1?"s":""} on timeline` : "drag an image below onto the track";
}

// Ghost/un-ghost the whole render session: button + timeline hard-lock veil.
// body.rendering drives the CSS veil over the timeline (pointer-events off) so
// blocks stay visible for the render animation but can't be edited (decision 6).
function setRendering(on){
  document.body.classList.toggle("rendering", on);
  // lock format + variant while a render is consuming this resolution (decision
  // 2/6): a mid-render change would mismatch the clips already produced.
  ["#format","#variant","#fmt-w","#fmt-h"].forEach(s=>{ const el=$(s); if(el) el.disabled=on; });
  if(!on) state.activeJobId=null;
  updateGating();
}

// Cross-client reflection (A4): poll the global render flag so a second visitor
// sees "someone is rendering…" even though their browser never fired the job.
async function pollRenderState(){
  try{
    const st=await api("/api/status");
    state.othersRendering = !!st.rendering && !st.rendering_mine && !state.activeJobId;
    applyGenerateButton();
  }catch(e){/* transient; keep last known */}
}
setInterval(pollRenderState, 4000);

// Gaps between consecutive clips — the empty stretches that render as black
// fill. Returns the adjacent pairs whose edges aren't butted together so we can
// warn about them and highlight the exact clips.
function findGaps(){
  const bs=[...state.blocks].sort((a,b)=>a.start-b.start);
  const gaps=[];
  for(let i=1;i<bs.length;i++){
    const prev=bs[i-1], cur=bs[i];
    const g=cur.start-(prev.start+prev.dur);
    if(g>GAP_MIN) gaps.push({prev, cur, gap:g});
  }
  return gaps;
}

// GENERATE: don't auto-correct gaps — surface them once, here, and let the user
// decide (a gap can be intentional, e.g. rendering a single segment).
$("#generate").addEventListener("click", ()=>{
  if(state.activeJobId || state.othersRendering) return;   // locked
  sortBlocks();
  const gaps=findGaps();
  if(gaps.length){ openGapModal(gaps); return; }
  runGenerate();
});

async function runGenerate(){
  if(state.activeJobId || state.othersRendering) return;   // locked
  sortBlocks();
  clearGapHighlights();
  const blocks=state.blocks.map(b=>({
    kind:b.kind, start:b.start, end:b.start+b.dur,
    image_id:b.imageId, prompt:b.prompt||"",
  }));
  // Snapshot block order → ids so WS events (which carry the block INDEX in this
  // submitted order) can be mapped back to the exact DOM element (Phase B).
  state.renderMap = state.blocks.map(b=>b.id);
  clearRenderStates();
  const seedVal=$("#seed").value;
  // Ghost immediately (optimistic) so a double-click can't fire a second job.
  state.activeJobId = "pending"; setRendering(true);
  let res;
  try{
    res=await api("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({audio_id:state.audioId, variant:state.variant,
        seed:seedVal?parseInt(seedVal):null, blocks,
        width:state.format&&state.format.width, height:state.format&&state.format.height})});
  }catch(e){ setRendering(false); flashHelp("generate error: network"); return; }
  if(res.error){
    setRendering(false);
    flashHelp(/already running/.test(res.error) ? "someone is already rendering — hang tight" : "generate error: "+res.error);
    return;
  }
  state.activeJobId = res.job_id;
  openProgress(res.job_id,res.total);
}

// ---- gap warning modal -----------------------------------------------------
const gapModal=$("#gap-modal");
function clearGapHighlights(){
  trackEl.querySelectorAll(".block.gap-warn").forEach(el=>el.classList.remove("gap-warn"));
}
function openGapModal(gaps){
  const n=gaps.length, total=gaps.reduce((s,g)=>s+g.gap,0);
  $("#gap-msg").innerHTML =
    `Found <b>${n}</b> gap${n>1?"s":""} between clips — ${total.toFixed(2)}s of empty `+
    `timeline that will render as <b>black fill</b>. Return to close the gaps up, or `+
    `generate anyway if they're intentional (e.g. rendering just a segment).`;
  gapModal._gaps=gaps;
  gapModal.classList.remove("hidden");
}
function closeGapModal(){ gapModal.classList.add("hidden"); gapModal._gaps=null; }
$("#gap-return").addEventListener("click",()=>{
  const gaps=gapModal._gaps||[];
  closeGapModal();
  clearGapHighlights();
  // highlight every clip touching a gap, then scroll the first offender into view
  for(const g of gaps){
    if(g.prev._el) g.prev._el.classList.add("gap-warn");
    if(g.cur._el) g.cur._el.classList.add("gap-warn");
  }
  if(gaps[0] && gaps[0].cur._el) scrollBlockIntoView(gaps[0].cur._el);
  flashHelp("highlighted the clips with gaps — drag them together to close the gap");
});
$("#gap-generate").addEventListener("click",()=>{ closeGapModal(); runGenerate(); });
gapModal.addEventListener("click",e=>{ if(e.target===gapModal) closeGapModal(); });
document.addEventListener("keydown",e=>{ if(e.key==="Escape"&&!gapModal.classList.contains("hidden")) closeGapModal(); });

// ---- per-block render state (Phase B) --------------------------------------
const R_CLASSES = ["r-queued","r-rendering","r-done","r-substituted","r-retry","r-filled"];
function clearRenderStates(){
  trackEl.querySelectorAll(".block").forEach(el=>el.classList.remove(...R_CLASSES));
}
function blockElForIndex(i){
  const id = state.renderMap && state.renderMap[i];
  return id ? trackEl.querySelector(`.block[data-id="${id}"]`) : null;
}
// set the persistent render state on a block (removes the other persistent ones;
// r-retry is handled separately as a transient flash so it can ride on top).
function setBlockState(i, cls){
  const el = blockElForIndex(i);
  if(!el) return null;
  el.classList.remove("r-queued","r-rendering","r-done","r-substituted","r-filled");
  if(cls) el.classList.add(cls);
  return el;
}
function scrollBlockIntoView(el){
  if(!el) return;
  const px = parseFloat(el.style.left)||0, vw = scrollEl.clientWidth;
  scrollEl.scrollLeft = clamp(px - vw*0.4, 0, Math.max(0, innerEl.clientWidth - vw));
}

function openProgress(jobId,total){
  $("#progress").classList.remove("hidden"); $("#result").classList.add("hidden");
  const log=$("#progress-log"), fill=$("#progress-fill"); log.textContent="";
  let finished=false;
  const line=(t,cls="",url=null)=>{
    const d=document.createElement("div");if(cls)d.className=cls;
    d.appendChild(document.createTextNode(t));
    if(url){
      // Download-only (decision 4): no target=_blank, no inline play. ?dl=1 makes
      // the server send Content-Disposition: attachment; the download attr backs
      // it up for same-origin so the render page is never navigated away from.
      const a=document.createElement("a");
      a.href = url + (url.indexOf("?")>=0 ? "&" : "?") + "dl=1";
      a.setAttribute("download","");
      a.className="clip-link"; a.textContent=" ⬇ download clip";
      d.appendChild(a);
    }
    log.appendChild(d);log.scrollTop=log.scrollHeight;};
  const ws=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/${jobId}`);
  ws.onmessage=(ev)=>{
    const e=JSON.parse(ev.data);
    switch(e.type){
      case "queued": line(`queued // ${e.position} ahead of you`); break;
      case "start":
        line(`rendering window ${e.window?e.window[0]+"–"+e.window[1]+"s":""} · ${total} block(s)…`,"live");
        (state.renderMap||[]).forEach((_,i)=>setBlockState(i,"r-queued"));
        break;
      case "clip_start": {
        line(`block ${e.current}/${e.total} [${e.kind||"image"}] :: ${(e.prompt||"").slice(0,64)}`);
        const el=setBlockState(e.index,"r-rendering"); scrollBlockIntoView(el);
        break; }
      case "gap_filled": line(`gap ${e.seconds}s → black fill`,"warn",e.clip_url); break;
      case "clip_retry": {
        line(`block ${e.index+1} :: ${e.reason} on ${e.failed_image} — recovering`,"warn");
        const el=blockElForIndex(e.index);
        if(el){ el.classList.add("r-retry"); setTimeout(()=>el.classList.remove("r-retry"),700); }
        if(e.reason==="GPU out of memory") toastOOM(e.index);
        else toast(`Block ${e.index+1}: ${e.reason||"render error"} — retrying.`,"warn");
        break; }
      case "clip_done": fill.style.width=`${(e.current/e.total)*90}%`;
        setBlockState(e.index, e.substituted?"r-substituted":"r-done");
        if(e.substituted){ const el=blockElForIndex(e.index); if(el&&e.note) el.title=e.note; }
        line(e.substituted?`block ${e.current}/${e.total} done (${e.note})`:`block ${e.current}/${e.total} done`, e.substituted?"warn":"live", e.clip_url); break;
      case "clip_filled": fill.style.width=`${(e.current/e.total)*90}%`;
        setBlockState(e.index,"r-filled");
        { const el=blockElForIndex(e.index); if(el&&e.message) el.title=e.message; }
        line(`block ${e.current}/${e.total} :: ${e.message||"filled to keep timing"}`,"warn",e.clip_url); break;
      case "muxing": fill.style.width="95%"; line("muxing audio + concat…","live"); break;
      case "done": finished=true; fill.style.width="100%"; line("done.","live");
        showResult(e.download); setRendering(false); ws.close(); break;
      case "error": finished=true; line("ERROR: "+e.message,"err"); setRendering(false); ws.close(); break;
    }
  };
  // Un-ghost on any unexpected drop so the user regains control to retry.
  ws.onclose=()=>{ if(!finished){ line("connection to render lost — you can try again","err"); setRendering(false); } };
  ws.onerror=()=>{ if(!finished){ setRendering(false); } };
}
function showResult(downloadUrl){
  $("#result").classList.remove("hidden");
  $("#result-video").src=downloadUrl; $("#download").href=downloadUrl;
  $("#result").scrollIntoView({behavior:"smooth",block:"nearest"});
}

// ---- boot ------------------------------------------------------------------
loadStatus();
