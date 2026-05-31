(() => {
  "use strict";

  const canvas = document.getElementById("quantized");
  const ctx = canvas.getContext("2d", { alpha: false });
  const panel = document.getElementById("panel");
  const panelToggle = document.getElementById("panelToggle");
  const resetParams = document.getElementById("resetParams");
  const audioToggle = document.getElementById("audioToggle");
  const bedAudio = document.getElementById("bedAudio");

  const WORLD_W = 1920;
  const WORLD_H = 1080;
  const SUB_W = 480;
  const SUB_H = 270;
  const TAU = Math.PI * 2;
  const DIRECTOR_ENABLED = true;
  const DIRECTOR_ENDPOINT = window.QUANTIZED_DIRECTOR_ENDPOINT || "/api/director";
  const DIRECTOR_MIN_STILL_MS = 3000;
  const DIRECTOR_FOLLOWUP_MIN_MS = 10000;
  const DIRECTOR_FOLLOWUP_MAX_MS = 240000;
  const DIRECTOR_MAX_ACTIONS = 18;
  const DIRECTOR_HEAT_W = 12;
  const DIRECTOR_HEAT_H = 7;
  const DIRECTOR_LOGGING = window.QUANTIZED_DIRECTOR_LOGGING !== false;

  const palette = [
    [132, 92, 238],
    [180, 255, 0],
    [0, 184, 96],
    [0, 214, 255],
    [246, 48, 176],
    [255, 90, 28],
    [4, 38, 206],
    [255, 238, 82],
    [232, 124, 96],
    [0, 0, 0],
  ];

  const defaults = {
    theme: "god",
    algorithm: "god",
    motion: 1.45,
    density: 1.65,
    blur: 2,
    rayBlur: 0,
    raySpeed: 1,
    glint: 1.4,
    blocks: 0,
  };

  const controls = { ...defaults };
  const themes = {
    god: {
      base: [8, 6, 24],
      palette: [0, 1, 2, 3, 4, 5, 6, 7],
      mix: 1,
      glint: [245, 250, 255],
    },
    bitplane: {
      base: [4, 7, 18],
      palette: [3, 6, 1, 4, 7],
      mix: 0.72,
      glint: [190, 245, 255],
    },
    phosphor: {
      base: [2, 13, 9],
      palette: [2, 3, 1, 7],
      mix: 0.68,
      glint: [215, 255, 210],
    },
    ember: {
      base: [15, 7, 5],
      palette: [5, 8, 7, 1, 4],
      mix: 0.76,
      glint: [255, 232, 170],
    },
    mono: {
      base: [5, 5, 16],
      palette: [0, 3, 6, 7],
      mix: 0.5,
      glint: [225, 235, 255],
    },
  };
  const algorithms = {
    god: {
      threshold: 0,
      snap: 1,
      group: 1,
      bit: 0.04,
      flow: 1,
      quant: 0,
    },
    bitplane: {
      threshold: 0.04,
      snap: 4,
      group: 12,
      bit: 0.16,
      flow: 0.7,
      quant: 1,
    },
    cellular: {
      threshold: 0.025,
      snap: 3,
      group: 11,
      bit: 0.18,
      flow: 0.78,
      quant: 0.8,
    },
    scanline: {
      threshold: 0.02,
      snap: 2,
      group: 16,
      bit: 0.12,
      flow: 0.64,
      quant: 0.55,
    },
    lattice: {
      threshold: 0.035,
      snap: 4,
      group: 10,
      bit: 0.22,
      flow: 0.58,
      quant: 1.15,
    },
    interference: {
      threshold: 0.015,
      snap: 3,
      group: 13,
      bit: 0.08,
      flow: 0.9,
      quant: 0.45,
    },
  };
  const state = {
    dpr: 1,
    zoom: 1,
    panX: 0,
    panY: 0,
    pointerX: WORLD_W * 0.5,
    pointerY: WORLD_H * 0.5,
    pointerDown: false,
    spaceDown: false,
    dragX: 0,
    dragY: 0,
    downX: 0,
    downY: 0,
    downTime: 0,
    downWorldX: 0,
    downWorldY: 0,
    lastWorldX: 0,
    lastWorldY: 0,
    dragDistance: 0,
    activeGesture: "none",
    activeRayIndex: -1,
    activeRayMode: "none",
    perturb: 0,
    keyboardPhase: 0,
    frame: 0,
    lastTime: 0,
    observerStillness: 0.4,
    observerAgitation: 0,
    observerAttention: 0,
    observerRayAffinity: 0,
    observerPressure: 0,
    observerLastMove: 0,
    observerEngaged: false,
  };

  const substrateCanvas = document.createElement("canvas");
  substrateCanvas.width = SUB_W;
  substrateCanvas.height = SUB_H;
  const substrateCtx = substrateCanvas.getContext("2d", { alpha: false });
  const substrateImage = substrateCtx.createImageData(SUB_W, SUB_H);

  const glintCanvas = document.createElement("canvas");
  glintCanvas.width = SUB_W;
  glintCanvas.height = SUB_H;
  const glintCtx = glintCanvas.getContext("2d", { alpha: true });
  const glintImage = glintCtx.createImageData(SUB_W, SUB_H);

  const fieldA = new Float32Array(SUB_W * SUB_H);
  const fieldB = new Float32Array(SUB_W * SUB_H);
  const rayField = new Float32Array(SUB_W * SUB_H);
  const rayDepthField = new Float32Array(SUB_W * SUB_H);
  const rayBrillianceField = new Float32Array(SUB_W * SUB_H);
  const probeField = new Float32Array(SUB_W * SUB_H);
  const dragField = new Float32Array(SUB_W * SUB_H);
  const collapseField = new Float32Array(SUB_W * SUB_H);
  const measurementField = new Float32Array(SUB_W * SUB_H);
  const rayPerturbColorField = new Float32Array(SUB_W * SUB_H);
  const rayPerturbColor = [255, 54, 18];

  const rand = mulberry32(xfnv1a("Quantized V2 / darker field / ray perturbation"));
  const rayFamilies = makeRayFamilies();
  const temporaryRayFamilies = [];
  const fossils = [];
  const eventMemories = [];
  const blocks = makeBlocks();
  const directorEffects = [];
  const directorTweens = [];
  const globalReconfiguration = {
    active: false,
    elapsed: 0,
    duration: 0,
    cooldown: 96,
    from: [],
    to: [],
    salt: 0,
  };
  const audioLayer = {
    ctx: null,
    enabled: false,
    loading: false,
    loaded: false,
    buffers: {},
    buses: {},
    filters: {},
    master: null,
    bed: null,
    bedTheme: "",
    noiseBuffer: null,
    lastProbe: 0,
    lastRayDrag: 0,
    lastScroll: 0,
    lastFossil: 0,
    lastGlint: 0,
    glintKeys: new Map(),
  };
  const directorState = {
    enabled: DIRECTOR_ENABLED,
    pending: false,
    failed: false,
    sessionStart: performance.now(),
    lastInput: performance.now(),
    lastCall: 0,
    nextDue: performance.now() + lerp(4000, 8000, rand()),
    cadenceIndex: 0,
    lastIdleBucket: -1,
    memoryNote: "",
    carry: "",
    recentViewer: [],
    recentAi: [],
    heatmap: new Uint16Array(DIRECTOR_HEAT_W * DIRECTOR_HEAT_H),
    counts: {
      clicks: 0,
      longPresses: 0,
      drags: 0,
      rayOriginDrags: 0,
      wheel: 0,
      keys: 0,
      idlePeriods: 0,
    },
    themeCooldown: 0,
    algorithmCooldown: 0,
  };

  const audioCategories = [
    "beds",
    "brilliance",
    "probe",
    "click",
    "rayHandle",
    "scroll",
    "fossilCreate",
    "fossilDisturb",
    "partials",
    "impulses",
    "textures",
    "glitches",
    "sub",
  ];
  const audioCategoryAliases = {
    beds: ["beds"],
    brilliance: ["brilliance", "partials"],
    probe: ["probe", "textures"],
    click: ["click", "impulses"],
    rayHandle: ["rayHandle", "partials"],
    scroll: ["scroll", "sub"],
    fossilCreate: ["fossilCreate", "glitches"],
    fossilDisturb: ["fossilDisturb", "glitches"],
  };
  const audioThemes = ["god", "bitplane", "phosphor", "ember", "mono"];
  const audioThemeVoices = {
    god: { root: 49, spread: 1.41, filter: 1800, bed: 0.25, ray: 0.58, color: "sine" },
    bitplane: { root: 65, spread: 2.01, filter: 2350, bed: 0.2, ray: 0.5, color: "square" },
    phosphor: { root: 58, spread: 1.5, filter: 1500, bed: 0.22, ray: 0.44, color: "triangle" },
    ember: { root: 43, spread: 1.34, filter: 950, bed: 0.24, ray: 0.48, color: "sawtooth" },
    mono: { root: 55, spread: 1.26, filter: 1300, bed: 0.18, ray: 0.4, color: "sine" },
  };
  const audioAlgorithmVoices = {
    god: { tempo: 0.72, pitch: 0, grain: 1, sub: 0.74 },
    bitplane: { tempo: 1.1, pitch: 7, grain: 0.86, sub: 0.56 },
    cellular: { tempo: 1.28, pitch: 12, grain: 1.18, sub: 0.62 },
    scanline: { tempo: 1.48, pitch: 19, grain: 0.62, sub: 0.52 },
    lattice: { tempo: 0.95, pitch: 5, grain: 1.42, sub: 0.68 },
    interference: { tempo: 0.82, pitch: -5, grain: 1.24, sub: 0.82 },
  };

  function xfnv1a(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i += 1) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(seed) {
    return function next() {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function smoothstep(edge0, edge1, x) {
    const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  function hash2(x, y, salt) {
    let h = Math.imul(x ^ salt, 374761393) + Math.imul(y, 668265263);
    h = (h ^ (h >>> 13)) >>> 0;
    h = Math.imul(h, 1274126177) >>> 0;
    return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
  }

  function valueNoise(x, y, scale, salt) {
    const fx = x / scale;
    const fy = y / scale;
    const ix = Math.floor(fx);
    const iy = Math.floor(fy);
    const tx = fx - ix;
    const ty = fy - iy;
    const ax = smoothstep(0, 1, tx);
    const ay = smoothstep(0, 1, ty);
    const a = hash2(ix, iy, salt);
    const b = hash2(ix + 1, iy, salt);
    const c = hash2(ix, iy + 1, salt);
    const d = hash2(ix + 1, iy + 1, salt);
    return lerp(lerp(a, b, ax), lerp(c, d, ax), ay);
  }

  function normalize3(x, y, z) {
    const mag = Math.hypot(x, y, z) || 1;
    return { x: x / mag, y: y / mag, z: z / mag };
  }

  function easeInOut(t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  function rotate3(v, axis, angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const dot = v.x * axis.x + v.y * axis.y + v.z * axis.z;
    return {
      x: v.x * c + (axis.y * v.z - axis.z * v.y) * s + axis.x * dot * (1 - c),
      y: v.y * c + (axis.z * v.x - axis.x * v.z) * s + axis.y * dot * (1 - c),
      z: v.z * c + (axis.x * v.y - axis.y * v.x) * s + axis.z * dot * (1 - c),
    };
  }

  function makeRayFamilies() {
    const origins = [
      [0.25, 0.72],
      [0.56, 0.42],
      [0.79, 0.58],
      [0.42, 0.25],
    ];
    return origins.map((origin, f) => {
      const rays = [];
      for (let i = 0; i < 28; i += 1) {
        const z = lerp(-1, 1, rand());
        const a = rand() * TAU;
        const r = Math.sqrt(1 - z * z);
        rays.push({
          x: Math.cos(a) * r,
          y: Math.sin(a) * r,
          z,
          length: lerp(640, 2300, rand()),
          width: lerp(1.2, 3.8, rand()),
          phase: rand() * TAU,
        });
      }
      return {
        originX: origin[0] * WORLD_W,
        originY: origin[1] * WORLD_H,
        baseOriginX: origin[0] * WORLD_W,
        baseOriginY: origin[1] * WORLD_H,
        offsetX: 0,
        offsetY: 0,
        globalOffsetX: 0,
        globalOffsetY: 0,
        projectionNudge: 0,
        axisNudge: normalize3(0, 0, 0),
        decay: 1,
        axis: normalize3(lerp(-1, 1, rand()), lerp(-1, 1, rand()), lerp(-1, 1, rand())),
        speed: lerp(-0.18, 0.22, rand()) || (f % 2 ? -0.14 : 0.14),
        phase: rand() * TAU,
        projection: lerp(0.58, 1.04, rand()),
        rays,
      };
    });
  }

  function makeTemporaryRayFamily(x, y) {
    const rays = [];
    for (let i = 0; i < 18; i += 1) {
      const z = lerp(-1, 1, rand());
      const a = rand() * TAU;
      const r = Math.sqrt(1 - z * z);
      rays.push({
        x: Math.cos(a) * r,
        y: Math.sin(a) * r,
        z,
        length: lerp(260, 1300, rand()),
        width: lerp(0.8, 2.8, rand()),
        phase: rand() * TAU,
      });
    }
    return {
      originX: x,
      originY: y,
      baseOriginX: x,
      baseOriginY: y,
      offsetX: 0,
      offsetY: 0,
      globalOffsetX: 0,
      globalOffsetY: 0,
      projectionNudge: 0,
      axisNudge: normalize3(0, 0, 0),
      decay: 1,
      axis: normalize3(lerp(-1, 1, rand()), lerp(-1, 1, rand()), lerp(-1, 1, rand())),
      speed: lerp(-0.34, 0.34, rand()) || 0.22,
      phase: rand() * TAU,
      projection: lerp(0.56, 0.98, rand()),
      rays,
    };
  }

  function familyOrigin(family) {
    return {
      x: family.originX + family.offsetX + (family.globalOffsetX || 0),
      y: family.originY + family.offsetY + (family.globalOffsetY || 0),
    };
  }

  function makeBlocks() {
    const out = [];
    for (let i = 0; i < 14; i += 1) {
      const cell = [8, 12, 16, 24, 32][Math.floor(rand() * 5)];
      out.push({
        x: Math.floor((rand() * WORLD_W) / cell) * cell,
        y: Math.floor((rand() * WORLD_H) / cell) * cell,
        w: Math.floor(lerp(48, 340, rand() * rand()) / cell) * cell,
        h: Math.floor(lerp(32, 220, rand() * rand()) / cell) * cell,
        color: palette[Math.floor(rand() * 8)],
        accent: palette[Math.floor(rand() * 8)],
        cell,
        salt: Math.floor(rand() * 9999),
      });
    }
    return out;
  }

  function resize() {
    state.dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = Math.floor(window.innerWidth * state.dpr);
    canvas.height = Math.floor(window.innerHeight * state.dpr);
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    if (state.frame === 0) {
      state.zoom = Math.max(window.innerWidth / WORLD_W, window.innerHeight / WORLD_H);
      state.panX = (window.innerWidth - WORLD_W * state.zoom) * 0.5;
      state.panY = (window.innerHeight - WORLD_H * state.zoom) * 0.5;
    }
    clampView();
  }

  function minCoverZoom() {
    return Math.max(window.innerWidth / WORLD_W, window.innerHeight / WORLD_H);
  }

  function clampView() {
    state.zoom = Math.max(state.zoom, minCoverZoom());
    const drawnW = WORLD_W * state.zoom;
    const drawnH = WORLD_H * state.zoom;
    state.panX = drawnW <= window.innerWidth ? (window.innerWidth - drawnW) * 0.5 : clamp(state.panX, window.innerWidth - drawnW, 0);
    state.panY = drawnH <= window.innerHeight ? (window.innerHeight - drawnH) * 0.5 : clamp(state.panY, window.innerHeight - drawnH, 0);
  }

  function screenToWorld(x, y) {
    return {
      x: (x - state.panX) / state.zoom,
      y: (y - state.panY) / state.zoom,
    };
  }

  function worldToSubX(x) {
    return Math.round((x / WORLD_W) * SUB_W);
  }

  function worldToSubY(y) {
    return Math.round((y / WORLD_H) * SUB_H);
  }

  function rgb(c, a) {
    if (a === undefined) return `rgb(${c[0]},${c[1]},${c[2]})`;
    return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  }

  function updateAudioToggle() {
    if (!audioToggle) return;
    audioToggle.textContent = audioLayer.enabled ? "audio on" : "audio off";
  }

  function audioVoice() {
    return audioThemeVoices[controls.theme] || audioThemeVoices.god;
  }

  function audioAlgorithm() {
    return audioAlgorithmVoices[controls.algorithm] || audioAlgorithmVoices.god;
  }

  function initAudio() {
    if (audioLayer.ctx) return true;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) {
      if (audioToggle) audioToggle.textContent = "no audio";
      return false;
    }
    const ctxAudio = new AudioContext();
    audioLayer.ctx = ctxAudio;
    audioLayer.master = ctxAudio.createGain();
    audioLayer.master.gain.value = 0.0001;
    const compressor = ctxAudio.createDynamicsCompressor();
    compressor.threshold.value = -18;
    compressor.knee.value = 16;
    compressor.ratio.value = 4;
    compressor.attack.value = 0.012;
    compressor.release.value = 0.28;
    audioLayer.master.connect(compressor);
    compressor.connect(ctxAudio.destination);

    makeAudioBus("bed", 0.24, "lowpass", 1800);
    makeAudioBus("ray", 0.5, "bandpass", 2200);
    makeAudioBus("probe", 0.42, "bandpass", 1200);
    makeAudioBus("artifact", 0.38, "highpass", 720);
    makeAudioBus("sub", 0.32, "lowpass", 180);
    makeNoiseBuffer();
    hydrateBedManifest();
    loadAudioManifest();
    setAudioScene();
    updateAudioToggle();
    return true;
  }

  function makeAudioBus(name, gainValue, filterType, filterFrequency) {
    const ctxAudio = audioLayer.ctx;
    const input = ctxAudio.createGain();
    const filter = ctxAudio.createBiquadFilter();
    const gain = ctxAudio.createGain();
    filter.type = filterType;
    filter.frequency.value = filterFrequency;
    filter.Q.value = name === "sub" ? 0.7 : 1.4;
    gain.gain.value = gainValue;
    input.connect(filter);
    filter.connect(gain);
    gain.connect(audioLayer.master);
    audioLayer.buses[name] = { input, gain };
    audioLayer.filters[name] = filter;
  }

  function makeNoiseBuffer() {
    const ctxAudio = audioLayer.ctx;
    const buffer = ctxAudio.createBuffer(1, ctxAudio.sampleRate * 2, ctxAudio.sampleRate);
    const data = buffer.getChannelData(0);
    let last = 0;
    for (let i = 0; i < data.length; i += 1) {
      last = last * 0.86 + (Math.random() * 2 - 1) * 0.14;
      data[i] = last;
    }
    audioLayer.noiseBuffer = buffer;
  }

  async function loadAudioManifest() {
    if (audioLayer.loading || audioLayer.loaded) return;
    audioLayer.loading = true;
    const manifest = window.QUANTIZED_AUDIO_MANIFEST || {};
    for (const category of audioCategories) {
      audioLayer.buffers[category] = {};
      const section = manifest[category] || {};
      const keys = Array.isArray(section) ? ["all"] : Array.from(new Set(["all", ...audioThemes, ...Object.keys(section)]));
      for (const key of keys) {
        const urls = Array.isArray(section) ? section : Array.isArray(section[key]) ? section[key] : [];
        audioLayer.buffers[category][key] = [];
        if (category === "beds") {
          for (const url of urls) audioLayer.buffers[category][key].push({ url, media: true });
          continue;
        }
        await Promise.all(
          urls.map(async (url) => {
            try {
              const response = await fetch(url);
              if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
              const arrayBuffer = await response.arrayBuffer();
              const buffer = await audioLayer.ctx.decodeAudioData(arrayBuffer);
              audioLayer.buffers[category][key].push({ url, buffer });
            } catch (error) {
              console.warn(`Could not load audio sample: ${url}`, error);
            }
          }),
        );
      }
    }
    audioLayer.loaded = true;
    audioLayer.loading = false;
    startBedForTheme(controls.theme);
  }

  function hydrateBedManifest() {
    const manifest = window.QUANTIZED_AUDIO_MANIFEST || {};
    const beds = manifest.beds || {};
    audioLayer.buffers.beds = audioLayer.buffers.beds || {};
    const keys = Array.isArray(beds) ? ["all"] : Array.from(new Set(["all", ...audioThemes, ...Object.keys(beds)]));
    for (const key of keys) {
      const urls = Array.isArray(beds) ? beds : Array.isArray(beds[key]) ? beds[key] : [];
      audioLayer.buffers.beds[key] = urls.map((url) => ({ url, media: true }));
    }
  }

  function samplePool(category, theme) {
    const categories = audioCategoryAliases[category] || [category];
    for (const name of categories) {
      const byTheme = audioLayer.buffers[name] || {};
      const direct = byTheme[theme] || [];
      if (direct.length > 0) return direct;
      const shared = byTheme.all || [];
      if (shared.length > 0) return shared;
      const god = byTheme.god || [];
      if (god.length > 0) return god;
      for (const fallbackTheme of audioThemes) {
        if (byTheme[fallbackTheme] && byTheme[fallbackTheme].length > 0) return byTheme[fallbackTheme];
      }
    }
    return [];
  }

  function chooseSample(category, theme = controls.theme) {
    const pool = samplePool(category, theme);
    if (pool.length === 0) return null;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  function playSample(category, options = {}) {
    if (!audioLayer.ctx || !audioLayer.enabled) return false;
    const picked = chooseSample(category, options.theme || controls.theme);
    if (!picked || !picked.buffer) return false;
    const ctxAudio = audioLayer.ctx;
    const now = ctxAudio.currentTime + (options.delay || 0);
    const source = ctxAudio.createBufferSource();
    const gain = ctxAudio.createGain();
    source.buffer = picked.buffer;
    source.loop = !!options.loop;
    source.playbackRate.value = options.rate || 1;
    const attack = options.attack === undefined ? 0.006 : options.attack;
    const duration = options.duration || Math.min(picked.buffer.duration, 1.8);
    const release = options.release === undefined ? 0.08 : options.release;
    const level = options.gain === undefined ? 0.2 : options.gain;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(level, now + attack);
    if (!options.loop) {
      gain.gain.setTargetAtTime(0.0001, now + Math.max(attack, duration), release);
    }
    let tail = gain;
    let panner = null;
    if (typeof ctxAudio.createStereoPanner === "function" && options.pan !== undefined) {
      panner = ctxAudio.createStereoPanner();
      panner.pan.value = clamp(options.pan, -1, 1);
      gain.connect(panner);
      tail = panner;
    }
    tail.connect(audioLayer.buses[options.bus || "probe"].input);
    source.connect(gain);
    source.start(now, options.offset || 0, options.loop ? undefined : duration + release);
    if (!options.loop) source.stop(now + duration + release + 0.05);
    return { source, gain, panner };
  }

  function playTone(bus, freq, duration, gainValue, options = {}) {
    if (!audioLayer.ctx || !audioLayer.enabled) return;
    const ctxAudio = audioLayer.ctx;
    const now = ctxAudio.currentTime + (options.delay || 0);
    const osc = ctxAudio.createOscillator();
    const gain = ctxAudio.createGain();
    const panner =
      typeof ctxAudio.createStereoPanner === "function" && options.pan !== undefined ? ctxAudio.createStereoPanner() : null;
    osc.type = options.type || audioVoice().color;
    osc.frequency.setValueAtTime(Math.max(18, freq), now);
    if (options.endFreq) osc.frequency.exponentialRampToValueAtTime(Math.max(18, options.endFreq), now + duration);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, gainValue), now + (options.attack || 0.012));
    gain.gain.setTargetAtTime(0.0001, now + duration, options.release || 0.09);
    osc.connect(gain);
    if (panner) {
      panner.pan.value = clamp(options.pan, -1, 1);
      gain.connect(panner);
      panner.connect(audioLayer.buses[bus].input);
    } else {
      gain.connect(audioLayer.buses[bus].input);
    }
    osc.start(now);
    osc.stop(now + duration + (options.release || 0.09) * 4);
  }

  function playNoise(bus, duration, gainValue, options = {}) {
    if (!audioLayer.ctx || !audioLayer.enabled || !audioLayer.noiseBuffer) return;
    const ctxAudio = audioLayer.ctx;
    const now = ctxAudio.currentTime + (options.delay || 0);
    const source = ctxAudio.createBufferSource();
    const gain = ctxAudio.createGain();
    const filter = ctxAudio.createBiquadFilter();
    const panner =
      typeof ctxAudio.createStereoPanner === "function" && options.pan !== undefined ? ctxAudio.createStereoPanner() : null;
    source.buffer = audioLayer.noiseBuffer;
    source.loop = true;
    filter.type = options.filterType || "bandpass";
    filter.frequency.value = options.frequency || 1200;
    filter.Q.value = options.q || 4;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(gainValue, now + (options.attack || 0.006));
    gain.gain.setTargetAtTime(0.0001, now + duration, options.release || 0.045);
    source.connect(filter);
    filter.connect(gain);
    if (panner) {
      panner.pan.value = clamp(options.pan, -1, 1);
      gain.connect(panner);
      panner.connect(audioLayer.buses[bus].input);
    } else {
      gain.connect(audioLayer.buses[bus].input);
    }
    source.start(now, Math.random() * 1.6);
    source.stop(now + duration + (options.release || 0.045) * 4);
  }

  function bedElementVolume() {
    return audioLayer.enabled ? clamp(audioVoice().bed * 0.86, 0, 0.29) : 0;
  }

  function rampElementVolume(element, target, duration = 650) {
    if (!element) return;
    const from = element.volume;
    const to = clamp(target, 0, 1);
    const started = performance.now();
    const tick = (now) => {
      const p = clamp((now - started) / duration, 0, 1);
      element.volume = lerp(from, to, smoothstep(0, 1, p));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function playBedElement(element, url) {
    if (!element || !audioLayer.enabled) return;
    const play = () => element.play().catch((error) => console.warn(`Could not start bed audio: ${url}`, error));
    if (element.ended || (Number.isFinite(element.duration) && element.duration > 0 && element.currentTime >= element.duration - 0.12)) {
      element.currentTime = 0;
    }
    play();
  }

  function attachBedLoopGuard(element, url) {
    if (!element || element.dataset.quantizedLoopGuard === "true") return;
    element.dataset.quantizedLoopGuard = "true";
    element.addEventListener("ended", () => {
      if (!audioLayer.enabled) return;
      element.currentTime = 0;
      playBedElement(element, url);
    });
    element.addEventListener("timeupdate", () => {
      if (!audioLayer.enabled || !Number.isFinite(element.duration) || element.duration <= 0) return;
      if (element.currentTime < element.duration - 0.18) return;
      element.currentTime = 0;
      playBedElement(element, url);
    });
    window.setInterval(() => {
      if (!audioLayer.enabled || !audioLayer.bed || audioLayer.bed.element !== element) return;
      if (element.paused || element.ended) playBedElement(element, url);
    }, 1500);
  }

  function startBedForTheme(theme) {
    if (!audioLayer.ctx) return;
    const picked = chooseSample("beds", theme);
    if (!picked) return;
    const bedUrl = picked.url || "";
    if (audioLayer.bed && (audioLayer.bedTheme === theme || audioLayer.bed.url === bedUrl)) {
      if (audioLayer.bed.gain) audioLayer.bed.gain.gain.setTargetAtTime(audioVoice().bed, audioLayer.ctx.currentTime, 0.35);
      if (audioLayer.bed.direct) {
        if (audioLayer.bed.element.paused) playBedElement(audioLayer.bed.element, bedUrl);
        rampElementVolume(audioLayer.bed.element, bedElementVolume());
      }
      audioLayer.bedTheme = theme;
      return;
    }
    if (audioLayer.bed) {
      const old = audioLayer.bed;
      const now = audioLayer.ctx.currentTime;
      if (old.gain) old.gain.gain.setTargetAtTime(0.0001, now, 0.45);
      if (old.source && typeof old.source.stop === "function") old.source.stop(now + 1.8);
      if (old.element) {
        if (old.direct && old.element === bedAudio) {
          old.element.pause();
          old.element.volume = 0;
        } else {
          rampElementVolume(old.element, 0, 600);
          setTimeout(() => {
            old.element.pause();
            old.element.removeAttribute("src");
            old.element.load();
          }, 750);
        }
      }
      audioLayer.bed = null;
      audioLayer.bedTheme = "";
    }
    if (picked.media && window.location.protocol === "file:") {
      const element = bedAudio || new Audio();
      if (element.src !== new URL(picked.url, window.location.href).href) {
        element.src = picked.url;
        element.load();
      }
      element.loop = true;
      element.preload = "auto";
      element.volume = 0;
      attachBedLoopGuard(element, picked.url);
      playBedElement(element, picked.url);
      rampElementVolume(element, bedElementVolume(), 900);
      audioLayer.bed = { element, url: bedUrl, direct: true };
      audioLayer.bedTheme = theme;
      return;
    }
    const gain = audioLayer.ctx.createGain();
    let source = null;
    let element = null;
    if (picked.buffer) {
      source = audioLayer.ctx.createBufferSource();
      source.buffer = picked.buffer;
      source.loop = true;
      source.connect(gain);
      source.start();
    } else {
      element = new Audio(picked.url);
      element.loop = true;
      element.preload = "auto";
      element.volume = 1;
      source = audioLayer.ctx.createMediaElementSource(element);
      source.connect(gain);
      attachBedLoopGuard(element, picked.url);
      playBedElement(element, picked.url);
    }
    gain.connect(audioLayer.buses.bed.input);
    gain.gain.value = 0.0001;
    gain.gain.setTargetAtTime(audioVoice().bed, audioLayer.ctx.currentTime, 0.9);
    audioLayer.bed = { source, gain, element, url: bedUrl };
    audioLayer.bedTheme = theme;
  }

  function setAudioScene() {
    if (!audioLayer.ctx) return;
    const now = audioLayer.ctx.currentTime;
    const voice = audioVoice();
    const alg = audioAlgorithm();
    audioLayer.buses.bed.gain.gain.setTargetAtTime(0.82, now, 0.25);
    audioLayer.buses.ray.gain.gain.setTargetAtTime(voice.ray * (0.7 + controls.glint * 0.18), now, 0.18);
    audioLayer.buses.probe.gain.gain.setTargetAtTime(0.52 + controls.motion * 0.12, now, 0.18);
    audioLayer.buses.artifact.gain.gain.setTargetAtTime(0.46 + alg.grain * 0.12, now, 0.18);
    audioLayer.buses.sub.gain.gain.setTargetAtTime(0.3 + alg.sub * 0.26, now, 0.18);
    audioLayer.filters.bed.frequency.setTargetAtTime(voice.filter * (0.68 + controls.density * 0.16), now, 0.35);
    audioLayer.filters.ray.frequency.setTargetAtTime(voice.filter * (1.1 + controls.glint * 0.28), now, 0.22);
    audioLayer.filters.probe.frequency.setTargetAtTime(760 + controls.motion * 420 + alg.tempo * 180, now, 0.2);
    audioLayer.filters.artifact.frequency.setTargetAtTime(900 + alg.grain * 1250, now, 0.18);
    audioLayer.filters.sub.frequency.setTargetAtTime(95 + alg.sub * 95, now, 0.25);
    if (audioLayer.bed && audioLayer.bed.direct) rampElementVolume(audioLayer.bed.element, bedElementVolume(), 400);
    startBedForTheme(controls.theme);
  }

  function enableAudio() {
    audioLayer.enabled = true;
    if (!initAudio()) {
      audioLayer.enabled = false;
      updateAudioToggle();
      return false;
    }
    audioLayer.ctx.resume();
    audioLayer.master.gain.setTargetAtTime(0.94, audioLayer.ctx.currentTime, 0.08);
    setAudioScene();
    updateAudioToggle();
    return true;
  }

  function disableAudio() {
    if (!audioLayer.ctx) return;
    audioLayer.enabled = false;
    audioLayer.master.gain.setTargetAtTime(0.0001, audioLayer.ctx.currentTime, 0.12);
    if (audioLayer.bed && audioLayer.bed.direct) rampElementVolume(audioLayer.bed.element, 0, 250);
    updateAudioToggle();
  }

  function ensureAudioFromGesture() {
    if (!audioLayer.enabled) enableAudio();
    else if (audioLayer.ctx && audioLayer.ctx.state === "suspended") audioLayer.ctx.resume();
  }

  function audioPan(wx) {
    return clamp((wx / WORLD_W) * 2 - 1, -1, 1);
  }

  function cMajorFrequency(degree, root = 65.4064) {
    const scale = [0, 2, 4, 5, 7, 9, 11];
    const octave = Math.floor(degree / scale.length);
    const step = ((degree % scale.length) + scale.length) % scale.length;
    return root * Math.pow(2, (octave * 12 + scale[step]) / 12);
  }

  function noteFromWorld(wx, wy, offset = 0) {
    const alg = audioAlgorithm();
    const lane = Math.floor((wy / WORLD_H) * 8);
    const drift = Math.floor((wx / WORLD_W) * 7);
    const algDegree = Math.round(alg.pitch * 7 / 12);
    const offsetDegree = Math.round(offset * 7 / 12);
    let freq = cMajorFrequency(lane + drift + algDegree + offsetDegree);
    while (freq < 32) freq *= 2;
    while (freq > 1480) freq *= 0.5;
    return freq;
  }

  function noteInCMajor(wx, wy, offset = 0) {
    const lane = Math.floor((wy / WORLD_H) * 7);
    const drift = Math.floor((wx / WORLD_W) * 5);
    let freq = cMajorFrequency(lane + drift + offset, 130.8128);
    while (freq > 659.255) freq *= 0.5;
    while (freq < 196) freq *= 2;
    return freq;
  }

  function audioProbeMove(wx, wy, velocity, dragging) {
    if (!audioLayer.ctx || !audioLayer.enabled) return;
    const now = audioLayer.ctx.currentTime;
    const gap = dragging ? 0.075 / audioAlgorithm().tempo : 0.18 / audioAlgorithm().tempo;
    if (now - audioLayer.lastProbe < gap) return;
    audioLayer.lastProbe = now;
    const energy = clamp(velocity / (dragging ? 48 : 26), 0.08, 1.3);
    const pan = audioPan(wx);
    const rate = clamp(0.72 + energy * 0.44 + audioAlgorithm().grain * 0.08, 0.45, 1.8);
    if (!playSample("probe", { bus: "probe", gain: 0.12 + energy * 0.19, duration: 0.12 + energy * 0.12, rate, pan })) {
      playNoise("probe", 0.07 + energy * 0.12, 0.048 + energy * 0.12, {
        pan,
        frequency: 480 + energy * 1300 + (wy / WORLD_H) * 500,
        q: 6,
      });
    }
    if (dragging && energy > 0.45 && hash2(Math.floor(wx), Math.floor(wy), Math.floor(now * 20)) > 0.7) {
      playTone("probe", noteFromWorld(wx, wy, 12), 0.1, 0.055 + energy * 0.042, { pan, type: "triangle" });
    }
  }

  function audioClick(wx, wy, held) {
    if (!audioLayer.ctx || !audioLayer.enabled) return;
    const pan = audioPan(wx);
    const gainValue = held > 620 ? 0.33 : 0.23;
    if (!playSample("click", { bus: "artifact", gain: gainValue, duration: 0.42, rate: held > 620 ? 0.76 : 1.05, pan })) {
      playTone("artifact", noteFromWorld(wx, wy, held > 620 ? -12 : 0), held > 620 ? 0.5 : 0.18, gainValue * 0.42, {
        pan,
        type: "square",
      });
      playNoise("artifact", held > 620 ? 0.22 : 0.08, held > 620 ? 0.09 : 0.064, {
        pan,
        frequency: held > 620 ? 720 : 1800,
        q: held > 620 ? 2 : 9,
      });
    }
  }

  function audioRayHandle(index, wx, wy, velocity) {
    if (!audioLayer.ctx || !audioLayer.enabled) return;
    const now = audioLayer.ctx.currentTime;
    if (now - audioLayer.lastRayDrag < 0.09) return;
    audioLayer.lastRayDrag = now;
    const pan = audioPan(wx);
    const energy = clamp(velocity / 44, 0.12, 1.2);
    const freq = noteFromWorld(wx, wy, index * 5);
    if (!playSample("rayHandle", { bus: "ray", gain: 0.1 + energy * 0.12, duration: 0.32, rate: 0.82 + energy * 0.34, pan })) {
      playTone("ray", freq, 0.3, 0.09 + energy * 0.07, {
        pan,
        type: audioVoice().color,
        release: 0.12,
      });
    }
  }

  function audioScroll(wx, wy, direction) {
    if (!audioLayer.ctx || !audioLayer.enabled) return;
    const now = audioLayer.ctx.currentTime;
    if (now - audioLayer.lastScroll < 0.12) return;
    audioLayer.lastScroll = now;
    const pan = audioPan(wx);
    const rate = direction > 0 ? 1.12 : 0.82;
    if (!playSample("scroll", { bus: "sub", gain: 0.24, duration: 0.55, rate, pan })) {
      playTone("sub", noteFromWorld(wx, wy, -24), 0.38, 0.15, {
        pan,
        type: "sine",
        release: 0.16,
      });
    }
    audioLayer.filters.bed.frequency.setTargetAtTime(audioVoice().filter * (direction > 0 ? 1.22 : 0.72), now, 0.08);
    audioLayer.filters.bed.frequency.setTargetAtTime(audioVoice().filter, now + 0.18, 0.42);
  }

  function audioFossilCreate(wx, wy, strength) {
    if (!audioLayer.ctx || !audioLayer.enabled) return;
    const pan = audioPan(wx);
    if (!playSample("fossilCreate", { bus: "artifact", gain: 0.16 + strength * 0.09, duration: 0.36, rate: 0.8 + strength * 0.4, pan })) {
      playNoise("artifact", 0.12 + strength * 0.06, 0.075 + strength * 0.047, {
        pan,
        frequency: 1100 + strength * 1800,
        q: 10,
      });
    }
  }

  function audioFossilDisturb(wx, wy, amount) {
    if (!audioLayer.ctx || !audioLayer.enabled) return;
    const now = audioLayer.ctx.currentTime;
    if (now - audioLayer.lastFossil < 0.18) return;
    audioLayer.lastFossil = now;
    const pan = audioPan(wx);
    const strength = clamp(Math.abs(amount) * 20, 0.08, 0.8);
    if (!playSample("fossilDisturb", { bus: "artifact", gain: 0.06 + strength * 0.072, duration: 0.14, rate: 1.2 + strength, pan })) {
      playNoise("artifact", 0.06 + strength * 0.05, 0.03 + strength * 0.05, {
        pan,
        frequency: 1500 + strength * 2600,
        q: 14,
      });
    }
  }

  function audioRayGlint(familyIndex, rayIndex, glintWindow, brilliance, wx, wy) {
    if (!audioLayer.ctx || !audioLayer.enabled || brilliance < 0.26) return;
    const now = audioLayer.ctx.currentTime;
    if (now - audioLayer.lastGlint < 1.8) return;
    const key = `${familyIndex}:${rayIndex}:${glintWindow}`;
    if (audioLayer.glintKeys.has(key)) return;
    audioLayer.glintKeys.set(key, now);
    for (const [storedKey, time] of audioLayer.glintKeys) {
      if (now - time > 16) audioLayer.glintKeys.delete(storedKey);
    }
    audioLayer.lastGlint = now;
    rememberEvent("brilliance", wx, wy, 0.85 + brilliance * 0.55);
    recordDirectorEvent("ray_brilliance", wx, wy, { system: true, family: familyIndex, strength: brilliance });
    const pan = audioPan(wx);
    const samplePlayed = playSample("brilliance", { bus: "ray", gain: 0.1 + brilliance * 0.12, duration: 1.1, rate: 0.82 + brilliance * 0.32, pan, attack: 0.025, release: 0.34 });
    if (!samplePlayed) {
      const freq = noteInCMajor(wx, wy, 4 + familyIndex + (rayIndex % 5));
      playTone("ray", freq, 0.72, 0.04 + brilliance * 0.032, {
        pan,
        type: "sine",
        attack: 0.04,
        release: 0.28,
      });
      playTone("ray", freq * 2, 0.42, 0.01 + brilliance * 0.016, {
        pan,
        type: "triangle",
        delay: 0.04,
        attack: 0.03,
        release: 0.24,
      });
    }
    playBrillianceMelody(brilliance, pan);
  }

  function playBrillianceMelody(brilliance, pan) {
    const start = Math.floor(Math.random() * 8);
    for (let i = 0; i < 3; i += 1) {
      let freq = cMajorFrequency(start + Math.floor(Math.random() * 9) + i, 130.8128);
      while (freq > 587.33) freq *= 0.5;
      while (freq < 196) freq *= 2;
      playTone("ray", freq, 0.22 + i * 0.03, 0.018 + brilliance * 0.018, {
        pan: clamp(pan + (Math.random() - 0.5) * 0.22, -1, 1),
        type: i === 1 ? "triangle" : "sine",
        delay: 0.12 + i * 0.16,
        attack: 0.012,
        release: 0.16,
      });
    }
  }

  function stampField(field, wx, wy, radius, amount) {
    const cx = worldToSubX(wx);
    const cy = worldToSubY(wy);
    const r = Math.max(1, Math.ceil((radius / WORLD_W) * SUB_W));
    for (let y = cy - r; y <= cy + r; y += 1) {
      if (y < 0 || y >= SUB_H) continue;
      for (let x = cx - r; x <= cx + r; x += 1) {
        if (x < 0 || x >= SUB_W) continue;
        const dist = Math.hypot(x - cx, y - cy);
        if (dist > r) continue;
        const falloff = 1 - dist / r;
        field[y * SUB_W + x] += amount * falloff * falloff;
      }
    }
  }

  function measureAt(wx, wy, radius, amount) {
    const cx = worldToSubX(wx);
    const cy = worldToSubY(wy);
    const r = Math.max(1, Math.ceil((radius / WORLD_W) * SUB_W));
    const cell = Math.max(2, Math.round(r / 6));
    const salt = Math.floor(state.frame * 13 + wx * 0.17 + wy * 0.11);
    for (let y = cy - r; y <= cy + r; y += 1) {
      if (y < 0 || y >= SUB_H) continue;
      for (let x = cx - r; x <= cx + r; x += 1) {
        if (x < 0 || x >= SUB_W) continue;
        const dx = x - cx;
        const dy = y - cy;
        const dist = Math.hypot(dx, dy);
        if (dist > r) continue;
        const edge = smoothstep(1, 0, Math.abs(dist - r * 0.72) / Math.max(1, r * 0.16));
        const core = smoothstep(r, 0, dist);
        const qx = Math.floor(x / cell);
        const qy = Math.floor(y / cell);
        const jitter = 0.72 + hash2(qx, qy, salt) * 0.42;
        const value = amount * (core * 0.78 + edge * 0.36) * jitter;
        const idx = y * SUB_W + x;
        measurementField[idx] = Math.max(measurementField[idx], value);
      }
    }
  }

  function stampRayPerturbAt(wx, wy, radius, amount) {
    const cx = worldToSubX(wx);
    const cy = worldToSubY(wy);
    const r = Math.max(1, Math.ceil((radius / WORLD_W) * SUB_W));
    for (let y = cy - r; y <= cy + r; y += 1) {
      if (y < 0 || y >= SUB_H) continue;
      for (let x = cx - r; x <= cx + r; x += 1) {
        if (x < 0 || x >= SUB_W) continue;
        const dist = Math.hypot(x - cx, y - cy);
        if (dist > r) continue;
        const idx = y * SUB_W + x;
        const ray = clamp(rayField[idx], 0, 1.4);
        if (ray < 0.025) continue;
        const rayDepth = clamp(rayDepthField[idx] / Math.max(rayField[idx], 0.0001), 0, 1);
        const falloff = 1 - dist / r;
        const stain = amount * falloff * falloff * (0.18 + ray * 0.72) * (0.72 + rayDepth * 0.34);
        rayPerturbColorField[idx] = Math.max(rayPerturbColorField[idx], stain);
      }
    }
  }

  function stampProbeLine(x0, y0, x1, y1, amount) {
    const distance = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(2, Math.ceil(distance / 22));
    const sweep = clamp(distance / 140, 0.8, 3.2);
    for (let i = 0; i <= steps; i += 1) {
      const p = i / steps;
      const x = lerp(x0, x1, p);
      const y = lerp(y0, y1, p);
      stampField(probeField, x, y, 76, amount * 1.45);
      stampField(dragField, x, y, 112, amount * 1.8);
      stampRayPerturbAt(x, y, 118, amount * 3.9 * sweep);
    }
  }

  function rememberEvent(kind, wx, wy, strength = 1) {
    const observer = observerDelta();
    const rayBias = kind === "ray" || kind === "brilliance" ? observer.ray * 0.38 : 0;
    const attentionBias = observer.attention * 0.22 + observer.pressure * 0.1;
    eventMemories.push({
      kind,
      x: clamp(wx, 0, WORLD_W),
      y: clamp(wy, 0, WORLD_H),
      strength: clamp(strength * (1 + attentionBias + rayBias), 0.25, 1.8),
      age: 0,
      life: lerp(1600, 4300, rand()),
      nextEcho: lerp(18, 75, rand()) * (1 - observer.attention * 0.16),
      echoes: 0,
      salt: Math.floor(rand() * 999999),
    });
    if (eventMemories.length > 28) eventMemories.shift();
  }

  function observerDelta() {
    return {
      stillness: clamp((state.observerStillness - 0.4) / 0.6, 0, 1),
      agitation: clamp(state.observerAgitation, 0, 1),
      attention: clamp(state.observerAttention, 0, 1),
      ray: clamp(state.observerRayAffinity, 0, 1),
      pressure: clamp(state.observerPressure, 0, 1),
    };
  }

  function observe(kind, amount = 1) {
    if (kind !== "still") state.observerEngaged = true;
    if (kind === "move") {
      state.observerAgitation = clamp(state.observerAgitation + amount * 0.012, 0, 1);
      state.observerStillness = clamp(state.observerStillness - amount * 0.01, 0, 1);
      state.observerLastMove = state.lastTime;
    } else if (kind === "still") {
      state.observerStillness = clamp(state.observerStillness + amount * 0.018, 0, 1);
      state.observerAgitation = clamp(state.observerAgitation - amount * 0.012, 0, 1);
    } else if (kind === "click") {
      state.observerAttention = clamp(state.observerAttention + amount * 0.16, 0, 1);
      state.observerAgitation = clamp(state.observerAgitation + amount * 0.05, 0, 1);
    } else if (kind === "ray") {
      state.observerRayAffinity = clamp(state.observerRayAffinity + amount * 0.18, 0, 1);
      state.observerAttention = clamp(state.observerAttention + amount * 0.08, 0, 1);
    } else if (kind === "scroll") {
      state.observerPressure = clamp(state.observerPressure + amount * 0.14, 0, 1);
      state.observerAgitation = clamp(state.observerAgitation + amount * 0.04, 0, 1);
    }
  }

  function updateObserver(dt, t) {
    const sinceMove = state.lastTime - state.observerLastMove;
    if (state.observerEngaged && !state.pointerDown && sinceMove > 1600) observe("still", dt * 1.9);
    state.observerAgitation *= Math.pow(0.84, dt);
    state.observerAttention *= Math.pow(0.988, dt);
    state.observerRayAffinity *= Math.pow(0.994, dt);
    state.observerPressure *= Math.pow(0.976, dt);
    state.observerStillness = clamp(state.observerStillness + Math.sin(t * 0.011) * 0.0004, 0, 1);
    updateObserverAudio();
  }

  function updateObserverAudio() {
    if (!audioLayer.ctx || !audioLayer.enabled || !audioLayer.buses.ray) return;
    const observer = observerDelta();
    const now = audioLayer.ctx.currentTime;
    const voice = audioVoice();
    const alg = audioAlgorithm();
    audioLayer.buses.ray.gain.gain.setTargetAtTime(voice.ray * (0.7 + controls.glint * 0.18) * (1 + observer.ray * 0.18), now, 0.24);
    audioLayer.buses.probe.gain.gain.setTargetAtTime(0.52 + controls.motion * 0.12 + observer.agitation * 0.08, now, 0.24);
    audioLayer.buses.artifact.gain.gain.setTargetAtTime(0.46 + alg.grain * 0.12 + observer.attention * 0.07, now, 0.28);
    audioLayer.filters.bed.frequency.setTargetAtTime(voice.filter * (0.68 + controls.density * 0.16 + observer.stillness * 0.08), now, 0.45);
    audioLayer.filters.ray.frequency.setTargetAtTime(voice.filter * (1.1 + controls.glint * 0.28 + observer.ray * 0.22), now, 0.28);
  }

  function updateEventMemories(dt, t) {
    const observer = observerDelta();
    for (let i = eventMemories.length - 1; i >= 0; i -= 1) {
      const memory = eventMemories[i];
      memory.age += dt;
      memory.nextEcho -= dt;
      memory.strength *= 0.9997;
      if (memory.age > memory.life || memory.strength < 0.12) {
        eventMemories.splice(i, 1);
        continue;
      }
      if (memory.nextEcho <= 0) {
        const maturity = clamp(memory.age / 42, 0.18, 1);
        const echoThreshold = clamp(0.45 - observer.attention * 0.08 - observer.stillness * 0.04, 0.34, 0.48);
        const rarityGate = hash2(memory.salt, memory.echoes, Math.floor(t * 0.11)) > echoThreshold;
        if (rarityGate) echoEventMemory(memory, maturity);
        memory.echoes += 1;
        memory.nextEcho = lerp(42, 125, rand()) * (1 + memory.echoes * 0.18) * (1 - observer.attention * 0.08);
      }
    }
  }

  function beginGlobalReconfiguration() {
    const cx = WORLD_W * 0.5;
    const cy = WORLD_H * 0.5;
    const axis = normalize3(lerp(-1, 1, rand()), lerp(-1, 1, rand()), lerp(-1, 1, rand()));
    const angle = lerp(Math.PI * 0.55, Math.PI * 1.35, rand()) * (rand() > 0.5 ? 1 : -1);
    const tilt = lerp(-0.75, 0.75, rand());
    const scale = lerp(0.82, 1.18, rand());
    globalReconfiguration.active = true;
    globalReconfiguration.elapsed = 0;
    globalReconfiguration.duration = lerp(18, 34, rand());
    globalReconfiguration.cooldown = lerp(150, 260, rand());
    globalReconfiguration.salt = Math.floor(rand() * 999999);
    globalReconfiguration.from = rayFamilies.map((family) => ({
      x: family.globalOffsetX || 0,
      y: family.globalOffsetY || 0,
    }));
    globalReconfiguration.to = rayFamilies.map((family, i) => {
      const px = ((family.baseOriginX || family.originX) - cx) / (WORLD_W * 0.36);
      const py = ((family.baseOriginY || family.originY) - cy) / (WORLD_H * 0.36);
      const pz = (hash2(i, globalReconfiguration.salt, 719) - 0.5) * 1.8 + tilt;
      const rotated = rotate3({ x: px, y: py, z: pz }, axis, angle);
      const depth = 1 / (1.18 + rotated.z * 0.24);
      const targetX = clamp(cx + rotated.x * WORLD_W * 0.36 * scale * depth, WORLD_W * 0.08, WORLD_W * 0.92);
      const targetY = clamp(cy + rotated.y * WORLD_H * 0.36 * scale * depth, WORLD_H * 0.08, WORLD_H * 0.92);
      return {
        x: targetX - (family.baseOriginX || family.originX),
        y: targetY - (family.baseOriginY || family.originY),
      };
    });
    stampField(collapseField, cx, cy, 420, 0.22);
    measureAt(cx, cy, 520, 0.28);
  }

  function updateGlobalReconfiguration(dt) {
    if (!globalReconfiguration.active) {
      const observer = observerDelta();
      globalReconfiguration.cooldown -= dt;
      const threshold = clamp(0.995 - observer.stillness * 0.004 - observer.attention * 0.002, 0.988, 0.997);
      if (globalReconfiguration.cooldown <= 0 && eventMemories.length > 2 && hash2(state.frame, eventMemories.length, 9051) > threshold) {
        beginGlobalReconfiguration();
      }
      return;
    }
    globalReconfiguration.elapsed += dt;
    const p = clamp(globalReconfiguration.elapsed / globalReconfiguration.duration, 0, 1);
    const e = easeInOut(p);
    for (let i = 0; i < rayFamilies.length; i += 1) {
      const family = rayFamilies[i];
      const from = globalReconfiguration.from[i] || { x: 0, y: 0 };
      const to = globalReconfiguration.to[i] || from;
      family.globalOffsetX = lerp(from.x, to.x, e);
      family.globalOffsetY = lerp(from.y, to.y, e);
    }
    state.perturb = Math.min(2.3, state.perturb + Math.sin(p * Math.PI) * 0.01);
    if (p >= 1) {
      globalReconfiguration.active = false;
    }
  }

  function echoEventMemory(memory, maturity) {
    const observer = observerDelta();
    const wobble = memory.echoes + 1;
    const ox = (hash2(memory.salt, wobble, 101) - 0.5) * 150 * maturity;
    const oy = (hash2(memory.salt, wobble, 202) - 0.5) * 150 * maturity;
    const x = clamp(memory.x + ox, 0, WORLD_W);
    const y = clamp(memory.y + oy, 0, WORLD_H);
    const force = memory.strength * maturity * (1 + observer.attention * 0.18 + observer.pressure * 0.14);
    stampField(collapseField, x, y, 135 + force * 80, 0.36 + force * 0.32);
    stampField(probeField, x, y, 180 + force * 90, 0.08 + force * 0.08);
    measureAt(x, y, 170 + force * 120, 0.48 + force * 0.34);
    stampRayPerturbAt(x, y, 170 + force * 90, 0.44 + force * 0.42);
    state.perturb = Math.min(2.1, state.perturb + 0.22 + force * 0.12);
    if ((memory.kind === "long" || memory.kind === "brilliance") && hash2(memory.salt, memory.echoes, 303) > 0.58) {
      const family = makeTemporaryRayFamily(x, y);
      family.decay = 0.58 + force * 0.18;
      family.speed *= 0.55;
      temporaryRayFamilies.push(family);
    }
  }

  function decayInteractionFields() {
    for (let i = 0; i < probeField.length; i += 1) {
      probeField[i] *= 0.925;
      dragField[i] *= 0.952;
      collapseField[i] *= 0.955;
      measurementField[i] *= 0.976;
      rayPerturbColorField[i] *= 0.968;
    }
    for (let i = fossils.length - 1; i >= 0; i -= 1) {
      fossils[i].life -= 0.0012;
      if (fossils[i].life <= 0) fossils.splice(i, 1);
    }
    for (let i = temporaryRayFamilies.length - 1; i >= 0; i -= 1) {
      if (temporaryRayFamilies[i].directorDuration) {
        temporaryRayFamilies[i].directorAge = (temporaryRayFamilies[i].directorAge || 0) + 0.016;
        if (temporaryRayFamilies[i].directorAge > temporaryRayFamilies[i].directorDuration) temporaryRayFamilies[i].decay *= 0.96;
      }
      temporaryRayFamilies[i].decay *= 0.992;
      if (temporaryRayFamilies[i].decay < 0.025) temporaryRayFamilies.splice(i, 1);
    }
  }

  function disturbFossils(wx, wy, radius, amount) {
    let disturbed = false;
    for (const fossil of fossils) {
      const d = Math.hypot(fossil.x - wx, fossil.y - wy);
      if (d > radius + fossil.radius) continue;
      const hit = 1 - clamp(d / (radius + fossil.radius), 0, 1);
      fossil.life = clamp(fossil.life - amount * hit, 0, 1.8);
      fossil.phase += amount * hit * 9;
      fossil.scale = clamp(fossil.scale + amount * hit * 0.4, 0.45, 3.4);
      if (hit > 0.12) disturbed = true;
    }
    if (disturbed) audioFossilDisturb(wx, wy, amount);
  }

  function createFossil(wx, wy, strength) {
    fossils.push({
      x: clamp(wx, 0, WORLD_W),
      y: clamp(wy, 0, WORLD_H),
      radius: lerp(28, 86, hash2(Math.floor(wx), Math.floor(wy), fossils.length + 91)) * (0.7 + strength),
      cell: [3, 4, 5, 6, 8][fossils.length % 5],
      phase: rand() * TAU,
      life: clamp(0.65 + strength * 0.55, 0.3, 1.4),
      scale: 1,
      salt: Math.floor(rand() * 99999),
      paletteShift: Math.floor(rand() * 8),
      glitch: lerp(0.6, 1.4, rand()),
    });
    if (fossils.length > 34) fossils.shift();
    audioFossilCreate(wx, wy, strength);
  }

  function directorNow() {
    return state.lastTime || performance.now();
  }

  function directorJitter(ms) {
    return ms * lerp(0.8, 1.2, rand());
  }

  function clampDirectorX(value) {
    return clamp(Number(value) || 0, 0, WORLD_W);
  }

  function clampDirectorY(value) {
    return clamp(Number(value) || 0, 0, WORLD_H);
  }

  function clampDirectorDuration(value, fallback = 1200, max = 30000) {
    return clamp(Number(value) || fallback, 60, max);
  }

  function setControlValue(param, value) {
    if (!(param in controls)) return false;
    controls[param] = value;
    const input = document.getElementById(param);
    if (input) input.value = String(value);
    setAudioScene();
    return true;
  }

  function recordDirectorEvent(kind, wx = state.pointerX, wy = state.pointerY, meta = {}) {
    if (!directorState.enabled) return;
    const now = directorNow();
    const x = clampDirectorX(wx);
    const y = clampDirectorY(wy);
    const system = !!meta.system;
    if (!system) {
      directorState.lastInput = now;
      for (const action of directorState.recentAi) {
        if (!action.reacted && now - action.t <= 10000) action.reacted = true;
      }
    }
    const hx = clamp(Math.floor((x / WORLD_W) * DIRECTOR_HEAT_W), 0, DIRECTOR_HEAT_W - 1);
    const hy = clamp(Math.floor((y / WORLD_H) * DIRECTOR_HEAT_H), 0, DIRECTOR_HEAT_H - 1);
    directorState.heatmap[hy * DIRECTOR_HEAT_W + hx] += 1;
    if (!system) {
      if (kind === "click") directorState.counts.clicks += 1;
      else if (kind === "long_press") directorState.counts.longPresses += 1;
      else if (kind === "drag") directorState.counts.drags += 1;
      else if (kind === "ray_origin_drag") directorState.counts.rayOriginDrags += 1;
      else if (kind === "wheel") directorState.counts.wheel += 1;
      else if (kind === "key") directorState.counts.keys += 1;
    }
    if (kind === "drag" || kind === "ray_origin_drag" || kind === "hover") {
      const last = directorState.recentViewer[directorState.recentViewer.length - 1];
      if (last && last.kind === kind && now - last.t < 650) {
        last.x = Math.round(x);
        last.y = Math.round(y);
        last.count = (last.count || 1) + 1;
        return;
      }
    }
    directorState.recentViewer.push({
      t: Math.round(now - directorState.sessionStart),
      kind,
      x: Math.round(x),
      y: Math.round(y),
      meta: compactDirectorMeta(meta),
    });
    while (directorState.recentViewer.length > 24) directorState.recentViewer.shift();
  }

  function compactDirectorMeta(meta) {
    const out = {};
    for (const key of ["gesture", "family", "direction", "held", "tool", "strength"]) {
      if (meta[key] !== undefined) out[key] = meta[key];
    }
    return out;
  }

  function buildDirectorTelemetry(now) {
    const heatmap = [];
    for (let y = 0; y < DIRECTOR_HEAT_H; y += 1) {
      const row = [];
      for (let x = 0; x < DIRECTOR_HEAT_W; x += 1) row.push(directorState.heatmap[y * DIRECTOR_HEAT_W + x]);
      heatmap.push(row);
    }
    return {
      sessionAgeMs: Math.round(now - directorState.sessionStart),
      sinceLastCallMs: directorState.lastCall ? Math.round(now - directorState.lastCall) : null,
      stillnessMs: Math.round(now - directorState.lastInput),
      current: {
        theme: controls.theme,
        algorithm: controls.algorithm,
        controls: { ...controls },
        audioEnabled: audioLayer.enabled,
      },
      observer: observerDelta(),
      counts: { ...directorState.counts },
      heatmap,
      recentViewer: directorState.recentViewer.slice(-24),
      recentAi: directorState.recentAi.slice(-24).map((action) => ({
        t: Math.round(action.t - directorState.sessionStart),
        tool: action.tool,
        x: action.x,
        y: action.y,
        reacted: !!action.reacted,
      })),
      memoryNote: directorState.memoryNote,
      carry: directorState.carry,
    };
  }

  function resetDirectorWindow() {
    directorState.recentViewer = [];
    directorState.heatmap.fill(0);
    directorState.counts = {
      clicks: 0,
      longPresses: 0,
      drags: 0,
      rayOriginDrags: 0,
      wheel: 0,
      keys: 0,
      idlePeriods: 0,
    };
  }

  function logDirector(label, payload) {
    if (!DIRECTOR_LOGGING) return;
    console.log(`[Quantized director] ${label}`, payload);
  }

  function nextDirectorCadence(now, overrideMs) {
    const cadence = [10000, 30000, 60000, 120000, 180000, 240000];
    const base =
      overrideMs === undefined || overrideMs === null
        ? cadence[Math.min(directorState.cadenceIndex, cadence.length - 1)]
        : clamp(Number(overrideMs) || cadence[0], DIRECTOR_FOLLOWUP_MIN_MS, DIRECTOR_FOLLOWUP_MAX_MS);
    if (overrideMs === undefined || overrideMs === null) directorState.cadenceIndex += 1;
    directorState.nextDue = now + directorJitter(base);
  }

  async function requestDirectorScore(now) {
    directorState.pending = true;
    directorState.lastCall = now;
    const telemetry = buildDirectorTelemetry(now);
    logDirector("request", {
      endpoint: DIRECTOR_ENDPOINT,
      sessionAgeMs: telemetry.sessionAgeMs,
      stillnessMs: telemetry.stillnessMs,
      counts: telemetry.counts,
      recentViewer: telemetry.recentViewer,
      recentAi: telemetry.recentAi,
      memoryNote: telemetry.memoryNote,
      carry: telemetry.carry,
    });
    try {
      const response = await fetch(DIRECTOR_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ telemetry }),
      });
      if (!response.ok) throw new Error(`director ${response.status}`);
      const raw = await response.json();
      logDirector("response raw", raw);
      const score = sanitizeDirectorScore(raw);
      logDirector("response sanitized", score);
      if (score) executeDirectorScore(score);
      directorState.memoryNote = typeof raw.memoryNote === "string" ? raw.memoryNote.slice(0, 480) : "";
      directorState.carry = raw.nextCall && typeof raw.nextCall.carry === "string" ? raw.nextCall.carry.slice(0, 600) : "";
      nextDirectorCadence(directorNow(), raw.nextCall ? raw.nextCall.afterMs : undefined);
      resetDirectorWindow();
      directorState.failed = false;
    } catch (error) {
      directorState.failed = true;
      directorState.nextDue = directorNow() + 60000;
      console.warn("Quantized director unavailable", error);
    } finally {
      directorState.pending = false;
    }
  }

  function updateDirector(dt, t) {
    if (!directorState.enabled) return;
    const now = directorNow();
    updateDirectorEffects(dt, t);
    if (directorState.themeCooldown > 0) directorState.themeCooldown -= dt;
    if (directorState.algorithmCooldown > 0) directorState.algorithmCooldown -= dt;
    const still = !state.pointerDown && now - directorState.lastInput >= DIRECTOR_MIN_STILL_MS;
    const idleBucket = Math.floor((now - directorState.sessionStart) / 17000);
    if (still && idleBucket !== directorState.lastIdleBucket) {
      directorState.lastIdleBucket = idleBucket;
      directorState.counts.idlePeriods += 1;
    }
    if (directorState.pending || document.hidden || !still || now < directorState.nextDue) return;
    requestDirectorScore(now);
  }

  function sanitizeDirectorScore(raw) {
    if (!raw || !Array.isArray(raw.actions)) return null;
    return {
      intent: typeof raw.intent === "string" ? raw.intent.slice(0, 80) : "unlabeled",
      actions: raw.actions.slice(0, DIRECTOR_MAX_ACTIONS).map(sanitizeDirectorAction).filter(Boolean),
      nextCall: raw.nextCall || null,
      memoryNote: typeof raw.memoryNote === "string" ? raw.memoryNote.slice(0, 480) : "",
    };
  }

  function sanitizeDirectorAction(action) {
    if (!action || typeof action.tool !== "string" || !directorActionHandlers[action.tool]) return null;
    const args = action.args && typeof action.args === "object" ? action.args : {};
    return {
      atMs: clamp(Number(action.atMs) || 0, 0, 30000),
      tool: action.tool,
      args,
      cancelOnInteraction: !!action.cancelOnInteraction,
    };
  }

  function executeDirectorScore(score) {
    const batchStart = directorNow();
    logDirector("execute score", score);
    for (const action of score.actions) {
      window.setTimeout(() => {
        if (action.cancelOnInteraction && directorNow() - directorState.lastInput < DIRECTOR_MIN_STILL_MS) return;
        const handler = directorActionHandlers[action.tool];
        if (!handler) return;
        logDirector("execute action", action);
        handler(action.args);
        directorState.recentAi.push({
          t: directorNow(),
          tool: action.tool,
          x: Math.round(action.args.x === undefined ? state.pointerX : clampDirectorX(action.args.x)),
          y: Math.round(action.args.y === undefined ? state.pointerY : clampDirectorY(action.args.y)),
          reacted: false,
        });
        while (directorState.recentAi.length > 24) directorState.recentAi.shift();
        recordDirectorEvent(
          "ai_action",
          action.args.x === undefined ? state.pointerX : action.args.x,
          action.args.y === undefined ? state.pointerY : action.args.y,
          { system: true, tool: action.tool },
        );
      }, Math.max(0, batchStart + action.atMs - directorNow()));
    }
  }

  function addDirectorEffect(effect) {
    directorEffects.push({ ...effect, age: 0 });
    if (directorEffects.length > 42) directorEffects.shift();
  }

  function updateDirectorEffects(dt, t) {
    for (let i = directorTweens.length - 1; i >= 0; i -= 1) {
      const tween = directorTweens[i];
      tween.age += dt * 1000;
      const p = clamp(tween.age / tween.durationMs, 0, 1);
      const value = lerp(tween.from, tween.to, easeInOut(p));
      setControlValue(tween.param, tween.integer ? Math.round(value) : value);
      if (p >= 1) directorTweens.splice(i, 1);
    }
    for (let i = directorEffects.length - 1; i >= 0; i -= 1) {
      const effect = directorEffects[i];
      effect.age += dt * 1000;
      if (effect.age >= effect.durationMs) {
        directorEffects.splice(i, 1);
        continue;
      }
      if (effect.type === "substrate_burst") {
        const p = 1 - effect.age / effect.durationMs;
        stampField(probeField, effect.x, effect.y, effect.radius, effect.amount * p * 0.12);
      }
    }
  }

  function directorEnvelope(effect) {
    return Math.sin(clamp(effect.age / effect.durationMs, 0, 1) * Math.PI);
  }

  function directorFamilyMatches(effect, index) {
    return effect.family === "all" || Number(effect.family) === index;
  }

  function directorRayNudge(index, t) {
    const out = { x: 0, y: 0, z: 0, projection: 0 };
    for (const effect of directorEffects) {
      if (effect.type !== "bend_ray_family" || !directorFamilyMatches(effect, index)) continue;
      const e = directorEnvelope(effect);
      out.x += effect.axisDx * e;
      out.y += effect.axisDy * e;
      out.z += effect.axisDz * e;
      out.projection += effect.projection * e;
    }
    return out;
  }

  function directorRayBrilliance(index) {
    let boost = 0;
    for (const effect of directorEffects) {
      if (effect.type !== "force_ray_brilliance" || !directorFamilyMatches(effect, index)) continue;
      boost += effect.strength * directorEnvelope(effect);
    }
    return boost;
  }

  function stampDirectorRayEffects(t) {
    for (const effect of directorEffects) {
      const e = directorEnvelope(effect);
      if (effect.type === "pulse_origins") {
        for (let i = 0; i < rayFamilies.length; i += 1) {
          if (!directorFamilyMatches(effect, i)) continue;
          const origin = familyOrigin(rayFamilies[i]);
          stampRayField(
            Math.round((origin.x / WORLD_W) * SUB_W),
            Math.round((origin.y / WORLD_H) * SUB_H),
            Math.max(2, (effect.radius / WORLD_W) * SUB_W),
            effect.amount * e,
            1,
            effect.amount * e,
          );
        }
      } else if (effect.type === "ray_glitch_patch") {
        const sx = worldToSubX(effect.x);
        const sy = worldToSubY(effect.y);
        const r = Math.max(1, Math.round((effect.radius / WORLD_W) * SUB_W));
        for (let y = sy - r; y <= sy + r; y += 2) {
          if (y < 0 || y >= SUB_H) continue;
          for (let x = sx - r; x <= sx + r; x += 2) {
            if (x < 0 || x >= SUB_W) continue;
            const d = Math.hypot(x - sx, y - sy);
            if (d > r) continue;
            const gate = Math.sin((x - sx) * 0.7 + t * 9.2 + effect.salt) + Math.sin((y - sy) * 1.1 - t * 5.8);
            if (gate < 0.35) continue;
            stampRayField(x, y, 1.2, effect.strength * e * (1 - d / r) * 0.24, 0.8, effect.strength * e * 0.5);
          }
        }
      }
    }
  }

  function directorPixelMod(x, y, t, shape, pixelEffects) {
    let next = shape;
    let glint = 0;
    for (const effect of pixelEffects) {
      const e = directorEnvelope(effect);
      if (e <= 0.001) continue;
      if (effect.type === "quantize_patch" || effect.type === "tile_pattern" || effect.type === "bresenham_circle") {
        const d = Math.hypot(x - (effect.x / WORLD_W) * SUB_W, y - (effect.y / WORLD_H) * SUB_H);
        const r = Math.max(1, (effect.radius / WORLD_W) * SUB_W);
        if (d > r) continue;
        const falloff = (1 - d / r) * e;
        if (effect.type === "quantize_patch") {
          const levels = 3 + Math.floor(effect.amount * 10);
          next = lerp(next, Math.floor(next * levels) / levels, clamp(falloff * effect.amount, 0, 0.9));
          glint += falloff * 0.18;
        } else if (effect.type === "tile_pattern") {
          const cell = Math.max(1, Math.round(effect.cell * SUB_W / WORLD_W));
          const tile = hash2(Math.floor(x / cell), Math.floor(y / cell), effect.salt);
          next = lerp(next, tile, clamp(falloff * effect.amount * 0.55, 0, 0.7));
          glint += falloff * 0.12;
        } else {
          const ring = Math.abs(Math.sin((d - effect.age * 0.018) * 0.9));
          next += smoothstep(0.12, 0, ring) * falloff * effect.amount * 0.28;
          glint += smoothstep(0.18, 0, ring) * falloff * 0.28;
        }
      } else if (effect.type === "scanline_tear") {
        const sx = (effect.x / WORLD_W) * SUB_W;
        const sy = (effect.y / WORLD_H) * SUB_H;
        const sw = (effect.width / WORLD_W) * SUB_W;
        const sh = (effect.height / WORLD_H) * SUB_H;
        if (Math.abs(x - sx) > sw * 0.5 || Math.abs(y - sy) > sh * 0.5) continue;
        const lane = effect.horizontal ? y : x;
        const tear = ((Math.floor(lane / 2) + Math.floor(effect.age * 0.02)) & 7) / 7;
        next = lerp(next, tear, clamp(effect.amount * e * 0.42, 0, 0.62));
        glint += effect.amount * e * 0.12;
      }
    }
    return { shape: next, glint };
  }

  const directorActionHandlers = {
    pulse_origins(args) {
      addDirectorEffect({
        type: "pulse_origins",
        family: args.family === "all" ? "all" : Math.floor(Number(args.family) || 0),
        radius: clamp(Number(args.radius) || 72, 12, 260),
        amount: clamp(Number(args.amount) || 0.5, 0.05, 1.8),
        durationMs: clampDirectorDuration(args.durationMs, 1600, 12000),
      });
    },
    force_ray_brilliance(args) {
      addDirectorEffect({
        type: "force_ray_brilliance",
        family: args.family === "all" ? "all" : Math.floor(Number(args.family) || 0),
        strength: clamp(Number(args.strength) || 0.5, 0.05, 1.6),
        durationMs: clampDirectorDuration(args.durationMs, 1800, 16000),
      });
    },
    bend_ray_family(args) {
      addDirectorEffect({
        type: "bend_ray_family",
        family: args.family === "all" ? "all" : Math.floor(Number(args.family) || 0),
        axisDx: clamp(Number(args.axisDx) || 0, -0.5, 0.5),
        axisDy: clamp(Number(args.axisDy) || 0, -0.5, 0.5),
        axisDz: clamp(Number(args.axisDz) || 0, -0.5, 0.5),
        projection: clamp(Number(args.projection) || 0, -0.22, 0.28),
        durationMs: clampDirectorDuration(args.durationMs, 3200, 30000),
      });
    },
    spawn_temporary_rays(args) {
      const family = makeTemporaryRayFamily(clampDirectorX(args.x), clampDirectorY(args.y));
      family.decay = clamp(Number(args.strength) || 0.75, 0.15, 1.4);
      family.directorAge = 0;
      family.directorDuration = clampDirectorDuration(args.durationMs, 9000, 30000) / 1000;
      temporaryRayFamilies.push(family);
    },
    stain_rays(args) {
      stampRayPerturbAt(clampDirectorX(args.x), clampDirectorY(args.y), clamp(Number(args.radius) || 150, 18, 520), clamp(Number(args.amount) || 0.5, 0.05, 1.8));
    },
    ray_glitch_patch(args) {
      addDirectorEffect({
        type: "ray_glitch_patch",
        x: clampDirectorX(args.x),
        y: clampDirectorY(args.y),
        radius: clamp(Number(args.radius) || 170, 24, 560),
        strength: clamp(Number(args.strength) || 0.55, 0.04, 1.3),
        durationMs: clampDirectorDuration(args.durationMs, 2600, 22000),
        salt: Math.floor(rand() * 99999),
      });
    },
    measure_region(args) {
      measureAt(clampDirectorX(args.x), clampDirectorY(args.y), clamp(Number(args.radius) || 160, 20, 620), clamp(Number(args.amount) || 0.5, 0.03, 1.2));
    },
    collapse_region(args) {
      stampField(collapseField, clampDirectorX(args.x), clampDirectorY(args.y), clamp(Number(args.radius) || 130, 18, 560), clamp(Number(args.amount) || 0.42, 0.03, 1.4));
    },
    probe_line(args) {
      stampProbeLine(clampDirectorX(args.x0), clampDirectorY(args.y0), clampDirectorX(args.x1), clampDirectorY(args.y1), clamp(Number(args.amount) || 0.12, 0.02, 0.7));
    },
    substrate_burst(args) {
      addDirectorEffect({
        type: "substrate_burst",
        x: clampDirectorX(args.x),
        y: clampDirectorY(args.y),
        radius: clamp(Number(args.radius) || 160, 24, 620),
        amount: clamp(Number(args.amount) || 0.4, 0.03, 1.2),
        durationMs: clampDirectorDuration(args.durationMs, 1200, 12000),
      });
    },
    quantize_patch(args) {
      addDirectorEffect({
        type: "quantize_patch",
        x: clampDirectorX(args.x),
        y: clampDirectorY(args.y),
        radius: clamp(Number(args.radius) || 180, 24, 620),
        amount: clamp(Number(args.amount) || 0.6, 0.05, 1.4),
        durationMs: clampDirectorDuration(args.durationMs, 2200, 24000),
      });
    },
    scanline_tear(args) {
      addDirectorEffect({
        type: "scanline_tear",
        x: clampDirectorX(args.x),
        y: clampDirectorY(args.y),
        width: clamp(Number(args.width) || 420, 32, WORLD_W),
        height: clamp(Number(args.height) || 160, 16, WORLD_H),
        horizontal: args.horizontal !== false,
        amount: clamp(Number(args.amount) || 0.5, 0.04, 1.2),
        durationMs: clampDirectorDuration(args.durationMs, 1400, 16000),
      });
    },
    tile_pattern(args) {
      addDirectorEffect({
        type: "tile_pattern",
        x: clampDirectorX(args.x),
        y: clampDirectorY(args.y),
        radius: clamp(Number(args.radius) || 190, 24, 620),
        cell: clamp(Number(args.cell) || 28, 4, 128),
        amount: clamp(Number(args.amount) || 0.5, 0.04, 1.2),
        durationMs: clampDirectorDuration(args.durationMs, 2200, 22000),
        salt: Math.floor(rand() * 99999),
      });
    },
    bresenham_circle(args) {
      addDirectorEffect({
        type: "bresenham_circle",
        x: clampDirectorX(args.x),
        y: clampDirectorY(args.y),
        radius: clamp(Number(args.radius) || 160, 18, 620),
        amount: clamp(Number(args.amount) || 0.6, 0.04, 1.4),
        durationMs: clampDirectorDuration(args.durationMs, 2600, 22000),
      });
    },
    lerp_param(args) {
      const allowed = ["blur", "rayBlur", "density", "glint", "raySpeed", "blocks"];
      const param = String(args.param || "");
      if (!allowed.includes(param)) return;
      const maxValues = { blur: 28, rayBlur: 18, density: 3, glint: 3, raySpeed: 3, blocks: 1 };
      directorTweens.push({
        param,
        from: Number(controls[param]) || 0,
        to: clamp(Number(args.to) || 0, 0, maxValues[param]),
        durationMs: clampDirectorDuration(args.durationMs, 2200, 30000),
        age: 0,
        integer: param === "blur" || param === "rayBlur",
      });
    },
    set_theme(args) {
      const theme = String(args.theme || "");
      if (!themes[theme] || directorState.themeCooldown > 0) return;
      setControlValue("theme", theme);
      directorState.themeCooldown = 180;
    },
    set_algorithm(args) {
      const algorithm = String(args.algorithm || "");
      if (!algorithms[algorithm] || directorState.algorithmCooldown > 0) return;
      setControlValue("algorithm", algorithm);
      directorState.algorithmCooldown = 180;
    },
    global_reconfiguration(args) {
      if (globalReconfiguration.active) return;
      beginGlobalReconfiguration();
      state.perturb = Math.min(2.3, state.perturb + clamp(Number(args.strength) || 0.4, 0.05, 1.1));
    },
    play_c_tone(args) {
      playTone("ray", cMajorFrequency(Math.floor(Number(args.degree) || 0), 130.8128), clampDirectorDuration(args.durationMs, 260, 2200) / 1000, clamp(Number(args.gain) || 0.08, 0.01, 0.24), {
        pan: audioPan(clampDirectorX(args.x)),
        type: "sine",
      });
    },
    play_c_melody(args) {
      const degrees = Array.isArray(args.degrees) ? args.degrees.slice(0, 5) : [0, 2, 4];
      const stepMs = clamp(Number(args.stepMs) || 180, 80, 900);
      const pan = audioPan(clampDirectorX(args.x));
      const gain = clamp(Number(args.gain) || 0.065, 0.01, 0.2);
      degrees.forEach((degree, i) => {
        playTone("ray", cMajorFrequency(Math.floor(Number(degree) || 0), 130.8128), 0.2, gain, {
          delay: (stepMs * i) / 1000,
          pan,
          type: i % 2 ? "triangle" : "sine",
        });
      });
    },
    play_brilliance_sound(args) {
      audioRayGlint(0, Math.floor(rand() * 12), Math.floor(directorNow() / 1000), clamp(Number(args.strength) || 0.55, 0.25, 1.2), clampDirectorX(args.x), clampDirectorY(args.y));
    },
    filter_pulse(args) {
      if (!audioLayer.ctx || !audioLayer.enabled) return;
      const bus = String(args.bus || "ray");
      const filter = audioLayer.filters[bus] || audioLayer.filters.ray;
      const now = audioLayer.ctx.currentTime;
      const base = filter.frequency.value;
      const amount = clamp(Number(args.amount) || 0.4, -0.8, 1.8);
      const duration = clampDirectorDuration(args.durationMs, 700, 6000) / 1000;
      filter.frequency.setTargetAtTime(Math.max(40, base * (1 + amount)), now, 0.08);
      filter.frequency.setTargetAtTime(base, now + duration, 0.35);
    },
    noise_tick(args) {
      playNoise("artifact", clampDirectorDuration(args.durationMs, 120, 1200) / 1000, clamp(Number(args.strength) || 0.05, 0.01, 0.16), {
        pan: audioPan(clampDirectorX(args.x)),
        frequency: 900 + clampDirectorY(args.y) * 1.8,
        q: 9,
      });
    },
  };

  function raySegmentsAt(t, family, familyIndex = -1) {
    const observer = observerDelta();
    const directorNudge = directorRayNudge(familyIndex, t);
    const axis = normalize3(
      family.axis.x + family.axisNudge.x + directorNudge.x,
      family.axis.y + family.axisNudge.y + directorNudge.y,
      family.axis.z + family.axisNudge.z + directorNudge.z,
    );
    const origin = familyOrigin(family);
    const ox = origin.x;
    const oy = origin.y;
    const observerBend = Math.sin(t * 0.05 + family.phase) * observer.ray * 0.08 + observer.pressure * 0.035;
    const angle = t * family.speed * controls.raySpeed + family.phase + state.keyboardPhase * 0.006 + observerBend;
    return family.rays.map((ray, i) => {
      const v = rotate3(ray, axis, angle + Math.sin(t * 0.04 + ray.phase) * 0.34);
      const depth = 1 / (family.projection + family.projectionNudge + directorNudge.projection + (v.z + 1.35) * 0.34);
      return {
        index: i,
        z: v.z,
        x0: ox,
        y0: oy,
        x1: ox + v.x * ray.length * depth,
        y1: oy + v.y * ray.length * depth,
      };
    });
  }

  function distanceToSegment(px, py, x0, y0, x1, y1) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len2 = dx * dx + dy * dy || 1;
    const t = clamp(((px - x0) * dx + (py - y0) * dy) / len2, 0, 1);
    return Math.hypot(px - (x0 + dx * t), py - (y0 + dy * t));
  }

  function hitTestRay(wx, wy, t) {
    let best = { index: -1, mode: "none", distance: Infinity };
    for (let i = 0; i < rayFamilies.length; i += 1) {
      const family = rayFamilies[i];
      const origin = familyOrigin(family);
      const ox = origin.x;
      const oy = origin.y;
      const od = Math.hypot(wx - ox, wy - oy);
      if (od < best.distance && od < 34) best = { index: i, mode: "origin", distance: od };
    }
    return best;
  }

  function drawRayToField(x0, y0, x1, y1, width, amount, depthNear, brilliance) {
    const sx0 = Math.round((x0 / WORLD_W) * SUB_W);
    const sy0 = Math.round((y0 / WORLD_H) * SUB_H);
    const sx1 = Math.round((x1 / WORLD_W) * SUB_W);
    const sy1 = Math.round((y1 / WORLD_H) * SUB_H);
    let x = sx0;
    let y = sy0;
    const dx = Math.abs(sx1 - sx0);
    const sx = sx0 < sx1 ? 1 : -1;
    const dy = -Math.abs(sy1 - sy0);
    const sy = sy0 < sy1 ? 1 : -1;
    let err = dx + dy;
    let guard = 0;
    while (true) {
      stampRayField(x, y, width, amount, depthNear, brilliance);
      if (x === sx1 && y === sy1) break;
      const e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y += sy;
      }
      guard += 1;
      if (guard > 3000) break;
    }
  }

  function stampRayField(cx, cy, radius, amount, depthNear = 0.5, brilliance = 0) {
    const r = Math.max(1, Math.ceil(radius));
    for (let y = cy - r; y <= cy + r; y += 1) {
      if (y < 0 || y >= SUB_H) continue;
      for (let x = cx - r; x <= cx + r; x += 1) {
        if (x < 0 || x >= SUB_W) continue;
        const dist = Math.hypot(x - cx, y - cy);
        if (dist > r) continue;
        const falloff = 1 - dist / r;
        const idx = y * SUB_W + x;
        const value = amount * (0.4 + falloff * 0.9);
        rayField[idx] += value;
        rayDepthField[idx] += value * depthNear;
        rayBrillianceField[idx] += value * brilliance;
        if (dragField[idx] > 0.025) {
          rayPerturbColorField[idx] = Math.max(rayPerturbColorField[idx], dragField[idx] * value * (0.95 + depthNear * 0.45));
        }
      }
    }
  }

  function updateRayField(t) {
    const observer = observerDelta();
    rayField.fill(0);
    rayDepthField.fill(0);
    rayBrillianceField.fill(0);
    const families = rayFamilies.concat(temporaryRayFamilies);
    for (let f = 0; f < families.length; f += 1) {
      const family = families[f];
      const decay = family.decay === undefined ? 1 : family.decay;
      const segments = raySegmentsAt(t, family, f);
      const directorBrilliance = directorRayBrilliance(f);
      for (const segment of segments) {
        if (segment.z < -0.98 && segment.index % 5 !== 0) continue;
        const ray = family.rays[segment.index];
        const near = smoothstep(-0.7, 1, segment.z);
        const glintWindow = Math.floor(t * 0.09 + ray.phase * 0.07);
        const brillianceThreshold = clamp(0.955 - observer.ray * 0.026 - observer.stillness * 0.012, 0.91, 0.965);
        const rareGate = hash2(segment.index + f * 31, glintWindow, 7171) > brillianceThreshold;
        const slowEnvelope = Math.max(0, Math.sin(t * 0.32 + ray.phase * 1.7));
        const brilliance =
          near > 0.86 && rareGate
            ? Math.pow(near, 5.5) * Math.pow(slowEnvelope, 3.2) * (1 + observer.ray * 0.28)
            : directorBrilliance * Math.pow(near, 2.2) * 0.42;
        if (brilliance > 0.28) {
          audioRayGlint(f, segment.index, glintWindow, brilliance, (segment.x0 + segment.x1) * 0.5, (segment.y0 + segment.y1) * 0.5);
        }
        drawRayToField(
          segment.x0,
          segment.y0,
          segment.x1,
          segment.y1,
          ray.width * controls.glint * (0.78 + near * 0.55),
          (0.055 + near * 0.13 + brilliance * 0.16) * controls.glint * decay,
          near,
          brilliance,
        );
      }
      stampRayField(
        Math.round((familyOrigin(family).x / WORLD_W) * SUB_W),
        Math.round((familyOrigin(family).y / WORLD_H) * SUB_H),
        5,
        0.7 * decay,
        1,
      );
    }
    stampDirectorRayEffects(t);
    for (const fossil of fossils) {
      const sx = Math.round((fossil.x / WORLD_W) * SUB_W);
      const sy = Math.round((fossil.y / WORLD_H) * SUB_H);
      const radius = Math.max(2, Math.round((fossil.radius * fossil.scale / WORLD_W) * SUB_W));
      const alpha = clamp(fossil.life, 0, 1.4);
      for (let y = sy - radius; y <= sy + radius; y += 1) {
        if (y < 0 || y >= SUB_H) continue;
        for (let x = sx - radius; x <= sx + radius; x += 1) {
          if (x < 0 || x >= SUB_W) continue;
          const dx = x - sx;
          const dy = y - sy;
          const dist = Math.hypot(dx, dy);
          if (dist > radius) continue;
          const angle = Math.atan2(dy, dx);
          const ring = Math.sin(dist * 1.7 - t * (5.2 + fossil.glitch) + fossil.phase);
          const radial = Math.sin(angle * 9 + t * 1.3 + fossil.phase);
          const gate = ring * 0.58 + radial * 0.42;
          if (gate < 0.16 && hash2(x, y, fossil.salt + Math.floor(t * 7)) < 0.78) continue;
          const falloff = 1 - dist / radius;
          stampRayField(x, y, 1.3 + fossil.scale, alpha * falloff * (0.05 + Math.max(0, gate) * 0.12), 0.65 + 0.25 * Math.sin(t + fossil.phase));
        }
      }
    }
  }

  function drawSubstrate(t) {
    const theme = themes[controls.theme] || themes.god;
    const algorithmName = controls.algorithm;
    const algorithm = algorithms[algorithmName] || algorithms.god;
    const base = theme.base;
    const data = substrateImage.data;
    const glint = glintImage.data;
    const observer = observerDelta();
    const motion = clamp(
      controls.motion * (1 + observer.agitation * 0.035 + observer.pressure * 0.025 - observer.stillness * 0.025),
      0.05,
      3.6,
    );
    const density = clamp(controls.density + observer.attention * 0.14 + observer.pressure * 0.08 - observer.stillness * 0.1, 0, 3.4);
    const coherence = observer.stillness * 0.18 + observer.ray * 0.05;
    const blur = controls.blur;
    const rayBlur = controls.rayBlur;
    const directorPixelEffects = directorEffects.filter(
      (effect) =>
        effect.type === "quantize_patch" ||
        effect.type === "scanline_tear" ||
        effect.type === "tile_pattern" ||
        effect.type === "bresenham_circle",
    );
    const px = (state.pointerX / WORLD_W) * SUB_W;
    const py = (state.pointerY / WORLD_H) * SUB_H;
    for (let y = 0; y < SUB_H; y += 1) {
      for (let x = 0; x < SUB_W; x += 1) {
        const i = y * SUB_W + x;
        const idx = i * 4;
        const ray = clamp(rayField[i], 0, 1.4);
        const rayDepth = ray > 0.0001 ? clamp(rayDepthField[i] / rayField[i], 0, 1) : 0;
        const rayBrilliance = ray > 0.0001 ? clamp(rayBrillianceField[i] / rayField[i], 0, 1) : 0;
        const rayPerturb = clamp(rayPerturbColorField[i], 0, 1.2);
        const probe = clamp(probeField[i], 0, 1.6);
        const collapse = clamp(collapseField[i], 0, 1.8);
        const measurement = clamp(measurementField[i], 0, 1.8);
        const interaction = clamp(probe * 0.7 + collapse, 0, 1.9);
        const square = algorithmName === "god" ? 1 : 0.2;
        const snap = Math.max(1, Math.round(algorithm.snap * square));
        const group = Math.max(snap, algorithm.group * square);
        const sx = Math.floor(x / snap) * snap;
        const sy = Math.floor(y / snap) * snap;
        const gx = Math.floor(x / group) * group;
        const gy = Math.floor(y / group) * group;
        const dist = Math.hypot(x - px, y - py);
        const pointer = Math.sin(dist * 0.42 - t * 5.7) * smoothstep(44, 0, dist) * state.perturb;
        const domain = valueNoise(gx + t * motion * 4.4, gy - t * motion * 3.2, 19 + square * 3, 5151);
        const a = valueNoise(sx + Math.sin(t * 0.13 * motion + domain) * 42 * algorithm.flow, sy - t * (7 + motion * 10) * algorithm.flow, 13 + square * 0.8, 9101);
        const b = valueNoise(sx * 0.78 - t * (9 + motion * 14) * algorithm.flow, sy * 0.78 + Math.cos(t * 0.17 + domain) * 37 * algorithm.flow, 8.5 + square * 0.6, 3127);
        const c = valueNoise(
          sx + Math.sin(sy * 0.045 + t * 0.9 * motion * algorithm.flow) * 26 * algorithm.flow,
          sy + Math.cos(sx * 0.033 - t * 0.72 * motion * algorithm.flow) * 26 * algorithm.flow,
          4.7 + square * 0.4,
          6401,
        );
        const memoryHold = clamp(0.94 - motion * 0.018 + coherence * 0.015, 0.88, 0.975);
        const memoryWrite = clamp(0.06 + motion * 0.026 - coherence * 0.01, 0.035, 0.16);
        fieldB[i] =
          fieldA[i] * memoryHold +
          (a * 0.36 + b * 0.31 + c * 0.33 + pointer * 0.09 + ray * 0.16 + probe * 0.34 + collapse * 0.3) *
            memoryWrite;
        const vein = Math.sin((a - b) * TAU * 3.4 + c * 5.8 + t * (0.85 + motion * 0.9));
        const grain = hash2(Math.floor(x / Math.max(2, square * 2)), Math.floor(y / Math.max(2, square * 2)), Math.floor(t * 1.2));
        const groupNoise = valueNoise(gx - t * motion * 2.7, gy + t * motion * 2.1, 23 + square * 4, 9901);
        const groupX = Math.floor(gx / group);
        const groupY = Math.floor(gy / group);
        const bitplane = (((Math.floor(x / snap) & Math.floor(y / snap)) ^ Math.floor(t * (1 + motion))) & 7) / 7;
        let shape = fieldB[i] * 0.42 + (vein * 0.5 + 0.5) * 0.24 + c * 0.14 + groupNoise * 0.2 + bitplane * algorithm.bit + probe * 0.2 + collapse * 0.34;
        if (coherence > 0) {
          const resolved = groupNoise * 0.46 + fieldB[i] * 0.34 + (vein * 0.5 + 0.5) * 0.2;
          shape = lerp(shape, resolved, clamp(coherence, 0, 0.28));
        }
        if (algorithmName === "cellular") {
          const tick = Math.floor(t * (0.8 + motion * 0.55));
          const n =
            hash2(groupX, groupY, tick) +
            hash2(groupX + 1, groupY, tick) +
            hash2(groupX - 1, groupY, tick) +
            hash2(groupX, groupY + 1, tick) +
            hash2(groupX, groupY - 1, tick);
          const alive = n > 2.45 ? 1 : n > 1.92 ? 0.62 : 0.12;
          shape = fieldB[i] * 0.28 + groupNoise * 0.28 + alive * 0.34 + (vein * 0.5 + 0.5) * 0.1;
        } else if (algorithmName === "scanline") {
          const scan = ((Math.floor(y / snap) + Math.floor(t * (5 + motion * 4))) & 7) / 7;
          const tear = valueNoise(gx + t * 18 * motion, gy, 5 + square, 1212);
          shape = fieldB[i] * 0.3 + groupNoise * 0.22 + scan * 0.22 + tear * 0.18 + bitplane * 0.08;
        } else if (algorithmName === "lattice") {
          const lx = Math.abs(Math.sin(sx * 0.045 + t * motion * 0.8));
          const ly = Math.abs(Math.sin(sy * 0.052 - t * motion * 0.65));
          const lattice = Math.max(lx, ly);
          shape = fieldB[i] * 0.24 + groupNoise * 0.24 + lattice * 0.36 + bitplane * 0.16;
        } else if (algorithmName === "interference") {
          const w0 = Math.sin(Math.hypot(x - SUB_W * 0.25, y - SUB_H * 0.72) * 0.12 - t * (2 + motion));
          const w1 = Math.sin(Math.hypot(x - SUB_W * 0.79, y - SUB_H * 0.58) * 0.1 + t * (1.6 + motion * 0.8));
          const waves = (w0 + w1) * 0.25 + 0.5;
          shape = fieldB[i] * 0.28 + groupNoise * 0.22 + waves * 0.34 + (vein * 0.5 + 0.5) * 0.16;
        }
        let directorGlint = 0;
        if (directorPixelEffects.length > 0) {
          const directorMod = directorPixelMod(x, y, t, shape, directorPixelEffects);
          shape = directorMod.shape;
          directorGlint = directorMod.glint;
        }
        if (algorithm.quant > 0) {
          const levels = 6 + algorithm.bit * 10 + algorithm.quant * 3;
          shape = Math.floor(shape * levels) / levels;
        }
        if (collapse > 0.03) {
          const levels = 5 + Math.floor(collapse * 9);
          shape = Math.floor(shape * levels) / levels;
        }
        if (measurement > 0.025) {
          const measuredLevels = 3 + Math.floor(measurement * 9);
          const measuredShape = Math.floor((shape + hash2(groupX, groupY, 4703) * 0.035) * measuredLevels) / measuredLevels;
          shape = lerp(shape, measuredShape, clamp(measurement * 0.92, 0, 0.96));
        }
        const threshold = 0.43 + algorithm.threshold - density * 0.045 - ray * (0.08 + rayDepth * 0.08);
        let color = base;
        if (shape > threshold || grain > 0.965 - density * 0.035) {
          const paletteSlot = Math.floor((shape * 9 + groupNoise * 5 + hash2(groupX, groupY, 8080) * 4) % theme.palette.length);
          const pc = palette[theme.palette[paletteSlot]];
          let mix = clamp(((shape - threshold) * 1.35 + 0.2 + ray * (0.22 + rayDepth * 0.32)) * theme.mix, 0.14, 0.92);
          if (measurement > 0.025) mix = Math.floor(lerp(mix, clamp(mix + measurement * 0.18, 0, 1), measurement) * 5) / 5;
          color = [
            Math.floor(lerp(base[0], pc[0], mix)),
            Math.floor(lerp(base[1], pc[1], mix)),
            Math.floor(lerp(base[2], pc[2], mix)),
          ];
        }
        if (measurement > 0.035) {
          const hard = clamp(measurement * 0.65, 0, 0.72);
          const band = Math.floor((shape + groupNoise * 0.35) * 6) / 6;
          const slot = Math.floor(Math.abs(band * theme.palette.length * 1.7 + groupX + groupY) % theme.palette.length);
          const pc = palette[theme.palette[slot]];
          color = [
            Math.floor(lerp(color[0], pc[0], hard * 0.34)),
            Math.floor(lerp(color[1], pc[1], hard * 0.34)),
            Math.floor(lerp(color[2], pc[2], hard * 0.34)),
          ];
          if (((groupX ^ groupY ^ Math.floor(measurement * 8)) & 3) === 0) {
            color = [
              Math.floor(lerp(color[0], theme.glint[0], hard * 0.16)),
              Math.floor(lerp(color[1], theme.glint[1], hard * 0.12)),
              Math.floor(lerp(color[2], theme.glint[2], hard * 0.16)),
            ];
          }
        }
        if (interaction > 0.035) {
          const wake = clamp(interaction * 0.36, 0, 0.5);
          color = [
            Math.floor(lerp(color[0], theme.glint[0], wake)),
            Math.floor(lerp(color[1], theme.glint[1], wake * 0.72)),
            Math.floor(lerp(color[2], theme.glint[2], wake)),
          ];
        }
        if (ray > 0.04) {
          const quietDepth = 0.38 + rayDepth * 0.34;
          const brilliant = rayBrilliance > 0.025 ? rayBrilliance : 0;
          const hot = ray * quietDepth + brilliant * 1.7;
          const perturbColorMix = clamp(rayPerturb * (0.66 + rayDepth * 0.24), 0, 0.68);
          color = [
            Math.floor(lerp(color[0], theme.glint[0], clamp(hot * (0.38 + rayDepth * 0.34), 0, 0.96))),
            Math.floor(lerp(color[1], theme.glint[1], clamp(hot * (0.44 + rayDepth * 0.38), 0, 0.98))),
            Math.floor(lerp(color[2], theme.glint[2], clamp(hot * (0.5 + rayDepth * 0.44), 0, 1))),
          ];
          if (perturbColorMix > 0.018) {
            color = [
              Math.floor(lerp(color[0], rayPerturbColor[0], perturbColorMix)),
              Math.floor(lerp(color[1], rayPerturbColor[1], perturbColorMix)),
              Math.floor(lerp(color[2], rayPerturbColor[2], perturbColorMix)),
            ];
          }
          glint[idx] = color[0];
          glint[idx + 1] = color[1];
          glint[idx + 2] = color[2];
          glint[idx + 3] = Math.floor(clamp(ray * (70 + rayDepth * 95) + brilliant * 185 + perturbColorMix * 78, 0, 245));
        } else if (interaction > 0.05) {
          glint[idx] = Math.floor(lerp(color[0], theme.glint[0], 0.46));
          glint[idx + 1] = Math.floor(lerp(color[1], theme.glint[1], 0.32));
          glint[idx + 2] = Math.floor(lerp(color[2], theme.glint[2], 0.46));
          glint[idx + 3] = Math.floor(clamp(interaction * 92, 0, 130));
        } else if (measurement > 0.08 && hash2(groupX, groupY, 6007) > 0.68) {
          const edge = clamp(measurement * 0.46, 0, 0.4);
          glint[idx] = Math.floor(lerp(color[0], theme.glint[0], edge));
          glint[idx + 1] = Math.floor(lerp(color[1], theme.glint[1], edge * 0.62));
          glint[idx + 2] = Math.floor(lerp(color[2], theme.glint[2], edge));
          glint[idx + 3] = Math.floor(clamp(measurement * 42, 0, 80));
        } else if (directorGlint > 0.02) {
          const edge = clamp(directorGlint, 0, 0.42);
          glint[idx] = Math.floor(lerp(color[0], theme.glint[0], edge));
          glint[idx + 1] = Math.floor(lerp(color[1], theme.glint[1], edge * 0.66));
          glint[idx + 2] = Math.floor(lerp(color[2], theme.glint[2], edge));
          glint[idx + 3] = Math.floor(clamp(edge * 150, 0, 110));
        } else {
          glint[idx + 3] = 0;
        }
        data[idx] = color[0];
        data[idx + 1] = color[1];
        data[idx + 2] = color[2];
        data[idx + 3] = 255;
      }
    }
    fieldA.set(fieldB);
    substrateCtx.putImageData(substrateImage, 0, 0);
    glintCtx.putImageData(glintImage, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.save();
    ctx.filter = `blur(${blur}px)`;
    const pad = blur * 3;
    ctx.drawImage(substrateCanvas, -pad, -pad, WORLD_W + pad * 2, WORLD_H + pad * 2);
    ctx.restore();
    if (rayBlur > 0) {
      ctx.save();
      ctx.filter = `blur(${rayBlur}px)`;
      const rayPad = rayBlur * 3;
      ctx.drawImage(glintCanvas, -rayPad, -rayPad, WORLD_W + rayPad * 2, WORLD_H + rayPad * 2);
      ctx.restore();
    } else {
      ctx.drawImage(glintCanvas, 0, 0, WORLD_W, WORLD_H);
    }
  }

  function drawBlocks() {
    const count = Math.floor(blocks.length * controls.blocks);
    for (let i = 0; i < count; i += 1) {
      const block = blocks[i];
      ctx.fillStyle = rgb(block.color);
      ctx.fillRect(block.x, block.y, block.w, block.h);
      if (i % 3 === 0) {
        ctx.fillStyle = rgb(block.accent);
        const cell = block.cell;
        for (let y = 0; y < block.h; y += cell * 2) {
          for (let x = 0; x < block.w; x += cell * 3) {
            if (hash2(x / cell, y / cell, block.salt) > 0.58) {
              ctx.fillRect(block.x + x, block.y + y, cell, cell);
            }
          }
        }
      }
    }
  }

  function setupControls() {
    for (const key of Object.keys(defaults)) {
      const input = document.getElementById(key);
      if (input.tagName === "SELECT") {
        input.addEventListener("change", () => {
          controls[key] = input.value;
          setAudioScene();
        });
        continue;
      }
      input.addEventListener("input", () => {
        controls[key] = key === "blur" || key === "rayBlur" ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value);
        setAudioScene();
      });
    }
    panelToggle.addEventListener("click", () => panel.classList.toggle("is-collapsed"));
    if (audioToggle) {
      audioToggle.addEventListener("click", () => {
        if (audioLayer.enabled) disableAudio();
        else enableAudio();
      });
    }
    resetParams.addEventListener("click", () => {
      for (const key of Object.keys(defaults)) {
        controls[key] = defaults[key];
        document.getElementById(key).value = String(defaults[key]);
      }
      setAudioScene();
    });
  }

  function drawWorld(timestamp) {
    const t = timestamp * 0.001;
    const dt = state.lastTime > 0 ? Math.min(0.08, (timestamp - state.lastTime) * 0.001) : 0.016;
    state.lastTime = timestamp;
    state.frame += 1;
    state.perturb *= 0.94;

    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    ctx.fillStyle = "#03020b";
    ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
    ctx.setTransform(
      state.dpr * state.zoom,
      0,
      0,
      state.dpr * state.zoom,
      state.dpr * state.panX,
      state.dpr * state.panY,
    );
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, WORLD_W, WORLD_H);
    ctx.clip();
    updateObserver(dt, t);
    decayInteractionFields();
    updateDirector(dt, t);
    updateEventMemories(dt, t);
    updateGlobalReconfiguration(dt);
    updateRayField(t);
    drawSubstrate(t);
    drawBlocks();
    ctx.restore();
    requestAnimationFrame(drawWorld);
  }

  function onPointerMove(event) {
    const w = screenToWorld(event.clientX, event.clientY);
    const movement = Math.hypot(event.movementX || 0, event.movementY || 0) / Math.max(0.2, state.zoom);
    if (movement > 0.25) observe("move", clamp(movement / 90, 0, state.pointerDown ? 0.55 : 0.18));
    if (movement > 0.25) recordDirectorEvent(state.pointerDown ? "drag" : "hover", w.x, w.y, { gesture: state.activeGesture });
    state.pointerX = w.x;
    state.pointerY = w.y;
    if (state.pointerDown) {
      const dx = event.clientX - state.dragX;
      const dy = event.clientY - state.dragY;
      const worldDx = w.x - state.lastWorldX;
      const worldDy = w.y - state.lastWorldY;
      state.dragDistance += Math.hypot(dx, dy);
      if (state.activeGesture === "pan") {
        state.panX += dx;
        state.panY += dy;
        clampView();
      } else if (state.activeGesture === "ray-origin" && state.activeRayIndex >= 0) {
        const family = rayFamilies[state.activeRayIndex];
        family.axisNudge = normalize3(
          family.axisNudge.x + worldDy * 0.0014,
          family.axisNudge.y - worldDx * 0.0014,
          family.axisNudge.z + (worldDx - worldDy) * 0.00045,
        );
        family.projectionNudge = clamp(family.projectionNudge + (worldDx + worldDy) * 0.00008, -0.18, 0.28);
        const origin = familyOrigin(family);
        stampField(probeField, origin.x, origin.y, 116, 0.14);
        observe("ray", clamp(movement / 70, 0.08, 0.8));
        recordDirectorEvent("ray_origin_drag", origin.x, origin.y, { family: state.activeRayIndex });
        audioRayHandle(state.activeRayIndex, w.x, w.y, movement);
      } else {
        stampProbeLine(state.lastWorldX, state.lastWorldY, w.x, w.y, 0.14);
        disturbFossils(w.x, w.y, 82, 0.026);
        audioProbeMove(w.x, w.y, movement, true);
      }
      state.dragX = event.clientX;
      state.dragY = event.clientY;
      state.lastWorldX = w.x;
      state.lastWorldY = w.y;
      state.perturb = Math.min(1.8, state.perturb + 0.04);
    } else {
      stampField(probeField, w.x, w.y, 48, 0.03);
      disturbFossils(w.x, w.y, 48, 0.004);
      audioProbeMove(w.x, w.y, movement, false);
    }
  }

  function onWheel(event) {
    if (event.target.closest && event.target.closest(".panel")) return;
    event.preventDefault();
    ensureAudioFromGesture();
    const w = screenToWorld(event.clientX, event.clientY);
    observe("scroll", clamp(Math.abs(event.deltaY) / 420, 0.22, 1.25));
    recordDirectorEvent("wheel", w.x, w.y, { direction: event.deltaY < 0 ? 1 : -1 });
    if (state.spaceDown) {
      const before = screenToWorld(event.clientX, event.clientY);
      const factor = Math.exp(-event.deltaY * 0.0012);
      state.zoom = clamp(state.zoom * factor, minCoverZoom(), 24);
      state.panX = event.clientX - before.x * state.zoom;
      state.panY = event.clientY - before.y * state.zoom;
      clampView();
      audioScroll(w.x, w.y, event.deltaY < 0 ? 1 : -1);
      return;
    }
    const direction = event.deltaY < 0 ? 1 : -1;
    stampField(direction > 0 ? collapseField : probeField, w.x, w.y, 118, 0.3);
    if (direction > 0) {
      measureAt(w.x, w.y, 150, 0.42);
      if (hash2(Math.floor(w.x), Math.floor(w.y), Math.floor(state.lastTime * 0.7)) > 0.72) rememberEvent("scroll", w.x, w.y, 0.45);
    }
    disturbFossils(w.x, w.y, 120, direction > 0 ? -0.018 : 0.04);
    audioScroll(w.x, w.y, direction);
    const hit = hitTestRay(w.x, w.y, state.lastTime * 0.001);
    if (hit.index >= 0 && hit.mode === "origin") {
      const family = rayFamilies[hit.index];
      family.projectionNudge = clamp(family.projectionNudge + event.deltaY * 0.0005, -0.28, 0.5);
    }
    for (const fossil of fossils) {
      const d = Math.hypot(fossil.x - w.x, fossil.y - w.y);
      if (d < fossil.radius + 80) fossil.scale = clamp(fossil.scale + direction * 0.08, 0.45, 3.4);
    }
    state.perturb = Math.min(1.5, state.perturb + 0.25);
  }

  function onKeyDown(event) {
    if (event.code === "Space") {
      state.spaceDown = true;
      event.preventDefault();
      return;
    }
    if (event.key === "0") {
      state.zoom = minCoverZoom();
      state.panX = (window.innerWidth - WORLD_W * state.zoom) * 0.5;
      state.panY = (window.innerHeight - WORLD_H * state.zoom) * 0.5;
      clampView();
    } else {
      state.keyboardPhase += event.key.length === 1 ? event.key.charCodeAt(0) : 17;
      state.perturb = 2.2;
      observe("click", 0.28);
      recordDirectorEvent("key", state.pointerX, state.pointerY);
    }
  }

  function onKeyUp(event) {
    if (event.code === "Space") state.spaceDown = false;
  }

  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerdown", (event) => {
    if (event.target.closest && event.target.closest(".panel")) return;
    ensureAudioFromGesture();
    state.pointerDown = true;
    state.dragX = event.clientX;
    state.dragY = event.clientY;
    state.downX = event.clientX;
    state.downY = event.clientY;
    state.dragDistance = 0;
    state.downTime = event.timeStamp;
    const w = screenToWorld(event.clientX, event.clientY);
    observe("click", 0.22);
    state.downWorldX = w.x;
    state.downWorldY = w.y;
    state.lastWorldX = w.x;
    state.lastWorldY = w.y;
    const hit = hitTestRay(w.x, w.y, state.lastTime * 0.001);
    state.activeRayIndex = hit.index;
    state.activeRayMode = hit.mode;
    state.activeGesture = state.spaceDown ? "pan" : hit.index >= 0 ? `ray-${hit.mode}` : "probe";
    stampField(probeField, w.x, w.y, state.activeGesture === "probe" ? 82 : 116, 0.1);
    if (state.activeGesture === "ray-origin") {
      observe("ray", 0.44);
      audioRayHandle(hit.index, w.x, w.y, 18);
    }
    canvas.setPointerCapture(event.pointerId);
  });
  window.addEventListener("pointerup", (event) => {
    if (!state.pointerDown) return;
    const w = screenToWorld(event.clientX, event.clientY);
    const held = event.timeStamp - state.downTime;
    if (state.activeGesture === "probe" && state.dragDistance < 18) {
      if (held > 620) {
        temporaryRayFamilies.push(makeTemporaryRayFamily(w.x, w.y));
        stampField(collapseField, w.x, w.y, 150, 0.68);
        measureAt(w.x, w.y, 210, 0.72);
        rememberEvent("long", w.x, w.y, 1.2);
        observe("click", 0.9);
        recordDirectorEvent("long_press", w.x, w.y, { held: Math.round(held) });
        audioClick(w.x, w.y, held);
      } else {
        stampField(collapseField, w.x, w.y, 128, 0.75);
        measureAt(w.x, w.y, 160, 0.9);
        createFossil(w.x, w.y, 0.7);
        rememberEvent("click", w.x, w.y, 1);
        observe("click", 0.72);
        recordDirectorEvent("click", w.x, w.y);
        audioClick(w.x, w.y, held);
      }
    } else if (state.activeGesture === "ray-origin") {
      stampField(collapseField, w.x, w.y, 112, 0.36);
      measureAt(w.x, w.y, 132, 0.46);
      if (state.dragDistance > 32) rememberEvent("ray", w.x, w.y, 0.78);
      observe("ray", state.dragDistance > 32 ? 0.78 : 0.36);
      recordDirectorEvent("ray_origin_drag", w.x, w.y, { family: state.activeRayIndex });
      audioClick(w.x, w.y, held);
      if (state.dragDistance > 80 && hash2(Math.floor(w.x), Math.floor(w.y), fossils.length) > 0.48) createFossil(w.x, w.y, 0.42);
    }
    state.pointerDown = false;
    state.activeGesture = "none";
    state.activeRayIndex = -1;
    state.activeRayMode = "none";
    canvas.releasePointerCapture(event.pointerId);
  });
  window.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);

  for (let i = 0; i < fieldA.length; i += 1) {
    fieldA[i] = rand();
  }

  setupControls();
  resize();
  requestAnimationFrame(drawWorld);
})();
