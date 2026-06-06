(() => {
  "use strict";

  const canvas = document.getElementById("field");
  const ctx = canvas.getContext("2d", { alpha: false });
  const panel = document.getElementById("panel");
  const panelToggle = document.getElementById("panelToggle");
  const svgInput = document.getElementById("svgInput");
  const selectionPanel = document.getElementById("selectionPanel");
  const objectList = document.getElementById("objectList");

  const WORLD_W = 1920;
  const WORLD_H = 1080;
  const SUB_W = 480;
  const SUB_H = 270;
  const N = SUB_W * SUB_H;
  const TAU = Math.PI * 2;

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

  const themes = {
    god: { base: [8, 6, 24], palette: [0, 1, 2, 3, 4, 5, 6, 7], mix: 1, glint: [245, 250, 255] },
    bitplane: { base: [4, 7, 18], palette: [3, 6, 1, 4, 7], mix: 0.72, glint: [190, 245, 255] },
    phosphor: { base: [2, 13, 9], palette: [2, 3, 1, 7], mix: 0.68, glint: [215, 255, 210] },
    ember: { base: [15, 7, 5], palette: [5, 8, 7, 1, 4], mix: 0.76, glint: [255, 232, 170] },
    mono: { base: [5, 5, 16], palette: [0, 3, 6, 7], mix: 0.5, glint: [225, 235, 255] },
    cyanotype: {
      base: [3, 16, 46],
      colors: [
        [4, 24, 74],
        [7, 43, 112],
        [10, 70, 150],
        [22, 103, 178],
        [64, 145, 204],
        [122, 190, 222],
        [190, 226, 230],
        [236, 241, 222],
      ],
      mix: 0.86,
      glint: [239, 244, 226],
    },
  };

  const builtinThemeCache = {};
  let customThemeCacheKey = "";
  let customThemeCache = null;
  let frameBaseTheme = null;

  const algorithms = {
    god: { threshold: 0, snap: 1, group: 1, bit: 0.04, flow: 1, quant: 0 },
    bitplane: { threshold: 0.04, snap: 4, group: 12, bit: 0.16, flow: 0.7, quant: 1 },
    cellular: { threshold: 0.025, snap: 3, group: 11, bit: 0.18, flow: 0.78, quant: 0.8 },
    scanline: { threshold: 0.02, snap: 2, group: 16, bit: 0.12, flow: 0.64, quant: 0.55 },
    lattice: { threshold: 0.035, snap: 4, group: 10, bit: 0.22, flow: 0.58, quant: 1.15 },
    interference: { threshold: 0.015, snap: 3, group: 13, bit: 0.08, flow: 0.9, quant: 0.45 },
  };

  const defaults = {
    showGrid: true,
    snapGrid: true,
    gridSize: 24,
    zoom: 1,
    pathRole: "current",
    pathStrength: 0.35,
    pathPixelation: 0.08,
    pathRadius: 34,
    pathFrequency: 1,
    pathSpeed: 1,
    pathColor: "#00d6ff",
    theme: "god",
    customLogic: "analog",
    customHue: 218,
    customSpread: 72,
    customSaturation: 0.82,
    customLight: 0.56,
    customContrast: 0.72,
    customTemperature: 0,
    customSeed: 137,
    algorithm: "god",
    density: 1.65,
    motion: 1.45,
    pixel: 1,
    groupScale: 1,
    flow: 1,
    bit: 0.04,
    quant: 0,
    snap: 1,
    threshold: 0,
    memory: 0.94,
    contour: 0.8,
    weave: 0,
    blur: 0,
    stipple: 0,
    exposure: 0,
    scanline: 0,
    glitch: 0,
    contrast: 1,
    svgMode: "current",
    svgForce: 1,
  };

  const substrateKeys = [
    "theme",
    "algorithm",
    "density",
    "motion",
    "pixel",
    "groupScale",
    "flow",
    "bit",
    "quant",
    "snap",
    "threshold",
    "memory",
    "contour",
    "weave",
    "blur",
    "stipple",
    "exposure",
    "scanline",
    "glitch",
    "contrast",
  ];
  const numericSubstrateKeys = substrateKeys.filter((key) => typeof defaults[key] === "number");
  const controls = { ...defaults };
  const fieldA = new Float32Array(N);
  const fieldB = new Float32Array(N);
  const influence = new Float32Array(N);
  const quantField = new Float32Array(N);
  const tearField = new Float32Array(N);
  const pigmentR = new Float32Array(N);
  const pigmentG = new Float32Array(N);
  const pigmentB = new Float32Array(N);
  const flowX = new Float32Array(N);
  const flowY = new Float32Array(N);
  const waveField = new Float32Array(N);
  const turbulenceField = new Float32Array(N);
  const phaseField = new Float32Array(N);
  const damField = new Float32Array(N);
  const pixelationField = new Float32Array(N);
  const substrateCanvas = document.createElement("canvas");
  substrateCanvas.width = SUB_W;
  substrateCanvas.height = SUB_H;
  const substrateCtx = substrateCanvas.getContext("2d", { alpha: false });
  const image = substrateCtx.createImageData(SUB_W, SUB_H);
  const hiddenR = new Float32Array(N);
  const hiddenG = new Float32Array(N);
  const hiddenB = new Float32Array(N);
  const hiddenShape = new Float32Array(N);

  const paths = [];
  const regions = [];
  let clipboard = [];
  let nextPathId = 1;
  let nextRegionId = 1;

  const state = {
    dpr: 1,
    tool: "pointer",
    pointerDown: false,
    dragMode: "none",
    panX: 0,
    panY: 0,
    zoom: 1,
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
    draft: null,
    selectionRect: null,
    time: 0,
    frame: 0,
    seed: 58321,
  };

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function smoothstep(edge0, edge1, x) {
    const t = clamp((x - edge0) / (edge1 - edge0 || 1), 0, 1);
    return t * t * (3 - 2 * t);
  }

  function hash2(x, y, salt = 0) {
    let n = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(salt | 0, 1442695041);
    n = (n ^ (n >>> 13)) | 0;
    n = Math.imul(n, 1274126177);
    return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
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

  function hexToRgb(hex) {
    const clean = String(hex || "#000000").replace("#", "");
    const n = Number.parseInt(clean.length === 3 ? clean.replace(/(.)/g, "$1$1") : clean, 16) || 0;
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function wrapHue(value) {
    return ((value % 360) + 360) % 360;
  }

  function hslToRgb(h, s, l) {
    const hue = wrapHue(h) / 360;
    const sat = clamp(s, 0, 1);
    const light = clamp(l, 0, 1);
    if (sat === 0) {
      const v = Math.round(light * 255);
      return [v, v, v];
    }
    const q = light < 0.5 ? light * (1 + sat) : light + sat - light * sat;
    const p = 2 * light - q;
    const channel = (offset) => {
      let t = hue + offset;
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    return [Math.round(channel(1 / 3) * 255), Math.round(channel(0) * 255), Math.round(channel(-1 / 3) * 255)];
  }

  function themeColor(theme, slot) {
    if (theme.colors) return theme.colors[slot % theme.colors.length];
    return palette[theme.palette[slot % theme.palette.length]];
  }

  function normalizeTheme(name) {
    if (name === "custom") return customTheme();
    if (builtinThemeCache[name]) return builtinThemeCache[name];
    const source = themes[name] || themes.god;
    const normalized = {
      base: source.base.slice(),
      colors: source.colors ? source.colors.map((color) => color.slice()) : source.palette.map((slot) => palette[slot].slice()),
      mix: source.mix,
      glint: source.glint.slice(),
    };
    builtinThemeCache[name] = normalized;
    return normalized;
  }

  function customTheme() {
    const key = [
      controls.customLogic,
      controls.customHue,
      controls.customSpread,
      controls.customSaturation,
      controls.customLight,
      controls.customContrast,
      controls.customTemperature,
      controls.customSeed,
    ].join(":");
    if (customThemeCache && customThemeCacheKey === key) return customThemeCache;
    const hue = Number(controls.customHue);
    const spread = Number(controls.customSpread);
    const sat = Number(controls.customSaturation);
    const light = Number(controls.customLight);
    const contrast = Number(controls.customContrast);
    const temp = Number(controls.customTemperature);
    const seed = Number(controls.customSeed) || 0;
    const offsetSets = {
      analog: [-0.9, -0.52, -0.2, 0.12, 0.45, 0.82, 1.12, 1.45],
      split: [0, 0.24, -0.28, 1.55, -1.55, 1.95, -1.95, 0.72],
      triad: [0, 0.18, 1.66, 1.92, -1.66, -1.94, 0.72, -0.74],
      spectral: [0, 0.46, 0.94, 1.36, 1.9, 2.35, 2.9, 3.45],
      constrained: [0, 0.12, -0.16, 0.34, -0.38, 1.74, 1.92, 2.12],
      cyanotype: [-0.16, -0.08, 0, 0.06, 0.14, 0.24, 0.36, 0.52],
    };
    const offsets = offsetSets[controls.customLogic] || offsetSets.analog;
    const colors = offsets.map((offset, index) => {
      const jitter = (hash2(seed + index * 19, seed - index * 7, 6203) - 0.5) * spread * 0.22;
      const tempBias = temp * (index % 2 === 0 ? 18 : -12);
      const h = hue + offset * spread + jitter + tempBias;
      const s = clamp(sat + (hash2(seed, index, 4441) - 0.5) * 0.22 * contrast, 0.05, 1);
      const l = clamp(light + (index - 3.5) * 0.035 * contrast + (hash2(index, seed, 9271) - 0.5) * 0.12 * contrast, 0.1, 0.94);
      return hslToRgb(h, s, l);
    });
    if (controls.customLogic === "cyanotype") {
      const paper = hslToRgb(62 + temp * 8, clamp(0.18 + sat * 0.12, 0.12, 0.32), clamp(0.78 + light * 0.16, 0.78, 0.96));
      const blue = hslToRgb(214 + temp * 10, clamp(0.58 + sat * 0.28, 0.45, 0.95), clamp(0.2 + light * 0.2, 0.16, 0.42));
      for (let i = 0; i < colors.length; i += 1) {
        const exposure = i / Math.max(1, colors.length - 1);
        const tonal = Math.pow(exposure, lerp(1.8, 0.72, contrast));
        colors[i] = [
          lerp(blue[0], paper[0], tonal),
          lerp(blue[1], paper[1], tonal),
          lerp(blue[2], paper[2], tonal),
        ];
      }
    }
    const baseHue = hue + temp * 26;
    const base = hslToRgb(baseHue, clamp(sat * 0.32, 0.05, 0.55), clamp(light * 0.11, 0.025, 0.16));
    customThemeCacheKey = key;
    customThemeCache = {
      base,
      colors,
      mix: clamp(0.58 + contrast * 0.42, 0.35, 1),
      glint: hslToRgb(hue + 18 + temp * 20, clamp(sat * 0.45, 0.1, 0.8), clamp(light + 0.3, 0.62, 0.98)),
    };
    return customThemeCache;
  }

  function blendThemeObjects(a, b, amount) {
    const t = clamp(amount, 0, 1);
    const count = Math.max(a.colors.length, b.colors.length);
    const colors = [];
    for (let i = 0; i < count; i += 1) {
      const ca = a.colors[i % a.colors.length];
      const cb = b.colors[i % b.colors.length];
      colors.push([lerp(ca[0], cb[0], t), lerp(ca[1], cb[1], t), lerp(ca[2], cb[2], t)]);
    }
    return {
      base: [lerp(a.base[0], b.base[0], t), lerp(a.base[1], b.base[1], t), lerp(a.base[2], b.base[2], t)],
      colors,
      mix: lerp(a.mix, b.mix, t),
      glint: [lerp(a.glint[0], b.glint[0], t), lerp(a.glint[1], b.glint[1], t), lerp(a.glint[2], b.glint[2], t)],
    };
  }

  function renderCustomPreview() {
    const preview = document.getElementById("customPreview");
    if (!preview) return;
    const theme = customTheme();
    preview.innerHTML = "";
    for (const color of theme.colors) {
      const swatch = document.createElement("span");
      swatch.style.backgroundColor = `rgb(${Math.round(color[0])}, ${Math.round(color[1])}, ${Math.round(color[2])})`;
      preview.appendChild(swatch);
    }
  }

  function setupControls() {
    for (const [key, value] of Object.entries(defaults)) {
      const input = document.getElementById(key);
      if (!input) continue;
      if (input.type === "checkbox") input.checked = !!value;
      else input.value = String(value);
      input.addEventListener("input", () => {
        controls[key] = input.type === "checkbox" ? input.checked : input.type === "range" ? Number.parseFloat(input.value) : input.value;
        if (key === "zoom") setZoom(Number.parseFloat(input.value), window.innerWidth * 0.5, window.innerHeight * 0.5);
        if (key.startsWith("custom")) renderCustomPreview();
        if (
          key === "pathRole" ||
          key === "pathStrength" ||
          key === "pathPixelation" ||
          key === "pathRadius" ||
          key === "pathFrequency" ||
          key === "pathSpeed" ||
          key === "pathColor"
        ) {
          updateSelectedStyle();
        }
      });
    }
    document.querySelectorAll("[data-region-key]").forEach((input) => {
      input.addEventListener("input", () => updateSelectedRegionSettings(input.dataset.regionKey, input.type === "range" ? Number.parseFloat(input.value) : input.value));
    });
    const regionAmount = document.getElementById("regionAmount");
    if (regionAmount) regionAmount.addEventListener("input", () => updateSelectedRegionAmount(Number.parseFloat(regionAmount.value)));
    document.querySelectorAll(".tool").forEach((button) => {
      button.addEventListener("click", () => setTool(button.dataset.tool));
    });
    panelToggle.addEventListener("click", () => panel.classList.toggle("is-collapsed"));
    document.getElementById("copyPath").addEventListener("click", copySelected);
    document.getElementById("pastePath").addEventListener("click", pasteClipboard);
    document.getElementById("deletePath").addEventListener("click", deleteSelected);
    document.getElementById("clearPaths").addEventListener("click", () => {
      paths.splice(0);
      updateInspector();
      renderObjectList();
    });
    document.getElementById("regionFromSelection").addEventListener("click", makeRegionsFromSelection);
    document.getElementById("randomize").addEventListener("click", randomizeSubstrate);
    document.getElementById("exportPng").addEventListener("click", exportPng);
    svgInput.addEventListener("change", importSvg);
    renderObjectList();
    updateInspector();
    renderCustomPreview();
  }

  function setTool(tool) {
    state.tool = tool;
    document.querySelectorAll(".tool").forEach((item) => item.classList.toggle("is-active", item.dataset.tool === tool));
  }

  function resize() {
    state.dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = Math.floor(window.innerWidth * state.dpr);
    canvas.height = Math.floor(window.innerHeight * state.dpr);
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    if (state.frame === 0) {
      state.zoom = Math.max(window.innerWidth / WORLD_W, window.innerHeight / WORLD_H);
      controls.zoom = state.zoom;
      const zoomInput = document.getElementById("zoom");
      if (zoomInput) zoomInput.value = String(state.zoom);
      state.panX = (window.innerWidth - WORLD_W * state.zoom) * 0.5;
      state.panY = (window.innerHeight - WORLD_H * state.zoom) * 0.5;
    }
  }

  function screenToWorld(clientX, clientY) {
    return {
      x: clamp((clientX - state.panX) / state.zoom, 0, WORLD_W),
      y: clamp((clientY - state.panY) / state.zoom, 0, WORLD_H),
    };
  }

  function worldToScreen(point) {
    return { x: point.x * state.zoom + state.panX, y: point.y * state.zoom + state.panY };
  }

  function snapPoint(point) {
    if (!controls.snapGrid) return point;
    const g = controls.gridSize;
    return { x: Math.round(point.x / g) * g, y: Math.round(point.y / g) * g };
  }

  function setZoom(next, sx, sy) {
    const before = screenToWorld(sx, sy);
    state.zoom = clamp(next, 0.4, 16);
    controls.zoom = state.zoom;
    state.panX = sx - before.x * state.zoom;
    state.panY = sy - before.y * state.zoom;
  }

  function createPath(type, points) {
    const path = {
      id: nextPathId++,
      type,
      points: points.map((point) => ({ ...point })),
      role: controls.pathRole,
      strength: controls.pathStrength,
      pixelation: controls.pathPixelation,
      radius: controls.pathRadius,
      frequency: controls.pathFrequency,
      speed: controls.pathSpeed,
      color: controls.pathColor,
      selected: false,
    };
    paths.push(path);
    selectOnly(path);
    renderObjectList();
    return path;
  }

  function selectOnly(path) {
    paths.forEach((item) => (item.selected = item === path));
    regions.forEach((item) => (item.selected = false));
    syncSelectedControls(path);
    updateInspector();
    renderObjectList();
  }

  function syncSelectedControls(path) {
    if (!path) return;
    for (const key of ["pathRole", "pathStrength", "pathPixelation", "pathRadius", "pathFrequency", "pathSpeed", "pathColor"]) {
      const prop = key.replace("path", "").replace(/^./, (c) => c.toLowerCase());
      controls[key] = path[prop];
      const input = document.getElementById(key);
      if (input) input.value = String(path[prop]);
    }
  }

  function selectRegionOnly(region) {
    paths.forEach((item) => (item.selected = false));
    regions.forEach((item) => (item.selected = item === region));
    syncSelectedRegionControls(region);
    updateInspector();
    renderObjectList();
  }

  function selectedPaths() {
    return paths.filter((path) => path.selected);
  }

  function selectedRegions() {
    return regions.filter((region) => region.selected);
  }

  function updateInspector() {
    const hasPaths = selectedPaths().length > 0;
    const hasRegions = selectedRegions().length > 0;
    selectionPanel.classList.toggle("is-hidden", !hasPaths && !hasRegions);
    document.getElementById("pathInspector").classList.toggle("is-hidden", !hasPaths);
    document.getElementById("regionInspector").classList.toggle("is-hidden", !hasRegions);
    if (hasPaths) syncSelectedControls(selectedPaths()[0]);
    if (hasRegions) syncSelectedRegionControls(selectedRegions()[0]);
  }

  function syncSelectedRegionControls(region) {
    if (!region) return;
    const amountInput = document.getElementById("regionAmount");
    if (amountInput) amountInput.value = String(region.amount);
    document.querySelectorAll("[data-region-key]").forEach((input) => {
      const key = input.dataset.regionKey;
      input.value = String(region.settings[key]);
    });
  }

  function updateSelectedRegionAmount(value) {
    for (const region of selectedRegions()) region.amount = clamp(value, 0, 1);
    renderObjectList();
  }

  function updateSelectedRegionSettings(key, value) {
    for (const region of selectedRegions()) region.settings[key] = value;
    renderObjectList();
  }

  function updateSelectedStyle() {
    for (const path of paths) {
      if (!path.selected) continue;
      path.role = controls.pathRole;
      path.strength = controls.pathStrength;
      path.pixelation = controls.pathPixelation;
      path.radius = controls.pathRadius;
      path.frequency = controls.pathFrequency;
      path.speed = controls.pathSpeed;
      path.color = controls.pathColor;
    }
    renderObjectList();
  }

  function pathBounds(path) {
    const pts = expandedPoints(path);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const point of pts) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }
    return { minX, minY, maxX, maxY };
  }

  function boundsIntersect(a, b) {
    return a.minX <= b.maxX && a.maxX >= b.minX && a.minY <= b.maxY && a.maxY >= b.minY;
  }

  function expandedPoints(path) {
    if (path.type === "rect") {
      const [a, b] = path.points;
      return [a, { x: b.x, y: a.y }, b, { x: a.x, y: b.y }, a];
    }
    if (path.type === "ellipse") {
      const [a, b] = path.points;
      const cx = (a.x + b.x) * 0.5;
      const cy = (a.y + b.y) * 0.5;
      const rx = Math.abs(b.x - a.x) * 0.5;
      const ry = Math.abs(b.y - a.y) * 0.5;
      const pts = [];
      for (let i = 0; i <= 64; i += 1) {
        const t = (i / 64) * TAU;
        pts.push({ x: cx + Math.cos(t) * rx, y: cy + Math.sin(t) * ry });
      }
      return pts;
    }
    return path.points;
  }

  function regionBounds(region) {
    return pathBounds(region);
  }

  function distanceToPolygonEdge(point, polygon) {
    let best = Infinity;
    for (let i = 0; i < polygon.length; i += 1) {
      const a = polygon[i];
      const b = polygon[(i + 1) % polygon.length];
      best = Math.min(best, distanceToSegment(point, a, b));
    }
    return best;
  }

  function regionMaskAt(point, region) {
    if (!pointInPolygon(point, region.points)) return 0;
    const blend = clamp(region.amount, 0, 1);
    if (blend <= 0) return 1;
    const bounds = region.bounds || regionBounds(region);
    region.bounds = bounds;
    const minDim = Math.max(1, Math.min(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY));
    const feather = Math.max(1, Math.min(140, minDim * 0.35) * blend);
    return smoothstep(0, feather, distanceToPolygonEdge(point, region.points));
  }

  function pointInPolygon(point, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
      const a = polygon[i];
      const b = polygon[j];
      const crosses = a.y > point.y !== b.y > point.y && point.x < ((b.x - a.x) * (point.y - a.y)) / ((b.y - a.y) || 1) + a.x;
      if (crosses) inside = !inside;
    }
    return inside;
  }

  function closedPathAt(point) {
    for (let i = paths.length - 1; i >= 0; i -= 1) {
      const path = paths[i];
      const pts = expandedPoints(path);
      if (pts.length < 3) continue;
      if (path.type === "line" || path.type === "pointCloud") continue;
      if (pointInPolygon(point, pts)) return path;
    }
    return null;
  }

  function createRegionFromPath(path) {
    const points = expandedPoints(path).map((point) => ({ ...point }));
    if (points.length < 3) return null;
    const region = {
      id: nextRegionId++,
      type: "region",
      name: `region ${nextRegionId - 1}`,
      points,
      amount: 0,
      settings: Object.fromEntries(substrateKeys.map((key) => [key, controls[key]])),
      selected: false,
      bounds: null,
    };
    regions.push(region);
    selectRegionOnly(region);
    return region;
  }

  function makeRegionsFromSelection() {
    const made = selectedPaths().map(createRegionFromPath).filter(Boolean);
    if (made.length > 0) {
      paths.forEach((path) => (path.selected = false));
      regions.forEach((region) => (region.selected = made.includes(region)));
      updateInspector();
      renderObjectList();
    }
  }

  function renderObjectList() {
    if (!objectList) return;
    objectList.innerHTML = "";
    const canvasRow = document.createElement("button");
    canvasRow.type = "button";
    canvasRow.className = `object-row${selectedPaths().length || selectedRegions().length ? "" : " is-selected"}`;
    canvasRow.innerHTML = `<span class="object-kind">canvas</span><span>whole field</span>`;
    canvasRow.addEventListener("click", () => {
      paths.forEach((path) => (path.selected = false));
      regions.forEach((region) => (region.selected = false));
      updateInspector();
      renderObjectList();
    });
    objectList.appendChild(canvasRow);
    for (const region of regions) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = `object-row${region.selected ? " is-selected" : ""}`;
      row.innerHTML = `<span class="object-kind">region</span><span>${region.name} · ${region.settings.theme}/${region.settings.algorithm}</span>`;
      row.addEventListener("click", (event) => {
        if (!event.shiftKey) {
          paths.forEach((path) => (path.selected = false));
          regions.forEach((item) => (item.selected = false));
        }
        region.selected = event.shiftKey ? !region.selected : true;
        updateInspector();
        renderObjectList();
      });
      objectList.appendChild(row);
    }
    for (const path of paths) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = `object-row${path.selected ? " is-selected" : ""}`;
      row.innerHTML = `<span class="object-kind">path</span><span>${path.type} · ${path.role}</span>`;
      row.addEventListener("click", (event) => {
        if (!event.shiftKey) {
          paths.forEach((item) => (item.selected = false));
          regions.forEach((region) => (region.selected = false));
        }
        path.selected = event.shiftKey ? !path.selected : true;
        updateInspector();
        renderObjectList();
      });
      objectList.appendChild(row);
    }
  }

  function hitPath(point) {
    let best = null;
    let bestDist = 16 / state.zoom;
    for (let i = paths.length - 1; i >= 0; i -= 1) {
      const pts = expandedPoints(paths[i]);
      if (paths[i].type === "pointCloud" || pts.length === 1) {
        for (const pt of pts) {
          const d = Math.hypot(point.x - pt.x, point.y - pt.y);
          if (d < bestDist) {
            bestDist = d;
            best = paths[i];
          }
        }
        continue;
      }
      for (let p = 0; p < pts.length - 1; p += 1) {
        const d = distanceToSegment(point, pts[p], pts[p + 1]);
        if (d < bestDist) {
          bestDist = d;
          best = paths[i];
        }
      }
    }
    return best;
  }

  function hitRegion(point) {
    for (let i = regions.length - 1; i >= 0; i -= 1) {
      if (pointInPolygon(point, regions[i].points)) return regions[i];
    }
    return null;
  }

  function distanceToSegment(p, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len2 = dx * dx + dy * dy || 1;
    const t = clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / len2, 0, 1);
    return Math.hypot(p.x - (a.x + dx * t), p.y - (a.y + dy * t));
  }

  function onPointerDown(event) {
    if (event.target.closest && event.target.closest(".panel, .toolbar, .inspector")) return;
    const raw = screenToWorld(event.clientX, event.clientY);
    const p = snapPoint(raw);
    state.pointerDown = true;
    state.startX = p.x;
    state.startY = p.y;
    state.lastX = p.x;
    state.lastY = p.y;
    state.selectionRect = null;
    state.draft = null;
    if (state.tool === "hand" || event.code === "Space") {
      state.dragMode = "pan";
    } else if (state.tool === "pointer") {
      const hit = hitPath(raw);
      if (hit) {
        if (!event.shiftKey) selectOnly(hit);
        else hit.selected = !hit.selected;
        state.dragMode = "move";
        updateInspector();
        renderObjectList();
      } else {
        const region = hitRegion(raw);
        if (region) {
          if (!event.shiftKey) {
            paths.forEach((path) => (path.selected = false));
            regions.forEach((item) => (item.selected = false));
          }
          region.selected = event.shiftKey ? !region.selected : true;
          state.dragMode = "move";
          updateInspector();
          renderObjectList();
        } else {
          paths.forEach((path) => (path.selected = false));
          regions.forEach((item) => (item.selected = false));
          state.dragMode = "none";
          updateInspector();
          renderObjectList();
        }
      }
    } else if (state.tool === "bucket") {
      const source = closedPathAt(raw);
      if (source) {
        createRegionFromPath(source);
        state.dragMode = "none";
      } else {
        const region = hitRegion(raw);
        if (region) selectRegionOnly(region);
        state.dragMode = "none";
      }
    } else if (state.tool === "select") {
      state.dragMode = "select";
      state.selectionRect = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    } else if (state.tool === "pen") {
      state.dragMode = "draw";
      state.draft = createPath("path", [p]);
    } else {
      state.dragMode = "shape";
      state.draft = createPath(state.tool, [p, p]);
    }
    canvas.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event) {
    const raw = screenToWorld(event.clientX, event.clientY);
    const p = snapPoint(raw);
    if (!state.pointerDown) return;
    const dx = p.x - state.lastX;
    const dy = p.y - state.lastY;
    if (state.dragMode === "pan") {
      state.panX += event.movementX || 0;
      state.panY += event.movementY || 0;
    } else if (state.dragMode === "move") {
      for (const path of paths) {
        if (!path.selected) continue;
        for (const point of path.points) {
          point.x = clamp(point.x + dx, 0, WORLD_W);
          point.y = clamp(point.y + dy, 0, WORLD_H);
        }
      }
      for (const region of regions) {
        if (!region.selected) continue;
        for (const point of region.points) {
          point.x = clamp(point.x + dx, 0, WORLD_W);
          point.y = clamp(point.y + dy, 0, WORLD_H);
        }
        region.bounds = null;
      }
    } else if (state.dragMode === "select" && state.selectionRect) {
      state.selectionRect.x1 = p.x;
      state.selectionRect.y1 = p.y;
    } else if (state.dragMode === "draw" && state.draft) {
      const last = state.draft.points[state.draft.points.length - 1];
      if (Math.hypot(p.x - last.x, p.y - last.y) > Math.max(4, controls.gridSize * 0.25)) state.draft.points.push(p);
    } else if (state.dragMode === "shape" && state.draft) {
      state.draft.points[1] = p;
    }
    state.lastX = p.x;
    state.lastY = p.y;
  }

  function onPointerUp(event) {
    if (state.dragMode === "select" && state.selectionRect) {
      const r = normalizedRect(state.selectionRect);
      paths.forEach((path) => {
        const b = pathBounds(path);
        path.selected = boundsIntersect(b, r);
      });
      regions.forEach((region) => {
        region.selected = boundsIntersect(regionBounds(region), r);
      });
    }
    state.pointerDown = false;
    state.dragMode = "none";
    state.draft = null;
    state.selectionRect = null;
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch (_) {
      // Pointer capture can already be released.
    }
    updateInspector();
    renderObjectList();
  }

  function normalizedRect(rect) {
    return {
      minX: Math.min(rect.x0, rect.x1),
      minY: Math.min(rect.y0, rect.y1),
      maxX: Math.max(rect.x0, rect.x1),
      maxY: Math.max(rect.y0, rect.y1),
    };
  }

  function onWheel(event) {
    if (event.target.closest && event.target.closest(".panel, .inspector")) return;
    event.preventDefault();
    const factor = Math.exp(-event.deltaY * 0.0012);
    setZoom(state.zoom * factor, event.clientX, event.clientY);
    const input = document.getElementById("zoom");
    if (input) input.value = String(state.zoom);
  }

  function onKeyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "c") {
      event.preventDefault();
      copySelected();
    } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "v") {
      event.preventDefault();
      pasteClipboard();
    } else if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      deleteSelected();
    } else if (event.key === "0") {
      state.zoom = Math.max(window.innerWidth / WORLD_W, window.innerHeight / WORLD_H);
      state.panX = (window.innerWidth - WORLD_W * state.zoom) * 0.5;
      state.panY = (window.innerHeight - WORLD_H * state.zoom) * 0.5;
      const input = document.getElementById("zoom");
      if (input) input.value = String(state.zoom);
    }
  }

  function copySelected() {
    clipboard = [
      ...selectedPaths().map((path) => ({ kind: "path", value: clonePath(path) })),
      ...selectedRegions().map((region) => ({ kind: "region", value: cloneRegion(region) })),
    ];
  }

  function pasteClipboard() {
    paths.forEach((path) => (path.selected = false));
    regions.forEach((region) => (region.selected = false));
    for (const source of clipboard) {
      if (source.kind === "region") {
        const region = cloneRegion(source.value);
        region.id = nextRegionId++;
        region.name = `region ${region.id}`;
        region.selected = true;
        for (const point of region.points) {
          point.x = clamp(point.x + controls.gridSize, 0, WORLD_W);
          point.y = clamp(point.y + controls.gridSize, 0, WORLD_H);
        }
        regions.push(region);
        continue;
      }
      const path = clonePath(source.value);
      path.id = nextPathId++;
      path.selected = true;
      for (const point of path.points) {
        point.x = clamp(point.x + controls.gridSize, 0, WORLD_W);
        point.y = clamp(point.y + controls.gridSize, 0, WORLD_H);
      }
      paths.push(path);
    }
    updateInspector();
    renderObjectList();
  }

  function clonePath(path) {
    return {
      ...path,
      points: path.points.map((point) => ({ ...point })),
      selected: path.selected,
    };
  }

  function cloneRegion(region) {
    return {
      ...region,
      points: region.points.map((point) => ({ ...point })),
      settings: { ...region.settings },
      selected: region.selected,
      bounds: null,
    };
  }

  function deleteSelected() {
    for (let i = paths.length - 1; i >= 0; i -= 1) {
      if (paths[i].selected) paths.splice(i, 1);
    }
    for (let i = regions.length - 1; i >= 0; i -= 1) {
      if (regions[i].selected) regions.splice(i, 1);
    }
    updateInspector();
    renderObjectList();
  }

  function randomizeSubstrate() {
    state.seed = Math.floor(Math.random() * 999999);
    for (const key of ["density", "motion", "pixel", "groupScale", "flow", "bit", "quant", "snap", "threshold", "memory", "contour", "weave", "scanline", "glitch", "contrast"]) {
      const input = document.getElementById(key);
      if (!input) continue;
      const min = Number(input.min);
      const max = Number(input.max);
      const value = input.step === "1" ? Math.round(lerp(min, max, Math.random())) : lerp(min, max, Math.random());
      controls[key] = value;
      input.value = String(value);
    }
  }

  function exportPng() {
    const link = document.createElement("a");
    link.download = `quantized-v3-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  function importSvg(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => imprintSvg(String(reader.result || ""));
    reader.readAsText(file);
    event.target.value = "";
  }

  function imprintSvg(svgText) {
    const intrinsic = svgIntrinsicSize(svgText);
    const blob = new Blob([svgText], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = SUB_W;
      maskCanvas.height = SUB_H;
      const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
      maskCtx.fillStyle = "#000";
      maskCtx.fillRect(0, 0, SUB_W, SUB_H);
      const iw = img.naturalWidth || img.width || intrinsic.width;
      const ih = img.naturalHeight || img.height || intrinsic.height;
      const fit = Math.min(SUB_W / iw, SUB_H / ih) * 0.82;
      const dw = iw * fit;
      const dh = ih * fit;
      maskCtx.drawImage(img, (SUB_W - dw) * 0.5, (SUB_H - dh) * 0.5, dw, dh);
      applySvgMask(maskCtx.getImageData(0, 0, SUB_W, SUB_H).data);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }

  function svgIntrinsicSize(svgText) {
    const fallback = { width: SUB_W, height: SUB_H };
    try {
      const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
      const svg = doc.documentElement;
      const viewBox = svg.getAttribute("viewBox");
      if (viewBox) {
        const parts = viewBox.split(/[\s,]+/).map(Number);
        if (parts.length >= 4 && parts[2] > 0 && parts[3] > 0) return { width: parts[2], height: parts[3] };
      }
      const width = Number.parseFloat(svg.getAttribute("width"));
      const height = Number.parseFloat(svg.getAttribute("height"));
      if (width > 0 && height > 0) return { width, height };
    } catch (_) {
      return fallback;
    }
    return fallback;
  }

  function applySvgMask(data) {
    const mode = controls.svgMode;
    const strength = controls.svgForce;
    const path = {
      id: nextPathId++,
      type: "pointCloud",
      points: [],
      role: mode,
      strength,
      pixelation: controls.pathPixelation,
      radius: controls.pathRadius,
      frequency: controls.pathFrequency,
      speed: controls.pathSpeed,
      color: controls.pathColor,
      selected: true,
    };
    for (let y = 1; y < SUB_H - 1; y += 2) {
      for (let x = 1; x < SUB_W - 1; x += 2) {
        const i = (y * SUB_W + x) * 4;
        const value = Math.max(data[i], data[i + 1], data[i + 2], data[i + 3]) / 255;
        if (value > 0.25 && hash2(x, y, state.seed) > 0.72) path.points.push({ x: (x / SUB_W) * WORLD_W, y: (y / SUB_H) * WORLD_H });
      }
    }
    if (path.points.length > 1) {
      paths.forEach((item) => (item.selected = false));
      regions.forEach((item) => (item.selected = false));
      paths.push(path);
      updateInspector();
      renderObjectList();
    }
  }

  function stampPathFields() {
    influence.fill(0);
    quantField.fill(0);
    tearField.fill(0);
    pigmentR.fill(0);
    pigmentG.fill(0);
    pigmentB.fill(0);
    flowX.fill(0);
    flowY.fill(0);
    waveField.fill(0);
    turbulenceField.fill(0);
    phaseField.fill(0);
    damField.fill(0);
    pixelationField.fill(0);
    for (const path of paths) stampPath(path);
  }

  function stampPath(path) {
    const points = expandedPoints(path);
    if (path.type === "pointCloud") {
      for (const point of points) stampPoint(path, point, point);
      return;
    }
    if (points.length === 1) stampPoint(path, points[0], points[0]);
    for (let i = 0; i < points.length - 1; i += 1) stampSegment(path, points[i], points[i + 1]);
  }

  function stampSegment(path, a, b) {
    const dist = Math.hypot(b.x - a.x, b.y - a.y);
    const steps = Math.max(1, Math.ceil(dist / Math.max(4, path.radius * 0.45)));
    for (let i = 0; i <= steps; i += 1) {
      const p = i / steps;
      stampPoint(path, { x: lerp(a.x, b.x, p), y: lerp(a.y, b.y, p) }, b);
    }
  }

  function stampPoint(path, point, next) {
    const sx = Math.round((point.x / WORLD_W) * SUB_W);
    const sy = Math.round((point.y / WORLD_H) * SUB_H);
    const r = Math.max(1, Math.ceil((path.radius / WORLD_W) * SUB_W));
    const dx = next.x - point.x;
    const dy = next.y - point.y;
    const len = Math.hypot(dx, dy);
    const dirX = len > 0.001 ? dx / len : Math.cos(path.id * 2.399);
    const dirY = len > 0.001 ? dy / len : Math.sin(path.id * 2.399);
    const normalX = -dirY;
    const normalY = dirX;
    const perturb = path.strength;
    const pixelation = path.pixelation ?? 0.08;
    const frequency = path.frequency || 1;
    const speed = path.speed || 1;
    for (let y = sy - r; y <= sy + r; y += 1) {
      if (y < 0 || y >= SUB_H) continue;
      for (let x = sx - r; x <= sx + r; x += 1) {
        if (x < 0 || x >= SUB_W) continue;
        const d = Math.hypot(x - sx, y - sy);
        if (d > r) continue;
        const baseFalloff = Math.pow(1 - d / r, 1.8);
        const forceFalloff = baseFalloff * perturb;
        const visualFalloff = baseFalloff * pixelation;
        const idx = y * SUB_W + x;
        pixelationField[idx] += Math.abs(visualFalloff);
        const radialX = d > 0.001 ? (x - sx) / d : normalX;
        const radialY = d > 0.001 ? (y - sy) / d : normalY;
        if (path.role === "current") {
          flowX[idx] += dirX * forceFalloff * 1.3;
          flowY[idx] += dirY * forceFalloff * 1.3;
          phaseField[idx] += forceFalloff * 0.09;
          influence[idx] += Math.abs(visualFalloff) * 0.05;
        } else if (path.role === "wave") {
          const wave = Math.sin((d / Math.max(1, r)) * TAU * 1.35 * frequency - state.time * speed * (2.1 + Math.abs(perturb) * 0.45) + path.id * 0.61) * forceFalloff;
          flowX[idx] += radialX * wave * 1.1;
          flowY[idx] += radialY * wave * 1.1;
          waveField[idx] += wave;
          phaseField[idx] += wave * 0.38;
          influence[idx] += Math.abs(visualFalloff) * 0.05;
        } else if (path.role === "turbulence") {
          const n = valueNoise(x + state.time * 28 * speed + path.id * 17, y - state.time * 19 * speed, 5.5 / Math.sqrt(frequency), 7601);
          const angle = n * TAU * 2 + perturb * 0.7;
          flowX[idx] += Math.cos(angle) * forceFalloff * 1.35;
          flowY[idx] += Math.sin(angle) * forceFalloff * 1.35;
          turbulenceField[idx] += Math.abs(forceFalloff) * 0.85;
          phaseField[idx] += (n - 0.5) * forceFalloff;
        } else if (path.role === "eddy") {
          flowX[idx] += -radialY * forceFalloff * 1.45;
          flowY[idx] += radialX * forceFalloff * 1.45;
          waveField[idx] += Math.sin(d * 0.62 * frequency - state.time * 1.8 * speed + path.id) * forceFalloff * 0.28;
          influence[idx] += Math.abs(visualFalloff) * 0.04;
        } else if (path.role === "pressure") {
          flowX[idx] += radialX * forceFalloff * 1.25;
          flowY[idx] += radialY * forceFalloff * 1.25;
          waveField[idx] += Math.cos((d / Math.max(1, r)) * TAU * frequency - state.time * 1.25 * speed) * forceFalloff * 0.2;
          influence[idx] += visualFalloff * 0.05;
        } else if (path.role === "shear") {
          const band = Math.sin((x * normalX + y * normalY) * 0.35 * frequency + state.time * 2.1 * speed + path.id);
          flowX[idx] += dirX * band * forceFalloff * 1.7;
          flowY[idx] += dirY * band * forceFalloff * 1.7;
          phaseField[idx] += band * forceFalloff * 0.5;
          turbulenceField[idx] += Math.abs(band * forceFalloff) * 0.28;
        } else if (path.role === "dam") {
          damField[idx] = Math.max(damField[idx], Math.abs(forceFalloff));
          phaseField[idx] -= forceFalloff * 0.18;
        } else if (path.role === "phase") {
          const pulse = Math.sin((x * 0.17 + y * 0.11) * frequency + state.time * speed * (1.2 + Math.abs(perturb)) + path.id);
          phaseField[idx] += pulse * forceFalloff;
          flowX[idx] += normalX * pulse * forceFalloff * 0.55;
          flowY[idx] += normalY * pulse * forceFalloff * 0.55;
        } else if (path.role === "quantize") {
          quantField[idx] = Math.max(quantField[idx], Math.abs(visualFalloff));
        } else if (path.role === "tear") {
          tearField[idx] = Math.max(tearField[idx], Math.abs(visualFalloff));
          const tearWave = Math.sin((y - sy) * 0.9 * frequency + state.time * 8.0 * speed + hash2(x, y, path.id) * TAU) * forceFalloff;
          flowX[idx] += tearWave * 1.15;
          phaseField[idx] += tearWave * 0.32;
          turbulenceField[idx] += Math.abs(tearWave) * 0.2;
        }
      }
    }
  }

  function controlsAtCell(x, y) {
    if (regions.length === 0) return controls;
    const world = { x: (x / SUB_W) * WORLD_W, y: (y / SUB_H) * WORLD_H };
    let local = { ...controls, themeObject: frameBaseTheme || normalizeTheme(controls.theme) };
    let strongest = 0;
    for (const region of regions) {
      const amount = regionMaskAt(world, region);
      if (amount <= 0) continue;
      for (const key of numericSubstrateKeys) local[key] = lerp(local[key], region.settings[key], amount);
      local.themeObject = blendThemeObjects(local.themeObject, normalizeTheme(region.settings.theme), amount);
      if (amount >= strongest) {
        strongest = amount;
        local.theme = region.settings.theme;
        local.algorithm = region.settings.algorithm;
      }
    }
    return local;
  }

  function drawSubstrate(t) {
    const data = image.data;
    frameBaseTheme = normalizeTheme(controls.theme);
    for (let y = 0; y < SUB_H; y += 1) {
      for (let x = 0; x < SUB_W; x += 1) {
        const i = y * SUB_W + x;
        const idx = i * 4;
        const local = controlsAtCell(x, y);
        const theme = local.themeObject || frameBaseTheme;
        const baseAlg = algorithms[local.algorithm] || algorithms.god;
        const algorithm = {
          threshold: baseAlg.threshold + local.threshold,
          snap: local.snap,
          group: baseAlg.group * local.groupScale,
          bit: local.bit,
          flow: baseAlg.flow * local.flow,
          quant: local.quant || baseAlg.quant,
        };
        const pathPixelation = clamp(pixelationField[i], 0, 3);
        const dam = clamp(damField[i], 0, 1.85);
        const wave = waveField[i];
        const turbulence = clamp(turbulenceField[i], 0, 3.2);
        const phase = phaseField[i];
        const carriedFlowX = flowX[i];
        const carriedFlowY = flowY[i];
        const localFlowX = carriedFlowX * (1 - dam * 0.55) + Math.sin(phase * 3.1 + wave) * wave * 0.24;
        const localFlowY = carriedFlowY * (1 - dam * 0.55) + Math.cos(phase * 2.7 - wave) * wave * 0.24;
        const motion = clamp(local.motion * (1 - dam * 0.26 + turbulence * 0.08), 0.02, 4.2);
        const density = local.density;
        const pixel = clamp(local.pixel, 0.2, 12);
        const snap = Math.max(1, Math.round(algorithm.snap * pixel));
        const group = Math.max(snap, algorithm.group * pixel);
        const sx = Math.floor(x / snap) * snap;
        const sy = Math.floor(y / snap) * snap;
        const gx = Math.floor(x / group) * group;
        const gy = Math.floor(y / group) * group;
        const effectiveFlow = algorithm.flow * (1 - dam * 0.5) + Math.abs(wave) * 0.11 + turbulence * 0.06;
        const domain = valueNoise(gx + t * motion * 4.4 + localFlowX * 16 + phase * 7, gy - t * motion * 3.2 + localFlowY * 16 - phase * 5, 22, 5151);
        const a = valueNoise(
          sx + Math.sin(t * 0.13 * motion + domain + phase * 0.7) * 42 * effectiveFlow + localFlowX * 38,
          sy - t * (7 + motion * 10) * effectiveFlow + localFlowY * 38 + wave * 10,
          13,
          9101,
        );
        const b = valueNoise(
          sx * 0.78 - t * (9 + motion * 14) * effectiveFlow + localFlowX * 18,
          sy * 0.78 + Math.cos(t * 0.17 + domain + phase) * 37 * effectiveFlow + localFlowY * 18,
          8.5,
          3127,
        );
        const c = valueNoise(
          sx + Math.sin(sy * 0.045 + t * 0.9 * motion * effectiveFlow + phase) * 26 * effectiveFlow,
          sy + Math.cos(sx * 0.033 - t * 0.72 * motion * effectiveFlow - phase) * 26 * effectiveFlow,
          4.7,
          6401,
        );
        const memory = clamp(local.memory + dam * 0.035 - turbulence * 0.018, 0.82, 0.99);
        fieldB[i] =
          fieldA[i] * memory +
          (a * 0.36 + b * 0.31 + c * 0.33 + influence[i] * 0.05 + Math.hypot(localFlowX, localFlowY) * 0.018 + Math.abs(wave) * 0.018) *
            (1 - memory + motion * 0.02 + turbulence * 0.004);
        const vein = Math.sin((a - b) * TAU * 3.4 + c * 5.8 + t * (0.85 + motion * 0.9) + phase);
        const groupNoise = valueNoise(gx - t * motion * 2.7 + localFlowX * 9, gy + t * motion * 2.1 + localFlowY * 9, 23, 9901);
        const groupX = Math.floor(gx / group);
        const groupY = Math.floor(gy / group);
        const bitLevels = 2 + Math.floor(clamp(algorithm.bit, 0, 1) * 14);
        const bitIndex = ((Math.floor(x / snap) & Math.floor(y / snap)) ^ Math.floor(groupX * 3 + groupY * 5) ^ Math.floor(algorithm.bit * 255)) & 31;
        const bitplane = (bitIndex % bitLevels) / Math.max(1, bitLevels - 1);
        const grain = hash2(Math.floor(x / 2), Math.floor(y / 2), Math.floor(t * 1.2));
        let shape =
          fieldB[i] * 0.42 +
          (vein * 0.5 + 0.5) * 0.24 +
          c * 0.14 +
          groupNoise * 0.2 +
          bitplane * algorithm.bit +
          influence[i] * 0.035 * pathPixelation +
          wave * 0.012 * pathPixelation +
          (valueNoise(x + t * 24, y - t * 17, 3.2, 4455) - 0.5) * turbulence * 0.028 * pathPixelation;
        if (local.algorithm === "cellular") {
          const tick = Math.floor(t * (0.8 + motion * 0.55));
          const n = hash2(groupX, groupY, tick) + hash2(groupX + 1, groupY, tick) + hash2(groupX - 1, groupY, tick) + hash2(groupX, groupY + 1, tick) + hash2(groupX, groupY - 1, tick);
          shape = fieldB[i] * 0.28 + groupNoise * 0.28 + (n > 2.45 ? 1 : n > 1.92 ? 0.62 : 0.12) * 0.34 + (vein * 0.5 + 0.5) * 0.1;
        } else if (local.algorithm === "scanline") {
          shape = fieldB[i] * 0.3 + groupNoise * 0.22 + (((Math.floor(y / snap) + Math.floor(t * (5 + motion * 4))) & 7) / 7) * 0.22 + bitplane * 0.08;
        } else if (local.algorithm === "lattice") {
          shape = fieldB[i] * 0.24 + groupNoise * 0.24 + Math.max(Math.abs(Math.sin(sx * 0.045 + t * motion * 0.8)), Math.abs(Math.sin(sy * 0.052 - t * motion * 0.65))) * 0.36 + bitplane * 0.16;
        } else if (local.algorithm === "interference") {
          const w0 = Math.sin(Math.hypot(x - SUB_W * 0.25, y - SUB_H * 0.72) * 0.12 - t * (2 + motion));
          const w1 = Math.sin(Math.hypot(x - SUB_W * 0.79, y - SUB_H * 0.58) * 0.1 + t * (1.6 + motion * 0.8));
          shape = fieldB[i] * 0.28 + groupNoise * 0.22 + ((w0 + w1) * 0.25 + 0.5) * 0.34 + (vein * 0.5 + 0.5) * 0.16;
        }
        shape += Math.sin(shape * TAU * 5 + localFlowX * 2 + phase) * local.contour * 0.04;
        shape += (Math.sin(x * 0.28 + localFlowY * 4) + Math.sin(y * 0.31 + localFlowX * 4)) * local.weave * 0.035;
        shape += (((y + Math.floor(t * 12)) & 7) / 7 - 0.5) * local.scanline * 0.08;
        if (hash2(Math.floor(y / 3), Math.floor(t * 7), 919) > 0.92) shape += (hash2(x, y, Math.floor(t * 4)) - 0.5) * local.glitch * 0.22;
        if (algorithm.quant > 0 || quantField[i] > 0.01) {
          const levels = 5 + algorithm.bit * 10 + algorithm.quant * 4 + quantField[i] * 14;
          shape = Math.floor(shape * levels) / levels;
        }
        if (tearField[i] > 0.01) shape = lerp(shape, hash2(Math.floor(x / 4), y, Math.floor(t * 12)), clamp(tearField[i] * 0.8, 0, 0.85));
        shape = (shape - 0.5) * local.contrast + 0.5;
        const threshold = 0.43 + algorithm.threshold - density * 0.045;
        let color = theme.base;
        if (shape > threshold || grain > 0.965 - density * 0.035) {
          const slot = Math.floor(Math.abs(shape * 9 + groupNoise * 5 + hash2(groupX, groupY, 8080) * 4) % theme.colors.length);
          const pc = themeColor(theme, slot);
          const mix = clamp(((shape - threshold) * 1.35 + 0.2 + influence[i] * 0.16) * theme.mix, 0.14, 0.94);
          color = [lerp(theme.base[0], pc[0], mix), lerp(theme.base[1], pc[1], mix), lerp(theme.base[2], pc[2], mix)];
        }
        if (pigmentR[i] || pigmentG[i] || pigmentB[i]) {
          const amount = clamp(Math.abs(influence[i]) * 0.7 + pigmentR[i] + pigmentG[i] + pigmentB[i], 0, 0.85);
          color = [lerp(color[0], pigmentR[i] * 255, amount), lerp(color[1], pigmentG[i] * 255, amount), lerp(color[2], pigmentB[i] * 255, amount)];
        }
        hiddenR[i] = color[0];
        hiddenG[i] = color[1];
        hiddenB[i] = color[2];
        hiddenShape[i] = shape;
        data[idx] = clamp(Math.floor(color[0]), 0, 255);
        data[idx + 1] = clamp(Math.floor(color[1]), 0, 255);
        data[idx + 2] = clamp(Math.floor(color[2]), 0, 255);
        data[idx + 3] = 255;
      }
    }
    fieldA.set(fieldB);
    if (stippleIsActive()) applyStipplePass(t);
    substrateCtx.putImageData(image, 0, 0);
  }

  function stippleIsActive() {
    if (controls.stipple > 0) return true;
    return regions.some((region) => region.settings && region.settings.stipple > 0);
  }

  function averageHiddenCell(x0, y0, x1, y1, sampleBlur) {
    let r = 0;
    let g = 0;
    let b = 0;
    let shape = 0;
    let count = 0;
    const step = sampleBlur > 3 ? 2 : 1;
    for (let yy = y0; yy < y1; yy += step) {
      for (let xx = x0; xx < x1; xx += step) {
        const ii = yy * SUB_W + xx;
        r += hiddenR[ii];
        g += hiddenG[ii];
        b += hiddenB[ii];
        shape += hiddenShape[ii];
        count += 1;
      }
    }
    const inv = count > 0 ? 1 / count : 1;
    return { r: r * inv, g: g * inv, b: b * inv, shape: shape * inv };
  }

  function applyStipplePass(t) {
    const data = image.data;
    const cellCache = new Map();
    for (let y = 0; y < SUB_H; y += 1) {
      for (let x = 0; x < SUB_W; x += 1) {
        const i = y * SUB_W + x;
        const idx = i * 4;
        const local = controlsAtCell(x, y);
        if (local.stipple <= 0) continue;
        const theme = local.themeObject || frameBaseTheme || normalizeTheme(local.theme);
        const cell = Math.max(2, Math.round(2 + local.stipple * 3.2));
        const cx = Math.floor(x / cell);
        const cy = Math.floor(y / cell);
        const key = `${cell}:${cx}:${cy}:${local.blur.toFixed(2)}:${local.exposure.toFixed(2)}:${local.stipple.toFixed(2)}:${local.theme}`;
        let dot = cellCache.get(key);
        if (!dot) {
          const x0 = cx * cell;
          const y0 = cy * cell;
          const x1 = Math.min(SUB_W, x0 + cell);
          const y1 = Math.min(SUB_H, y0 + cell);
          const avg = averageHiddenCell(x0, y0, x1, y1, local.blur);
          const cellHash = hash2(cx, cy, Math.floor(local.stipple * 101));
          const exposure = clamp(avg.shape + local.exposure * 0.28 + (cellHash - 0.5) * 0.18, 0, 1);
          const radius = cell * clamp(0.08 + Math.pow(exposure, 0.82) * 0.54, 0.06, 0.62);
          const jitterX = (hash2(cx, cy, 173) - 0.5) * cell * 0.34;
          const jitterY = (hash2(cx, cy, 941) - 0.5) * cell * 0.34;
          let dotR = 0;
          let dotG = 0;
          let dotB = 0;
          let dotCount = 0;
          for (let yy = y0; yy < y1; yy += 1) {
            for (let xx = x0; xx < x1; xx += 1) {
              const lx = (xx % cell) - cell * 0.5 - jitterX;
              const ly = (yy % cell) - cell * 0.5 - jitterY;
              if (Math.hypot(lx, ly) > radius) continue;
              const ii = yy * SUB_W + xx;
              dotR += hiddenR[ii];
              dotG += hiddenG[ii];
              dotB += hiddenB[ii];
              dotCount += 1;
            }
          }
          const dotInv = dotCount > 0 ? 1 / dotCount : 1;
          const fiber = hash2(cx, cy, 2217) - 0.5;
          const paper = theme.glint || [238, 242, 226];
          const groundAmount = clamp(0.1 + local.exposure * 0.22 + fiber * 0.08, 0, 0.34);
          const shadowAmount = clamp((1 - exposure) * 0.18, 0, 0.22);
          dot = {
            radius,
            jitterX,
            jitterY,
            shadowAmount,
            ground: [
              lerp(theme.base[0], paper[0], groundAmount),
              lerp(theme.base[1], paper[1], groundAmount),
              lerp(theme.base[2], paper[2], groundAmount),
            ],
            ink: [
              lerp(dotR * dotInv || avg.r, theme.base[0], shadowAmount),
              lerp(dotG * dotInv || avg.g, theme.base[1], shadowAmount),
              lerp(dotB * dotInv || avg.b, theme.base[2], shadowAmount),
            ],
          };
          cellCache.set(key, dot);
        }
        const lx = (x % cell) - cell * 0.5 - dot.jitterX;
        const ly = (y % cell) - cell * 0.5 - dot.jitterY;
        const inside = Math.hypot(lx, ly) <= dot.radius;
        const blurMix = clamp(local.blur / 5, 0, 1);
        const pixelInk = [
          lerp(hiddenR[i], theme.base[0], dot.shadowAmount),
          lerp(hiddenG[i], theme.base[1], dot.shadowAmount),
          lerp(hiddenB[i], theme.base[2], dot.shadowAmount),
        ];
        const ink = [
          lerp(pixelInk[0], dot.ink[0], blurMix),
          lerp(pixelInk[1], dot.ink[1], blurMix),
          lerp(pixelInk[2], dot.ink[2], blurMix),
        ];
        const color = inside ? ink : dot.ground;
        data[idx] = clamp(Math.floor(color[0]), 0, 255);
        data[idx + 1] = clamp(Math.floor(color[1]), 0, 255);
        data[idx + 2] = clamp(Math.floor(color[2]), 0, 255);
      }
    }
  }

  function render(timestamp) {
    const t = timestamp * 0.001;
    state.time = t;
    state.frame += 1;
    stampPathFields();
    drawSubstrate(t);
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    ctx.fillStyle = "#03020b";
    ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
    ctx.save();
    ctx.setTransform(state.dpr * state.zoom, 0, 0, state.dpr * state.zoom, state.dpr * state.panX, state.dpr * state.panY);
    ctx.imageSmoothingEnabled = false;
    const canvasBlur = stippleIsActive() ? 0 : clamp(controls.blur, 0, 5);
    ctx.filter = canvasBlur > 0 ? `blur(${canvasBlur}px)` : "none";
    ctx.drawImage(substrateCanvas, 0, 0, WORLD_W, WORLD_H);
    ctx.filter = "none";
    if (controls.showGrid) {
      drawGrid();
      drawRegions();
      drawPaths();
      drawSelectionRect();
    }
    ctx.restore();
    requestAnimationFrame(render);
  }

  function drawGrid() {
    const g = controls.gridSize;
    ctx.save();
    for (let x = 0; x <= WORLD_W; x += g) {
      const major = Math.round(x / g) % 4 === 0;
      ctx.lineWidth = (major ? 1.5 : 0.85) / state.zoom;
      ctx.strokeStyle = major ? "rgba(255,255,255,0.32)" : "rgba(255,255,255,0.2)";
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, WORLD_H);
      ctx.stroke();
    }
    for (let y = 0; y <= WORLD_H; y += g) {
      const major = Math.round(y / g) % 4 === 0;
      ctx.lineWidth = (major ? 1.5 : 0.85) / state.zoom;
      ctx.strokeStyle = major ? "rgba(255,255,255,0.32)" : "rgba(255,255,255,0.2)";
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(WORLD_W, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawRegions() {
    ctx.save();
    for (const region of regions) {
      if (region.points.length < 2) continue;
      ctx.globalAlpha = region.selected ? 0.42 : 0.2;
      ctx.fillStyle = region.selected ? "#d4ff15" : "#00d6ff";
      ctx.strokeStyle = region.selected ? "#d4ff15" : "#00d6ff";
      ctx.lineWidth = (region.selected ? 2 : 1.25) / state.zoom;
      ctx.beginPath();
      ctx.moveTo(region.points[0].x, region.points[0].y);
      for (let i = 1; i < region.points.length; i += 1) ctx.lineTo(region.points[i].x, region.points[i].y);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = region.selected ? 0.95 : 0.62;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawPaths() {
    ctx.save();
    ctx.lineWidth = 2 / state.zoom;
    for (const path of paths) {
      const pts = expandedPoints(path);
      if (pts.length === 0) continue;
      ctx.strokeStyle = path.selected ? "#d4ff15" : path.color;
      ctx.globalAlpha = path.selected ? 0.95 : 0.58;
      if (path.type === "pointCloud") {
        const size = Math.max(2 / state.zoom, Math.min(path.radius * 0.12, 8 / state.zoom));
        ctx.fillStyle = ctx.strokeStyle;
        for (const point of pts) ctx.fillRect(point.x - size * 0.5, point.y - size * 0.5, size, size);
      } else {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
      }
      if (path.selected) {
        const b = pathBounds(path);
        ctx.setLineDash([6 / state.zoom, 4 / state.zoom]);
        ctx.strokeRect(b.minX, b.minY, b.maxX - b.minX, b.maxY - b.minY);
        ctx.setLineDash([]);
      }
    }
    ctx.restore();
  }

  function drawSelectionRect() {
    if (!state.selectionRect) return;
    const r = normalizedRect(state.selectionRect);
    ctx.save();
    ctx.lineWidth = 1 / state.zoom;
    ctx.strokeStyle = "#00d6ff";
    ctx.setLineDash([8 / state.zoom, 5 / state.zoom]);
    ctx.strokeRect(r.minX, r.minY, r.maxX - r.minX, r.maxY - r.minY);
    ctx.restore();
  }

  for (let i = 0; i < N; i += 1) fieldA[i] = hash2(i, i >>> 4, state.seed);
  setupControls();
  resize();
  window.addEventListener("resize", resize);
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  window.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKeyDown);
  requestAnimationFrame(render);
})();
