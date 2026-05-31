(() => {
  "use strict";

  const canvas = document.getElementById("quantized");
  const ctx = canvas.getContext("2d", { alpha: false });
  const panel = document.getElementById("panel");
  const panelToggle = document.getElementById("panelToggle");
  const resetParams = document.getElementById("resetParams");

  const WORLD_W = 1920;
  const WORLD_H = 1080;
  const SUB_W = 480;
  const SUB_H = 270;
  const seedText =
    "Quantized / emergence of discrete structure from noisy probability spaces / human + mechanism";

  const TAU = Math.PI * 2;
  const palette = [
    [153, 115, 246],
    [168, 255, 0],
    [0, 180, 88],
    [0, 214, 255],
    [246, 48, 176],
    [255, 91, 27],
    [3, 36, 211],
    [255, 238, 82],
    [232, 124, 96],
    [16, 104, 58],
    [0, 0, 0],
  ];

  const state = {
    dpr: 1,
    zoom: 1,
    panX: 0,
    panY: 0,
    pointerX: WORLD_W * 0.5,
    pointerY: WORLD_H * 0.5,
    pointerDown: false,
    dragX: 0,
    dragY: 0,
    perturb: 0,
    keyboardPhase: 0,
    frame: 0,
    lastTime: 0,
  };

  const defaults = {
    motion: 1.35,
    grain: 1,
    blur: 0,
    mutation: 1,
    raySpeed: 1,
    density: 1,
  };

  const controls = { ...defaults };

  const substrateCanvas = document.createElement("canvas");
  substrateCanvas.width = SUB_W;
  substrateCanvas.height = SUB_H;
  const substrateCtx = substrateCanvas.getContext("2d", { alpha: false });
  const substrateImage = substrateCtx.createImageData(SUB_W, SUB_H);
  const fieldA = new Float32Array(SUB_W * SUB_H);
  const fieldB = new Float32Array(SUB_W * SUB_H);

  const hashSeed = xfnv1a(seedText);
  const rand = mulberry32(hashSeed);
  const blocks = makeBlocks();
  const artifacts = makeArtifacts();
  const rayFamilies = makeRayFamilies();
  const gliders = makeGliders();

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

  function fractalNoise(x, y, t, salt) {
    const n0 = valueNoise(x + t * 3.1, y - t * 2.4, 19, salt);
    const n1 = valueNoise(x - t * 6.2, y + t * 4.7, 8, salt + 101);
    const n2 = valueNoise(x + t * 9.3, y + t * 5.1, 3.5, salt + 211);
    return n0 * 0.34 + n1 * 0.31 + n2 * 0.35;
  }

  function nearestColor(r, g, b) {
    let best = palette[0];
    let bestD = Infinity;
    for (const c of palette) {
      const dr = r - c[0];
      const dg = g - c[1];
      const db = b - c[2];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    return best;
  }

  function makeBlocks() {
    const out = [];
    for (let i = 0; i < 48; i += 1) {
      const s = rand();
      const vertical = rand() < 0.18;
      const cell = [4, 6, 8, 12, 16, 24, 32][Math.floor(rand() * 7)];
      const x = Math.floor((rand() * WORLD_W) / cell) * cell;
      const y = Math.floor((rand() * WORLD_H) / cell) * cell;
      out.push({
        x,
        y,
        w: Math.floor(lerp(28, vertical ? 92 : 420, rand() * rand())),
        h: Math.floor(lerp(22, vertical ? 520 : 250, s * rand())),
        phase: rand() * TAU,
        rate: lerp(0.018, 0.12, rand()),
        duty: lerp(0.82, 0.985, rand()),
        anchored: rand() < 0.92,
        hop: [0, 0, 0, 8, 16][Math.floor(rand() * 5)],
        colorA: palette[Math.floor(rand() * palette.length)],
        colorB: palette[Math.floor(rand() * palette.length)],
        cell,
        mode: Math.floor(rand() * 7),
      });
    }
    return out;
  }

  function makeArtifacts() {
    const out = [];
    for (let i = 0; i < 26; i += 1) {
      const cell = [4, 6, 8, 12, 16][Math.floor(rand() * 5)];
      out.push({
        x: Math.floor((rand() * WORLD_W) / cell) * cell,
        y: Math.floor((rand() * WORLD_H) / cell) * cell,
        w: Math.floor(lerp(80, 520, rand()) / cell) * cell,
        h: Math.floor(lerp(24, 220, rand()) / cell) * cell,
        cell,
        phase: rand() * TAU,
        rate: lerp(0.018, 0.09, rand()),
        kind: Math.floor(rand() * 5),
        a: palette[Math.floor(rand() * palette.length)],
        b: palette[Math.floor(rand() * palette.length)],
      });
    }
    return out;
  }

  function makeRayFamilies() {
    const out = [];
    const origins = [
      [0.32, 0.47],
      [0.66, 0.31],
      [0.76, 0.68],
      [0.18, 0.74],
    ];
    for (let f = 0; f < origins.length; f += 1) {
      const rays = [];
      for (let i = 0; i < 18; i += 1) {
        const z = lerp(-1, 1, rand());
        const a = rand() * TAU;
        const r = Math.sqrt(1 - z * z);
        rays.push({
          x: Math.cos(a) * r,
          y: Math.sin(a) * r,
          z,
          length: lerp(520, 1900, rand()),
          step: [2, 3, 4, 5, 7][Math.floor(rand() * 5)],
          phase: rand() * TAU,
          color: rand() < 0.72 ? [0, 44, 205] : rand() < 0.5 ? [0, 0, 0] : [0, 214, 255],
        });
      }
      out.push({
        originX: origins[f][0] * WORLD_W,
        originY: origins[f][1] * WORLD_H,
        axis: normalize3(lerp(-1, 1, rand()), lerp(-1, 1, rand()), lerp(-1, 1, rand())),
        speed: lerp(-0.22, 0.24, rand()),
        phase: rand() * TAU,
        projection: lerp(0.62, 1.08, rand()),
        rays,
      });
    }
    return out;
  }

  function normalize3(x, y, z) {
    const mag = Math.hypot(x, y, z) || 1;
    return { x: x / mag, y: y / mag, z: z / mag };
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

  function makeColonies() {
    const out = [];
    for (let i = 0; i < 16; i += 1) {
      out.push({
        x: rand() * WORLD_W,
        y: rand() * WORLD_H,
        w: lerp(48, 210, rand()),
        h: lerp(36, 150, rand()),
        phase: rand() * TAU,
        scale: [3, 4, 6, 8][Math.floor(rand() * 4)],
        salt: Math.floor(rand() * 999999),
      });
    }
    return out;
  }

  function makeGliders() {
    const out = [];
    for (let i = 0; i < 19; i += 1) {
      out.push({
        x: rand() * WORLD_W,
        y: rand() * WORLD_H,
        vx: lerp(-46, 64, rand()),
        vy: lerp(-36, 42, rand()),
        phase: rand() * TAU,
        cell: [4, 5, 6, 8][Math.floor(rand() * 4)],
        color: palette[Math.floor(rand() * palette.length)],
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
    if (drawnW <= window.innerWidth) {
      state.panX = (window.innerWidth - drawnW) * 0.5;
    } else {
      state.panX = clamp(state.panX, window.innerWidth - drawnW, 0);
    }
    if (drawnH <= window.innerHeight) {
      state.panY = (window.innerHeight - drawnH) * 0.5;
    } else {
      state.panY = clamp(state.panY, window.innerHeight - drawnH, 0);
    }
  }

  function screenToWorld(x, y) {
    return {
      x: (x - state.panX) / state.zoom,
      y: (y - state.panY) / state.zoom,
    };
  }

  function blockFrame(block, t) {
    const cycle = (t * block.rate * controls.mutation + block.phase) % TAU;
    const appear = cycle / TAU;
    const age = appear / block.duty;
    const large = block.w * block.h > 18000 || block.w > 180 || block.h > 160;
    const hopTick = Math.floor(t * 0.025 * controls.mutation + block.phase);
    const gridHopX = large ? 0 : ((hopTick % 3) - 1) * block.hop;
    const gridHopY = large ? 0 : (((hopTick + 1) % 3) - 1) * block.hop;
    const driftX = large ? 0 : block.anchored ? gridHopX : Math.sin(t * 0.05 + block.phase) * 18;
    const driftY = large ? 0 : block.anchored ? gridHopY : Math.cos(t * 0.043 + block.phase) * 14;
    const mutate = Math.floor(t * 0.35 * controls.mutation + block.phase) % 11 === 0;
    return {
      visible: appear <= block.duty,
      age,
      large,
      mutate,
      x: clamp(Math.floor((block.x + driftX) / block.cell) * block.cell, 0, WORLD_W - block.w),
      y: clamp(Math.floor((block.y + driftY) / block.cell) * block.cell, 0, WORLD_H - block.h),
      w: block.w + (large ? 0 : mutate ? block.cell * 2 : 0),
      h: block.h + (large ? 0 : mutate ? -block.cell : 0),
    };
  }

  function drawSubstrate(t) {
    const data = substrateImage.data;
    const px = (state.pointerX / WORLD_W) * SUB_W;
    const py = (state.pointerY / WORLD_H) * SUB_H;
    const pulse = Math.max(0, state.perturb);
    const motion = controls.motion;
    const grainSize = controls.grain;
    const motionTick = Math.floor(t * (3 + motion * 5));
    const base = [142, 116, 238];
    for (let y = 0; y < SUB_H; y += 1) {
      for (let x = 0; x < SUB_W; x += 1) {
        const i = y * SUB_W + x;
        const gx = Math.floor(x / grainSize);
        const gy = Math.floor(y / grainSize);
        const cx = gx * grainSize;
        const cy = gy * grainSize;
        const dx = x - px;
        const dy = y - py;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const pointerWave = Math.sin(dist * 0.42 - t * 5.5) * smoothstep(42, 0, dist) * pulse;
        const flowA = valueNoise(cx + Math.sin(t * 0.19 * motion) * 34, cy - t * (5.5 + motion * 7.5), 18, 9101);
        const flowB = valueNoise(cx * 0.72 - t * (7 + motion * 11), cy * 0.72 + Math.cos(t * 0.13) * 28, 11, 3127);
        const flowC = valueNoise(cx + Math.sin(cy * 0.045 + t * 0.8 * motion) * 24, cy + Math.cos(cx * 0.034 - t * 0.7 * motion) * 24, 7, 6401);
        fieldB[i] =
          fieldA[i] * (0.955 - motion * 0.018) +
          (flowA * 0.42 + flowB * 0.34 + flowC * 0.24 + pointerWave * 0.08) *
            (0.045 + motion * 0.025);
        const current = clamp(fieldB[i], 0, 1);
        const vein = Math.sin((flowA - flowB) * TAU * 2.7 + flowC * 5.1 + t * (0.75 + motion * 0.85));
        const cellHash = hash2(gx, gy, 8803);
        const shape =
          current * 0.62 +
          (vein * 0.5 + 0.5) * 0.25 +
          valueNoise(gx + t * motion * 1.7, gy - t * motion * 1.3, 4.2, 991) * 0.13;
        let color = base;
        if (shape > 0.58) {
          const paletteIndex = Math.floor((cellHash * 17 + shape * 9 + motionTick * 0.11) % 9);
          const shapeColor = palette[paletteIndex];
          const mix = clamp((shape - 0.5) * 1.7, 0.18, 0.92);
          color = [
            Math.floor(lerp(base[0], shapeColor[0], mix)),
            Math.floor(lerp(base[1], shapeColor[1], mix)),
            Math.floor(lerp(base[2], shapeColor[2], mix)),
          ];
        }
        const idx = i * 4;
        data[idx] = color[0];
        data[idx + 1] = color[1];
        data[idx + 2] = color[2];
        data[idx + 3] = 255;
      }
    }
    fieldA.set(fieldB);
    substrateCtx.putImageData(substrateImage, 0, 0);
    ctx.imageSmoothingEnabled = false;
    if (controls.blur > 0) {
      ctx.save();
      ctx.filter = `blur(${controls.blur}px)`;
      const pad = controls.blur * 3;
      ctx.drawImage(substrateCanvas, -pad, -pad, WORLD_W + pad * 2, WORLD_H + pad * 2);
      ctx.restore();
    } else {
      ctx.drawImage(substrateCanvas, 0, 0, WORLD_W, WORLD_H);
    }
  }

  function drawDitherRect(x, y, w, h, colorA, colorB, cell, salt, t, mode) {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const iw = Math.floor(w);
    const ih = Math.floor(h);
    ctx.fillStyle = rgb(colorA);
    ctx.fillRect(ix, iy, iw, ih);

    for (let yy = 0; yy < ih; yy += cell) {
      for (let xx = 0; xx < iw; xx += cell) {
        const gx = Math.floor(xx / cell);
        const gy = Math.floor(yy / cell);
        const tick = Math.floor(t * (0.6 + controls.mutation * 1.1));
        const n = hash2(gx, gy, salt + Math.floor(t * controls.mutation * 0.8));
        const checker = ((gx ^ gy ^ salt) & 1) === 0;
        const address = ((gx * 3 + gy * 5 + tick + salt) & 15) < 2;
        const scan = (gy + tick) % 6 === 0;
        const diagonal = (gx + gy + tick) % 9 === 0;
        const sparse = n > 0.94;
        let draw = false;
        if (mode === 1) draw = checker;
        else if (mode === 2) draw = scan;
        else if (mode === 3) draw = address;
        else if (mode === 4) draw = sparse;
        else if (mode === 5) draw = diagonal || ((gx & gy) % 13 === 0);
        else if (mode === 6) draw = ((gx * 11 + gy * 7 + salt) & 31) === 0 || ((gx + tick) % 11 === 0 && (gy & 3) === 0);
        if (draw) {
          ctx.fillStyle = rgb(n > 0.84 ? colorB : palette[(gx + gy + salt) % palette.length]);
          ctx.fillRect(ix + xx, iy + yy, Math.min(cell, iw - xx), Math.min(cell, ih - yy));
        }
      }
    }
  }

  function rgb(c, a) {
    if (a === undefined) return `rgb(${c[0]},${c[1]},${c[2]})`;
    return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  }

  function setupControls() {
    for (const key of Object.keys(defaults)) {
      const input = document.getElementById(key);
      input.addEventListener("input", () => {
      controls[key] = key === "grain" || key === "blur" ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value);
      });
    }
    panelToggle.addEventListener("click", () => {
      panel.classList.toggle("is-collapsed");
    });
    resetParams.addEventListener("click", () => {
      for (const key of Object.keys(defaults)) {
        controls[key] = defaults[key];
        document.getElementById(key).value = String(defaults[key]);
      }
    });
  }

  function drawBlocks(t) {
    for (let i = 0; i < blocks.length; i += 1) {
      if (hash2(i, 2, 99) > controls.density) continue;
      const b = blocks[i];
      const frame = blockFrame(b, t);
      if (!frame.visible) continue;
      drawDitherRect(frame.x, frame.y, frame.w, frame.h, b.colorA, b.colorB, b.cell, i * 991, t, b.mode);

      if (frame.age > 0.92 || frame.mutate) {
        ctx.fillStyle = rgb([0, 0, 0]);
        ctx.fillRect(Math.floor(frame.x + frame.w - 7), Math.floor(frame.y + frame.h - 29), 7, 29);
      }
    }
  }

  function drawArtifacts(t) {
    for (let i = 0; i < artifacts.length; i += 1) {
      if (hash2(i, 7, 33) > controls.density) continue;
      const a = artifacts[i];
      const cycle = (t * a.rate * controls.mutation + a.phase) % 1;
      if (cycle > 0.965) continue;
      const tick = Math.floor(t * (0.8 + controls.mutation * 0.8) + i * 3 + Math.floor(cycle * 5));
      const x = clamp(a.x, 0, WORLD_W - a.w);
      const y = clamp(a.y, 0, WORLD_H - a.h);
      if (a.kind === 0) drawScanRegister(x, y, a.w, a.h, a.cell, a.a, a.b, tick);
      else if (a.kind === 1) drawAddressBus(x, y, a.w, a.h, a.cell, a.a, a.b, tick);
      else if (a.kind === 2) drawParityPlane(x, y, a.w, a.h, a.cell, a.a, a.b, tick);
      else if (a.kind === 3) drawNibbleGlyphs(x, y, a.w, a.h, a.cell, a.a, a.b, tick);
      else drawCompressionBlocks(x, y, a.w, a.h, a.cell, a.a, a.b, tick);
    }
  }

  function drawScanRegister(x, y, w, h, cell, a, b, tick) {
    ctx.fillStyle = rgb(a);
    ctx.fillRect(x, y, w, h);
    for (let yy = 0; yy < h; yy += cell * 2) {
      ctx.fillStyle = (yy / cell + tick) % 8 === 0 ? rgb(b) : "rgb(0,0,0)";
      ctx.fillRect(x, y + yy, w, cell);
    }
    for (let xx = 0; xx < w; xx += cell * 9) {
      ctx.fillStyle = rgb(b);
      ctx.fillRect(x + xx, y, cell, h);
    }
  }

  function drawAddressBus(x, y, w, h, cell, a, b, tick) {
    ctx.fillStyle = rgb([0, 0, 0]);
    ctx.fillRect(x, y, w, h);
    for (let row = 0; row < h / cell; row += 1) {
      const bits = (row * 37 + tick * 11) & 255;
      ctx.fillStyle = row % 3 === 0 ? rgb(a) : rgb(b);
      for (let bit = 0; bit < 8; bit += 1) {
        if ((bits >> bit) & 1) {
          ctx.fillRect(x + bit * cell * 3, y + row * cell, cell * (1 + (bit & 1)), cell);
        }
      }
    }
  }

  function drawParityPlane(x, y, w, h, cell, a, b, tick) {
    for (let yy = 0; yy < h; yy += cell) {
      for (let xx = 0; xx < w; xx += cell) {
        const gx = Math.floor(xx / cell);
        const gy = Math.floor(yy / cell);
        const bit = ((gx & gy) ^ (gx | tick) ^ gy) & 3;
        ctx.fillStyle = bit === 0 ? rgb(a) : bit === 1 ? rgb(b) : "rgb(0,0,0)";
        if (bit < 3) ctx.fillRect(x + xx, y + yy, cell, cell);
      }
    }
  }

  function drawNibbleGlyphs(x, y, w, h, cell, a, b, tick) {
    ctx.fillStyle = rgb(a);
    ctx.fillRect(x, y, w, h);
    const glyphW = cell * 4;
    const glyphH = cell * 5;
    for (let gy = 0; gy < h; gy += glyphH + cell) {
      for (let gx = 0; gx < w; gx += glyphW + cell) {
        const code = ((gx / cell) * 5 + (gy / cell) * 3 + tick) & 15;
        ctx.fillStyle = code & 1 ? rgb(b) : "rgb(0,0,0)";
        for (let row = 0; row < 5; row += 1) {
          for (let col = 0; col < 4; col += 1) {
            if (((code * 29 + row * 7 + col * 11) >> (col % 4)) & 1) {
              ctx.fillRect(x + gx + col * cell, y + gy + row * cell, cell, cell);
            }
          }
        }
      }
    }
  }

  function drawCompressionBlocks(x, y, w, h, cell, a, b, tick) {
    ctx.fillStyle = rgb(a);
    ctx.fillRect(x, y, w, h);
    for (let yy = 0; yy < h; yy += cell * 4) {
      for (let xx = 0; xx < w; xx += cell * 4) {
        const n = hash2(xx / cell, yy / cell, tick);
        ctx.fillStyle = n > 0.55 ? rgb(b) : n > 0.34 ? "rgb(0,0,0)" : rgb(a);
        ctx.fillRect(x + xx, y + yy, cell * (1 + Math.floor(n * 4)), cell * (1 + ((tick + xx) & 3)));
      }
    }
  }

  function bresenhamLine(x0, y0, x1, y1, cell, color, gapPhase) {
    x0 = Math.round(x0 / cell);
    y0 = Math.round(y0 / cell);
    x1 = Math.round(x1 / cell);
    y1 = Math.round(y1 / cell);
    let dx = Math.abs(x1 - x0);
    let sx = x0 < x1 ? 1 : -1;
    let dy = -Math.abs(y1 - y0);
    let sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    let n = 0;
    ctx.fillStyle = rgb(color);
    while (true) {
      if ((n + gapPhase) % 13 !== 0) {
        ctx.fillRect(x0 * cell, y0 * cell, cell, cell);
      }
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x0 += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y0 += sy;
      }
      n += 1;
      if (n > 5000) break;
    }
  }

  function drawRays(t) {
    for (let f = 0; f < rayFamilies.length; f += 1) {
      const family = rayFamilies[f];
      const ox = family.originX;
      const oy = family.originY;
      const angle = t * family.speed * controls.raySpeed + family.phase + state.keyboardPhase * 0.006;
      for (let i = 0; i < family.rays.length; i += 1) {
        const r = family.rays[i];
        const v = rotate3(r, family.axis, angle + Math.sin(t * 0.031 + r.phase) * 0.38);
        const depth = 1 / (family.projection + (v.z + 1.35) * 0.34);
        const x1 = ox + v.x * r.length * depth;
        const y1 = oy + v.y * r.length * depth;
        const visible = v.z > -0.92 || ((i + Math.floor(t * 2)) % 5 === 0);
        if (visible) bresenhamLine(ox, oy, x1, y1, r.step, r.color, Math.floor(t * 5 + i + f * 7));
      }

      ctx.fillStyle = "rgb(0,0,0)";
      ctx.fillRect(Math.floor(ox - 13), Math.floor(oy - 13), 26, 26);
      ctx.fillStyle = f % 2 === 0 ? "rgb(212,255,21)" : "rgb(246,48,176)";
      ctx.fillRect(Math.floor(ox - 8), Math.floor(oy - 8), 16, 16);
    }
  }

  function drawCircleBresenham(cx, cy, radius, cell, color) {
    let x = Math.round(radius / cell);
    let y = 0;
    let err = 0;
    ctx.fillStyle = rgb(color);
    while (x >= y) {
      plotCirclePoints(cx, cy, x * cell, y * cell, cell);
      y += 1;
      if (err <= 0) {
        err += 2 * y + 1;
      }
      if (err > 0) {
        x -= 1;
        err -= 2 * x + 1;
      }
    }
  }

  function plotCirclePoints(cx, cy, x, y, cell) {
    const pts = [
      [cx + x, cy + y],
      [cx + y, cy + x],
      [cx - y, cy + x],
      [cx - x, cy + y],
      [cx - x, cy - y],
      [cx - y, cy - x],
      [cx + y, cy - x],
      [cx + x, cy - y],
    ];
    for (const p of pts) {
      ctx.fillRect(Math.round(p[0] / cell) * cell, Math.round(p[1] / cell) * cell, cell, cell);
    }
  }

  function drawGliderPattern(x, y, cell, color, phase) {
    const patterns = [
      [
        [0, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
      ],
      [
        [1, 0],
        [2, 1],
        [0, 2],
        [1, 2],
        [2, 2],
      ],
      [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [2, 1],
      ],
      [
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 2],
      ],
    ];
    ctx.fillStyle = rgb(color);
    const p = patterns[Math.floor(phase) % patterns.length];
    for (const point of p) {
      ctx.fillRect(Math.floor(x + point[0] * cell), Math.floor(y + point[1] * cell), cell, cell);
    }
  }

  function drawGliders(t, dt) {
    for (const g of gliders) {
      g.x = (g.x + g.vx * dt + WORLD_W) % WORLD_W;
      g.y = (g.y + g.vy * dt + WORLD_H) % WORLD_H;
      const flicker = Math.floor(t * 8 + g.phase);
      drawGliderPattern(g.x, g.y, g.cell, g.color, flicker);
      if (flicker % 5 === 0) {
        bresenhamLine(g.x, g.y, g.x - g.vx * 1.3, g.y - g.vy * 1.3, Math.max(2, g.cell - 2), [0, 36, 215], 0);
      }
    }
  }

  function drawStaticBlockCircles(t) {
    for (let i = 0; i < blocks.length; i += 1) {
      if (hash2(i, 2, 99) > controls.density) continue;
      if (i % 6 !== 1 && i % 9 !== 4) continue;
      const b = blocks[i];
      const frame = blockFrame(b, t);
      if (!frame.visible || !frame.large || !b.anchored) continue;
      const cell = Math.max(3, Math.min(10, b.cell));
      const cx = Math.floor((frame.x + frame.w * (0.34 + hash2(i, 1, 77) * 0.36)) / cell) * cell;
      const cy = Math.floor((frame.y + frame.h * (0.32 + hash2(i, 2, 77) * 0.36)) / cell) * cell;
      const maxR = Math.max(cell * 3, Math.min(frame.w, frame.h) * 0.34);
      const radius = cell * 2 + (Math.sin(t * (0.7 + hash2(i, 5, 22)) + b.phase) * 0.5 + 0.5) * maxR;
      ctx.save();
      ctx.beginPath();
      ctx.rect(frame.x, frame.y, frame.w, frame.h);
      ctx.clip();
      drawCircleBresenham(cx, cy, radius, cell, palette[(i + Math.floor(t * 0.4)) % 9]);
      if (hash2(i, Math.floor(t * 0.8), 123) > 0.62) {
        drawCircleBresenham(cx, cy, Math.max(cell * 2, radius * 0.58), cell, palette[(i + 4) % 9]);
      }
      ctx.restore();
    }
  }

  function drawClawStreaks(t) {
    for (let i = 0; i < 26; i += 1) {
      const baseX = (hash2(i, 4, 77) * WORLD_W + Math.sin(t * 0.035 + i) * 200) % WORLD_W;
      const baseY = (hash2(i, 7, 91) * WORLD_H + Math.cos(t * 0.041 + i) * 140) % WORLD_H;
      const len = 34 + hash2(i, 8, 14) * 190;
      const angle = -0.45 + hash2(i, 2, 19) * 1.1 + Math.sin(t * 0.08 + i) * 0.28;
      const cell = [2, 3, 4, 6][i % 4];
      const color = palette[(i * 3 + Math.floor(t * 0.7)) % palette.length];
      if (hash2(i, Math.floor(t * 3), 500) < 0.34) continue;
      for (let k = 0; k < 3; k += 1) {
        bresenhamLine(
          baseX + k * 9,
          baseY + k * 4,
          baseX + Math.cos(angle) * len + k * 9,
          baseY + Math.sin(angle) * len + k * 4,
          cell,
          color,
          k + i,
        );
      }
    }
  }

  function drawMacroFields(t) {
    const panels = [
      [0.02, 0.03, 0.27, 0.36, [153, 115, 246]],
      [0.59, 0.05, 0.31, 0.18, [186, 255, 0]],
      [0.04, 0.73, 0.2, 0.2, [246, 24, 216]],
      [0.72, 0.78, 0.2, 0.18, [215, 38, 0]],
      [0.82, 0.29, 0.14, 0.35, [237, 225, 173]],
    ];
    for (let i = 0; i < panels.length; i += 1) {
      const p = panels[i];
      ctx.fillStyle = rgb(p[4], 0.86);
      const grid = 16;
      const x = Math.floor((p[0] * WORLD_W) / grid) * grid;
      const y = Math.floor((p[1] * WORLD_H) / grid) * grid;
      const w = Math.floor(p[2] * WORLD_W);
      const h = Math.floor(p[3] * WORLD_H);
      ctx.fillRect(x, y, w, h);
      if (i % 2 === 0) {
        const insetW = w * (0.18 + (Math.floor(t * 0.04 + i) % 3) * 0.06);
        drawDitherRect(x + w * 0.52, y + h * 0.2, insetW, h * 0.58, p[4], palette[(i + 2) % palette.length], 8, i * 55, t, 4);
      }
    }
  }

  function drawWorld(timestamp) {
    const t = timestamp * 0.001;
    const dt = state.lastTime ? Math.min(0.05, (timestamp - state.lastTime) * 0.001) : 0.016;
    state.lastTime = timestamp;
    state.frame += 1;
    state.perturb *= 0.94;

    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
    ctx.setTransform(
      state.dpr * state.zoom,
      0,
      0,
      state.dpr * state.zoom,
      state.dpr * state.panX,
      state.dpr * state.panY,
    );

    ctx.imageSmoothingEnabled = false;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, WORLD_W, WORLD_H);
    ctx.clip();
    drawSubstrate(t);
    drawMacroFields(t);
    drawArtifacts(t);
    drawRays(t);
    drawBlocks(t);
    drawStaticBlockCircles(t);
    drawGliders(t, dt);
    drawClawStreaks(t);
    ctx.restore();

    requestAnimationFrame(drawWorld);
  }

  function onPointerMove(event) {
    const w = screenToWorld(event.clientX, event.clientY);
    state.pointerX = w.x;
    state.pointerY = w.y;
    if (state.pointerDown) {
      state.panX += event.clientX - state.dragX;
      state.panY += event.clientY - state.dragY;
      state.dragX = event.clientX;
      state.dragY = event.clientY;
      clampView();
      state.perturb = Math.min(1.8, state.perturb + 0.04);
    }
  }

  function onWheel(event) {
    if (event.target.closest && event.target.closest(".panel")) return;
    event.preventDefault();
    const before = screenToWorld(event.clientX, event.clientY);
    const factor = Math.exp(-event.deltaY * 0.0012);
    state.zoom = clamp(state.zoom * factor, minCoverZoom(), 24);
    state.panX = event.clientX - before.x * state.zoom;
    state.panY = event.clientY - before.y * state.zoom;
    clampView();
    state.perturb = Math.min(1.5, state.perturb + 0.25);
  }

  function onKeyDown(event) {
    if (event.key === "0") {
      state.zoom = minCoverZoom();
      state.panX = (window.innerWidth - WORLD_W * state.zoom) * 0.5;
      state.panY = (window.innerHeight - WORLD_H * state.zoom) * 0.5;
      clampView();
    } else {
      state.keyboardPhase += event.key.length === 1 ? event.key.charCodeAt(0) : 17;
      state.perturb = 2.4;
      for (let i = 0; i < 5; i += 1) {
        const b = blocks[Math.floor(rand() * blocks.length)];
        b.x = (b.x + rand() * 440 - 220 + WORLD_W) % WORLD_W;
        b.y = (b.y + rand() * 260 - 130 + WORLD_H) % WORLD_H;
        b.mode = (b.mode + 1 + Math.floor(rand() * 4)) % 7;
      }
      clampView();
    }
  }

  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerdown", (event) => {
    if (event.target.closest && event.target.closest(".panel")) return;
    state.pointerDown = true;
    state.dragX = event.clientX;
    state.dragY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  window.addEventListener("pointerup", (event) => {
    if (!state.pointerDown) return;
    state.pointerDown = false;
    canvas.releasePointerCapture(event.pointerId);
  });
  window.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKeyDown);

  for (let i = 0; i < fieldA.length; i += 1) {
    fieldA[i] = rand();
  }

  setupControls();
  resize();
  requestAnimationFrame(drawWorld);
})();
