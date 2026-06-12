var parameters = [
  ["rhythm", ["steady pulse", "private pulse", "free time", "pointillism"]],
  ["harmony", ["one pedal", "two keys", "fast chromatic circulation", "no plan"]],
  ["scale", ["five notes", "014 cell", "blues shadow", "all twelve"]],
  ["theme", ["one seed", "repeat badly", "quote and erase", "no repeats"]],
  ["time", ["45 seconds", "one breath", "three miniatures", "end early"]],
  ["continuity", ["drone", "slow morph", "jump cuts", "silence edits"]],
  ["role", ["left hand leads", "right hand leads", "two players", "hands avoid"]],
  ["timbre", ["dry", "pedal cloud", "muted", "edge registers"]],
  ["density", ["mostly rests", "thin line", "busy", "sudden mass"]]
];

var cards = [
  ["clock", "play 45 seconds"],
  ["ending", "stop too early"],
  ["chromatic", "use all twelve before repeating"],
  ["register", "avoid the middle"],
  ["continuity", "one texture only"],
  ["cuts", "never continue an idea"],
  ["free time", "clear tempo, no shared tempo"],
  ["pointillism", "no neighbors"],
  ["role", "left hand takes over"],
  ["density", "burst / silence / burst"],
  ["timbre", "normal / muted / pedal"],
  ["motive", "one interval survives"],
  ["harmony", "hold one bass note too long"],
  ["memory", "bring something back wrong"],
  ["hands", "two players, one piano"]
];

var starters = [
  "first sound is answerable",
  "start in the middle",
  "left hand misunderstands",
  "do not fix the accident",
  "find an ending before the middle",
  "make one familiar thing strange"
];

var turns = [
  "change register only",
  "keep rhythm, replace pitch",
  "answer density with space",
  "hard cut",
  "repeat as memory",
  "weaken the tonal center",
  "move to private pulse",
  "let a rest lead"
];

var free = {
  harmony: true,
  continuity: true
};

var points = [];

function pick(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function shuffle(list) {
  var copy = list.slice();
  copy.sort(function () {
    return Math.random() - 0.5;
  });
  return copy;
}

function renderParameters() {
  var list = document.getElementById("parameterList");
  list.innerHTML = "";

  for (var i = 0; i < parameters.length; i++) {
    var name = parameters[i][0];
    var button = document.createElement("button");
    button.type = "button";
    button.className = "parameter" + (free[name] ? " free" : "");
    button.appendChild(document.createTextNode(name));
    button.onclick = toggleParameter(name);
    list.appendChild(button);
    list.appendChild(document.createTextNode(" "));
  }
}

function toggleParameter(name) {
  return function () {
    free[name] = !free[name];
    renderParameters();
    newFrame();
  };
}

function setAll(value) {
  for (var i = 0; i < parameters.length; i++) {
    free[parameters[i][0]] = value;
  }
  renderParameters();
  newFrame();
}

function randomizeParameters() {
  for (var i = 0; i < parameters.length; i++) {
    free[parameters[i][0]] = Math.random() > 0.5;
  }
  renderParameters();
  newFrame();
}

function newFrame() {
  var fixed = [];
  var open = [];

  for (var i = 0; i < parameters.length; i++) {
    if (free[parameters[i][0]]) {
      open.push(parameters[i]);
    } else {
      fixed.push(parameters[i]);
    }
  }

  fixed = shuffle(fixed).slice(0, 3);
  open = shuffle(open).slice(0, 3);

  var lines = [];
  lines.push(pick(["0:45", "1:00", "1:30", "2:00", "3:00"]));
  lines.push(pick(starters));
  lines.push("");
  lines.push("in:");
  for (i = 0; i < fixed.length; i++) {
    lines.push("  " + fixed[i][0] + ": " + pick(fixed[i][1]));
  }
  lines.push("");
  lines.push("out:");
  if (open.length === 0) {
    lines.push("  choose one while playing");
  } else {
    for (i = 0; i < open.length; i++) {
      lines.push("  " + open[i][0]);
    }
  }
  lines.push("");
  lines.push("turn: " + pick(turns));

  document.getElementById("frameOutput").textContent = lines.join("\n");
}

function newLine() {
  var count = 8 + Math.floor(Math.random() * 7);
  points = [];

  for (var i = 0; i < count; i++) {
    points.push({
      x: i / (count - 1),
      y: Math.random()
    });
  }

  drawLine();
}

function invertLine() {
  for (var i = 0; i < points.length; i++) {
    points[i].y = 1 - points[i].y;
  }
  drawLine();
}

function drawLine() {
  var canvas = document.getElementById("lineCanvas");
  var context = canvas.getContext("2d");
  var width = canvas.width;
  var height = canvas.height;
  var pad = 20;

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "#cccccc";
  context.lineWidth = 1;
  for (var i = 1; i < 4; i++) {
    var y = pad + ((height - pad * 2) * i) / 4;
    context.beginPath();
    context.moveTo(pad, y);
    context.lineTo(width - pad, y);
    context.stroke();
  }

  context.strokeStyle = "#000000";
  context.lineWidth = 3;
  context.beginPath();
  for (i = 0; i < points.length; i++) {
    var x = pad + points[i].x * (width - pad * 2);
    y = pad + points[i].y * (height - pad * 2);
    if (i === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }
  context.stroke();

  context.fillStyle = "#aa0000";
  for (i = 0; i < points.length; i++) {
    x = pad + points[i].x * (width - pad * 2);
    y = pad + points[i].y * (height - pad * 2);
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fill();
  }
}

function drawThree() {
  var deck = document.getElementById("cardDeck");
  var hand = shuffle(cards).slice(0, 3);
  var html = "<tr>";

  for (var i = 0; i < hand.length; i++) {
    html += "<td><b>" + hand[i][0] + "</b><br>" + hand[i][1] + "</td>";
  }

  html += "</tr>";
  deck.innerHTML = html;
}

document.getElementById("allIn").onclick = function () {
  setAll(false);
};
document.getElementById("allOut").onclick = function () {
  setAll(true);
};
document.getElementById("randomize").onclick = randomizeParameters;
document.getElementById("newFrame").onclick = newFrame;
document.getElementById("newLine").onclick = newLine;
document.getElementById("invertLine").onclick = invertLine;
document.getElementById("drawCards").onclick = drawThree;

renderParameters();
newFrame();
newLine();
drawThree();
