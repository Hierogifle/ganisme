/* Presentation engine: versions, scaling, navigation, steps inside a slide, rooms, overview, timer.
   A slide is a <section class="slide" data-room="..." data-title="...">.
   Inside a slide, elements with class "step" appear one after the other (data-step="n" to group or order them);
   a slide can also declare data-steps="n" for steps handled by a widget (it receives a "step" event).

   Two versions share this page: a 15-minute talk and a 30-minute one. Opened without parameter, the page asks
   which one to give; the choice reloads it as index.html?duree=15 or index.html?duree=30.
   An element marked data-in="short" only exists in the 15-minute version; a slide marked data-in="long" belongs
   to the 30-minute talk and, in the 15-minute version, is moved to the reserves (backup screens) instead of
   being lost. */
(function () {
  "use strict";

  const stage = document.getElementById("stage");

  // ---------------------------------------------------------------- scaling
  function fit() {
    const k = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
    stage.style.transform = "scale(" + k + ")";
  }
  window.addEventListener("resize", fit);
  fit();

  // ---------------------------------------------------------------- which version?
  const asked = new URLSearchParams(location.search).get("duree");
  if (asked !== "15" && asked !== "30") {
    // No version requested: show the choice and stop here (the choice reloads the page)
    const chooser = document.getElementById("chooser");
    const count = (other) => document.querySelectorAll(
      '.slide:not([data-room="reserves"]):not([data-in="' + other + '"])').length;
    chooser.querySelector('[data-duree="15"] em').textContent = count("long") + " écrans";
    chooser.querySelector('[data-duree="30"] em').textContent = count("short") + " écrans";
    const choose = (d) => { location.search = "?duree=" + d; };
    const buttons = Array.from(chooser.querySelectorAll("button"));
    buttons.forEach((b) => b.addEventListener("click", () => choose(b.dataset.duree)));
    document.addEventListener("keydown", (e) => {
      if (e.key === "1") choose("15");
      else if (e.key === "3") choose("30");
      else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        buttons[(buttons.indexOf(document.activeElement) + 1) % buttons.length].focus();
        e.preventDefault();
      }
    });
    chooser.classList.add("open");
    buttons[0].focus();
    return;
  }
  const MODE = asked === "30" ? "long" : "short";
  document.documentElement.dataset.mode = MODE;

  // Rooms in visiting order: label shown top-left, wall colour, planned minutes for each version (timer)
  const ROOMS = [
    { id: "prologue", label: "Prologue · Le hall, la nuit", wall: "--wall-prologue", minutes: { long: 2, short: 1 } },
    { id: "plan", label: "Salle 1 · Le plan", wall: "--wall-plan", minutes: { long: 4, short: 2 } },
    { id: "archives", label: "Salle 2 · Les archives", wall: "--wall-archives", minutes: { long: 3, short: 1.5 } },
    { id: "workshop", label: "Salle 3 · L'atelier", wall: "--wall-workshop", minutes: { long: 3, short: 1.25 } },
    { id: "inspector", label: "Salle 4 · Le bureau de l'Inspecteur", wall: "--wall-inspector", minutes: { long: 6, short: 2.5 } },
    { id: "attempts", label: "Salle 5 · Les essais ratés", wall: "--wall-attempts", minutes: { long: 4, short: 2 } },
    { id: "rhythm", label: "Salle 6 · Le bon rythme", wall: "--wall-rhythm", minutes: { long: 4, short: 2.5 } },
    { id: "hanging", label: "Salle 7 · L'accrochage", wall: "--wall-hanging", minutes: { long: 2, short: 0.75 } },
    { id: "epilogue", label: "Épilogue · L'arrestation", wall: "--wall-prologue", minutes: { long: 3, short: 1.5 } },
    { id: "reserves", label: "Les réserves · écrans de secours", wall: "--wall-reserves", minutes: { long: 0, short: 0 } },
  ];
  const CAST = {
    curator: { name: "Le Conservateur", img: "assets/cast/curator.jpg" },
    forger: { name: "Le Faussaire", img: "assets/cast/forger.jpg" },
    inspector: { name: "L'Inspecteur", img: "assets/cast/inspector.jpg" },
  };

  const roomLabel = document.getElementById("room-label");
  const counter = document.getElementById("counter");
  const floorplan = document.getElementById("floorplan");
  const overview = document.getElementById("overview");
  const timerEl = document.getElementById("timer");

  // ---------------------------------------------------------------- versions
  // Elements of the other version are removed, except the slides of the long talk, kept as backup screens.
  const demoted = [];
  document.querySelectorAll("[data-in]").forEach((el) => {
    if (el.dataset.in === MODE) return;
    if (el.classList.contains("slide") && el.dataset.in === "long") {
      el.dataset.fromRoom = (ROOMS.find((r) => r.id === el.dataset.room) || {}).label || "";
      el.dataset.room = "reserves";
      if (!el.id) el.id = "long-" + (demoted.length + 1);
      demoted.push(el);
    } else {
      el.remove();
    }
  });

  const all = Array.from(document.querySelectorAll(".slide"));
  const core = all.filter((s) => s.dataset.room !== "reserves");
  const reserves = all.filter((s) => s.dataset.room === "reserves" && !demoted.includes(s));
  let slides = core.concat(reserves);

  if (demoted.length) {
    // A generated index of the screens of the long version, placed right after the index of the reserves
    const index = document.createElement("section");
    index.className = "slide";
    index.id = "long-index";
    index.dataset.room = "reserves";
    index.dataset.title = "Écrans de la version 30 minutes";
    index.innerHTML = "<h2>Les écrans de la version 30 minutes</h2>"
      + "<p class=\"lede\">Retirés de l'exposé de 15 minutes, disponibles pour les questions.</p>";
    const grid = document.createElement("div");
    grid.className = "rooms compact fill";
    grid.style.alignContent = "center";
    demoted.forEach((s) => {
      const b = document.createElement("button");
      b.className = "room";
      b.type = "button";
      b.dataset.goto = s.id;
      const from = document.createElement("span");
      from.textContent = s.dataset.fromRoom.split(" · ").pop();
      const title = document.createElement("b");
      title.textContent = s.dataset.title;
      b.append(from, title);
      grid.appendChild(b);
    });
    index.appendChild(grid);
    stage.insertBefore(index, floorplan);
    slides = core.concat(reserves.slice(0, 1), [index], demoted, reserves.slice(1));

    const mainIndex = reserves[0] && reserves[0].querySelector(".rooms");
    if (mainIndex) {
      const b = document.createElement("button");
      b.className = "room";
      b.type = "button";
      b.dataset.goto = "long-index";
      b.innerHTML = "<b>Écrans de la version 30 minutes</b><p>" + demoted.length + " écrans retirés de l'exposé court</p>";
      mainIndex.insertBefore(b, mainIndex.firstChild);
    }
  }

  const switcher = document.getElementById("mode-switch");
  if (switcher) {
    switcher.innerHTML = "Version " + (MODE === "short" ? "15" : "30")
      + " minutes · <a href=\"index.html\">changer de version</a>";
  }

  const coreCount = core.length;
  let current = 0;
  let step = 0;
  let returnTo = null;        // slide to come back to after a detour in the reserves
  let startedAt = null;

  // ---------------------------------------------------------------- dialogue lines
  document.querySelectorAll(".talk").forEach((el) => {
    const who = CAST[el.dataset.who];
    const text = el.innerHTML;
    el.innerHTML = "";
    const badge = document.createElement("div");
    badge.className = "who";
    const img = document.createElement("img");
    img.src = who.img;
    img.alt = "";
    badge.append(img, document.createTextNode(who.name));
    const line = document.createElement("div");
    line.className = "line";
    line.innerHTML = text;
    el.append(badge, line);
  });

  // ---------------------------------------------------------------- steps
  function stepCount(slide) {
    let max = Number(slide.dataset.steps || 0);
    let auto = 0;
    slide.querySelectorAll(".step").forEach((el) => {
      if (!el.dataset.step) el.dataset.step = String(++auto);
      else auto = Math.max(auto, Number(el.dataset.step));
      max = Math.max(max, Number(el.dataset.step));
    });
    return max;
  }
  const counts = slides.map(stepCount);

  function applyStep() {
    const slide = slides[current];
    slide.querySelectorAll(".step").forEach((el) => {
      const n = Number(el.dataset.step);
      el.classList.toggle("shown", n <= step);
      if (el.dataset.until) el.classList.toggle("gone", step > Number(el.dataset.until));
    });
    slide.dispatchEvent(new CustomEvent("step", { detail: step }));
  }

  // ---------------------------------------------------------------- navigation
  function roomOf(i) {
    return ROOMS.find((r) => r.id === slides[i].dataset.room) || ROOMS[0];
  }

  function show(i, atStep) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    slides[current].classList.remove("active");
    slides[current].dispatchEvent(new CustomEvent("leave"));
    current = i;
    step = atStep === "last" ? counts[i] : 0;
    const slide = slides[i];
    slide.classList.add("active");
    const room = roomOf(i);
    stage.style.backgroundColor = "var(" + room.wall + ")";
    roomLabel.textContent = slide.dataset.fromRoom ? room.label + " · " + slide.dataset.fromRoom.split(" · ").pop() : room.label;
    counter.textContent = slide.dataset.room === "reserves"
      ? "Réserve " + (i - coreCount + 1) + " / " + (slides.length - coreCount)
      : (i + 1) + " / " + coreCount;
    slide.dispatchEvent(new CustomEvent("enter"));
    applyStep();
    updateFloorplan();
    try { history.replaceState(null, "", "#" + (i + 1)); } catch (e) { /* file:// may refuse */ }
    if (startedAt === null && i > 0) startedAt = Date.now();
  }

  function next() {
    if (step < counts[current]) { step++; applyStep(); }
    else if (current < slides.length - 1) show(current + 1);
  }
  function prev() {
    if (step > 0) { step--; applyStep(); }
    else if (current > 0) show(current - 1, "last");
  }
  function goToRoom(id) {
    const i = slides.findIndex((s) => s.dataset.room === id);
    if (i >= 0) show(i);
  }

  // ---------------------------------------------------------------- floor plan (progress by room)
  const roomButtons = {};
  ROOMS.forEach((room) => {
    const n = slides.filter((s) => s.dataset.room === room.id).length;
    if (!n) return;
    const b = document.createElement("button");
    b.type = "button";
    b.title = room.label;
    b.setAttribute("aria-label", room.label);
    b.style.setProperty("--n", room.id === "reserves" ? 2 : n);
    b.appendChild(document.createElement("i"));
    b.addEventListener("click", () => goToRoom(room.id));
    floorplan.appendChild(b);
    roomButtons[room.id] = { el: b, n: n, first: slides.findIndex((s) => s.dataset.room === room.id) };
  });
  function updateFloorplan() {
    Object.keys(roomButtons).forEach((id) => {
      const r = roomButtons[id];
      const done = Math.max(0, Math.min(r.n, current - r.first + 1));
      r.el.firstChild.style.width = (100 * done / r.n) + "%";
    });
  }

  // ---------------------------------------------------------------- overview
  function buildOverview() {
    const box = overview.querySelector(".ov-body");
    box.innerHTML = "";
    ROOMS.forEach((room) => {
      const items = slides.map((s, i) => [s, i]).filter(([s]) => s.dataset.room === room.id);
      if (!items.length) return;
      const wrap = document.createElement("div");
      wrap.className = "ov-room";
      const h = document.createElement("h3");
      h.textContent = room.label;
      const list = document.createElement("div");
      list.className = "ov-list";
      items.forEach(([s, i]) => {
        const b = document.createElement("button");
        b.type = "button";
        const title = s.dataset.title || (s.querySelector("h2, h1") || {}).textContent || "Écran " + (i + 1);
        b.innerHTML = "<i>" + (i + 1) + "</i>";
        b.appendChild(document.createTextNode(title));
        if (i === current) b.classList.add("current");
        b.addEventListener("click", () => { toggleOverview(false); show(i); });
        list.appendChild(b);
      });
      wrap.append(h, list);
      box.appendChild(wrap);
    });
  }
  function toggleOverview(force) {
    const open = force === undefined ? !overview.classList.contains("open") : force;
    if (open) buildOverview();
    overview.classList.toggle("open", open);
  }

  // ---------------------------------------------------------------- timer
  function plannedEnd(i) {
    const id = slides[i].dataset.room;
    let total = 0;
    for (const r of ROOMS) { total += r.minutes[MODE]; if (r.id === id) break; }
    return total;
  }
  function fmt(sec) {
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return m + ":" + String(s).padStart(2, "0");
  }
  setInterval(() => {
    if (timerEl.hidden) return;
    const elapsed = startedAt === null ? 0 : (Date.now() - startedAt) / 1000;
    const target = plannedEnd(current) * 60;
    timerEl.classList.toggle("late", elapsed > target && target > 0);
    timerEl.innerHTML = "<b>" + fmt(elapsed) + "</b><br>fin de salle prévue à " + fmt(target);
  }, 500);

  // ---------------------------------------------------------------- keyboard
  document.addEventListener("keydown", (e) => {
    if (e.target.matches && e.target.matches("input, textarea, select")) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (overview.classList.contains("open")) {
      if (k === "Escape" || k === "m" || k === "M") { toggleOverview(false); e.preventDefault(); }
      return;
    }
    if (k === "ArrowRight" || k === "PageDown" || k === " " || k === "Enter") { next(); e.preventDefault(); }
    else if (k === "ArrowLeft" || k === "PageUp" || k === "Backspace") { prev(); e.preventDefault(); }
    else if (k === "Home") { show(0); }
    else if (k === "End") { show(coreCount - 1); }
    else if (k === "m" || k === "M" || k === "Escape") { toggleOverview(); }
    else if (k === "r" || k === "R") {
      // from the talk: remember where we were; from a backup screen: back to the index of the reserves
      if (slides[current].dataset.room !== "reserves") returnTo = current;
      goToRoom("reserves");
    }
    else if (k === "b" || k === "B") { if (returnTo !== null) { show(returnTo, "last"); returnTo = null; } }
    else if (k === "f" || k === "F") {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
    }
    else if (k === "t" || k === "T") { timerEl.hidden = !timerEl.hidden; }
  });

  // Links inside the page that jump to a slide or a room: <button data-goto="res-fid"> or data-room-goto="plan"
  document.addEventListener("click", (e) => {
    const g = e.target.closest("[data-goto]");
    if (g) {
      const i = slides.findIndex((s) => s.id === g.dataset.goto);
      if (i >= 0) { if (slides[i].dataset.room === "reserves" && slides[current].dataset.room !== "reserves") returnTo = current; show(i); }
      return;
    }
    const rg = e.target.closest("[data-room-goto]");
    if (rg) goToRoom(rg.dataset.roomGoto);
  });

  window.Deck = { next: next, prev: prev, show: show, slides: slides, mode: MODE, coreCount: coreCount,
    current: () => current, step: () => step };

  // Widgets register their listeners in widgets.js; the first slide is shown once everything is loaded
  window.addEventListener("load", () => {
    const fromHash = parseInt((location.hash || "").slice(1), 10);
    show(Number.isFinite(fromHash) ? fromHash - 1 : 0);
  });
})();
