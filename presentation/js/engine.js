/* Presentation engine: scaling, navigation, steps inside a slide, rooms, overview, timer.
   A slide is a <section class="slide" data-room="..." data-title="...">.
   Inside a slide, elements with class "step" appear one after the other (data-step="n" to group or order them);
   a slide can also declare data-steps="n" for steps handled by a widget (it receives a "step" event). */
(function () {
  "use strict";

  // Rooms in visiting order: label shown top-left, wall colour, planned minutes (used by the timer)
  const ROOMS = [
    { id: "prologue", label: "Prologue · Le hall, la nuit", wall: "--wall-prologue", minutes: 2 },
    { id: "plan", label: "Salle 1 · Le plan", wall: "--wall-plan", minutes: 4 },
    { id: "archives", label: "Salle 2 · Les archives", wall: "--wall-archives", minutes: 3 },
    { id: "workshop", label: "Salle 3 · L'atelier", wall: "--wall-workshop", minutes: 3 },
    { id: "inspector", label: "Salle 4 · Le bureau de l'Inspecteur", wall: "--wall-inspector", minutes: 6 },
    { id: "attempts", label: "Salle 5 · Les essais ratés", wall: "--wall-attempts", minutes: 4 },
    { id: "rhythm", label: "Salle 6 · Le bon rythme", wall: "--wall-rhythm", minutes: 4 },
    { id: "hanging", label: "Salle 7 · L'accrochage", wall: "--wall-hanging", minutes: 2 },
    { id: "epilogue", label: "Épilogue · L'arrestation", wall: "--wall-prologue", minutes: 3 },
    { id: "reserves", label: "Les réserves · écrans de secours", wall: "--wall-reserves", minutes: 0 },
  ];
  const CAST = {
    curator: { name: "Le Conservateur", img: "assets/cast/curator.jpg" },
    forger: { name: "Le Faussaire", img: "assets/cast/forger.jpg" },
    inspector: { name: "L'Inspecteur", img: "assets/cast/inspector.jpg" },
  };

  const stage = document.getElementById("stage");
  const slides = Array.from(document.querySelectorAll(".slide"));
  const roomLabel = document.getElementById("room-label");
  const counter = document.getElementById("counter");
  const floorplan = document.getElementById("floorplan");
  const overview = document.getElementById("overview");
  const timerEl = document.getElementById("timer");

  const coreCount = slides.filter((s) => s.dataset.room !== "reserves").length;
  let current = 0;
  let step = 0;
  let returnTo = null;        // slide to come back to after a detour in the reserves
  let startedAt = null;

  // ---------------------------------------------------------------- scaling
  function fit() {
    const k = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
    stage.style.transform = "scale(" + k + ")";
  }
  window.addEventListener("resize", fit);
  fit();

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
    roomLabel.textContent = room.label;
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
    for (const r of ROOMS) { total += r.minutes; if (r.id === id) break; }
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
      if (slides[current].dataset.room !== "reserves") { returnTo = current; goToRoom("reserves"); }
    }
    else if (k === "b" || k === "B") { if (returnTo !== null) { show(returnTo, "last"); returnTo = null; } }
    else if (k === "f" || k === "F") {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
    }
    else if (k === "t" || k === "T") { timerEl.hidden = !timerEl.hidden; }
  });

  // Links inside the page that jump to a slide or a room: <button data-goto="reserve-fid"> or data-room-goto="plan"
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

  window.Deck = { next: next, prev: prev, show: show, slides: slides, current: () => current, step: () => step };

  // Widgets register on DOMContentLoaded (widgets.js); the first slide is shown once they are ready
  window.addEventListener("load", () => {
    const fromHash = parseInt((location.hash || "").slice(1), 10);
    show(Number.isFinite(fromHash) ? fromHash - 1 : 0);
  });
})();
