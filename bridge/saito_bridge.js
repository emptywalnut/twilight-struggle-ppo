"use strict";

function jqObject(selector, currentId = null) {
  const adapter = global.__headlessAdapter;
  const api = {
    selector,
    currentId,
    css: () => api,
    hide: () => api,
    show: () => api,
    text: () => api,
    html: () => api,
    delay: () => api,
    dequeue: () => api,
    remove: () => { if (adapter) adapter.remove(selector); return api; },
    queue: (fn) => { if (typeof fn === "function") fn.call(api); return api; },
    off: (event = null) => { if (adapter) adapter.off(selector, event); return api; },
    on: (event, handler) => { if (adapter) adapter.on(selector, event, handler); return api; },
    addClass: (cls) => { if (adapter) adapter.addClass(selector, cls); return api; },
    removeClass: (cls) => { if (adapter) adapter.removeClass(selector, cls); return api; },
    attr: (name) => {
      if (name === "id") return currentId || (String(selector).startsWith("#") ? String(selector).slice(1).split(/\s+/)[0] : "");
      if (name === "class") return "";
      return "";
    },
  };
  return api;
}

global.$ = function jqueryStub(selector) {
  if (selector && typeof selector === "object" && selector.__headlessId) return jqObject(`#${selector.__headlessId}`, selector.__headlessId);
  return jqObject(selector);
};
global.document = global.document || { querySelector: () => null, getElementById: () => null };
global.document.querySelector = global.document.querySelector || (() => null);
global.document.getElementById = global.document.getElementById || (() => null);
global.window = global.window || {};
global.salert = global.salert || (() => {});
global.sconfirm = global.sconfirm || (async () => true);

const readline = require("readline");
const Twilight = require("../third_party/saito-lite-rust-materialized/mods/twilight/twilight.js");

const PRESET = { id: "optional_us_plus_2", deck: "optional", usbonus: 2, backend: "saito" };
const MAX_ROUNDS = 10;

class HeadlessDecisionAdapter {
  constructor(kernel) {
    this.kernel = kernel;
    this.pending = null;
    this.waitingList = null;
    this.handlers = new Map();
    this.classes = new Map();
    this.lastStatus = "";
    this.backButtonCallback = null;
    this.boundedCountryPrompt = null;
    this.boundedCountryCounts = new Map();
  }

  reset() {
    this.pending = null;
    this.waitingList = null;
    this.handlers.clear();
    this.classes.clear();
    this.lastStatus = "";
    this.backButtonCallback = null;
    this.boundedCountryPrompt = null;
    this.boundedCountryCounts.clear();
  }

  setStatus(message) {
    this.lastStatus = String(message || "");
    if (this.lastStatus.toLowerCase().includes("submitting moves")) {
      this.pending = null;
      this.waitingList = null;
      this.handlers.clear();
      this.classes.clear();
      this.backButtonCallback = null;
      this.boundedCountryPrompt = null;
      this.boundedCountryCounts.clear();
      return;
    }
    this.updateBoundedCountryPrompt();
  }

  setList(message, items, callback = null) {
    this.setStatus(message);
    if (!Array.isArray(items)) items = [];
    const choices = items.map((item) => ({
      type: "saito_choice",
      decision: "list",
      value: String(item),
      label: this.labelForValue(String(item)),
    }));
    this.pending = {
      prompt: this.lastStatus,
      choices,
      callback: callback ? async (choice) => callback.call(this.kernel.twilight, choice.value) : null,
    };
    if (!callback) this.waitingList = this.pending;
    this.refreshBackButtonChoice();
  }

  setOptions(message, html, callback = null) {
    this.setStatus(message);
    const ids = this.parseOptionIds(html);
    let choices = ids.map((id) => ({
      type: "saito_choice",
      decision: "option",
      value: id,
      label: id,
    }));
    choices = this.filterOptionChoicesForContext(choices);
    this.pending = {
      prompt: this.lastStatus,
      choices,
      callback: callback ? async (choice) => callback.call(this.kernel.twilight, choice.value) : null,
    };
    if (!callback) this.waitingList = this.pending;
    this.refreshBackButtonChoice();
  }

  filterOptionChoicesForContext(choices) {
    const status = this.normalizeStatus(this.lastStatus);
    if (!/\bplays \d+ ops:/.test(status)) return choices;
    const countryIds = Object.keys(this.kernel.twilight?.countries || {});
    const hasLegalCoupTarget = countryIds.some((id) => this.kernel.isLegalCoupTarget(id));
    const hasLegalRealignTarget = countryIds.some((id) => this.kernel.isLegalRealignTarget(id));
    const filtered = choices.filter((choice) => {
      if (choice.value === "coup") return hasLegalCoupTarget;
      if (choice.value === "realign") return hasLegalRealignTarget;
      return true;
    });
    if (!filtered.length) {
      if (this.kernel.isTargetlessJuntaFreeOpsContext(choices, hasLegalCoupTarget, hasLegalRealignTarget)) {
        return [{
          type: "saito_bridge",
          decision: "skip_targetless_junta_ops",
          value: "__skip_targetless_junta_ops__",
          label: "skip unavailable Junta coup/realign",
        }];
      }
      if (this.kernel.isTargetlessOpsContext(choices, hasLegalCoupTarget, hasLegalRealignTarget)) {
        return [{
          type: "saito_bridge",
          decision: "skip_targetless_ops",
          value: "__skip_targetless_ops__",
          label: "skip unavailable OPS target modes",
        }];
      }
    }
    return filtered;
  }

  attachControlCallback(callback) {
    if (this.waitingList) {
      this.waitingList.callback = async (choice) => callback.call(this.kernel.twilight, choice.value);
      this.pending = this.waitingList;
      this.waitingList = null;
    } else if (this.pending && !this.pending.callback) {
      this.pending.callback = async (choice) => callback.call(this.kernel.twilight, choice.value);
    } else {
      this.controlCallback = callback;
    }
  }

  bindBackButton(callback) {
    this.backButtonCallback = typeof callback === "function"
      ? async () => callback.call(this.kernel.twilight)
      : null;
    this.refreshBackButtonChoice();
  }

  cancelBackButton() {
    this.backButtonCallback = null;
    this.removeBackButtonChoice();
  }

  refreshBackButtonChoice() {
    if (!this.pending || !this.backButtonCallback) return;
    const prompt = this.normalizeStatus(this.pending.prompt || this.lastStatus);
    if (!prompt.includes("choose card to reclaim")) return;
    if (this.pending.choices.some((choice) => choice.type === "saito_back" && choice.value === "__back__")) return;
    this.pending.choices.push({
      type: "saito_back",
      decision: "back",
      value: "__back__",
      label: "No card",
    });
    this.pending.backButtonCallback = this.backButtonCallback;
  }

  removeBackButtonChoice() {
    for (const target of [this.pending, this.waitingList]) {
      if (!target?.choices) continue;
      target.choices = target.choices.filter((choice) => !(choice.type === "saito_back" && choice.value === "__back__"));
      delete target.backButtonCallback;
    }
  }

  parseOptionIds(html) {
    const ids = [];
    const re = /id=["']([^"']+)["']/g;
    let match;
    while ((match = re.exec(String(html || "")))) ids.push(match[1]);
    return [...new Set(ids)];
  }

  labelForValue(value) {
    const card = this.kernel.cardsById?.[value];
    if (card) return card.name || value;
    const country = this.kernel.twilight?.countries?.[value];
    if (country) return country.name || value;
    return value;
  }

  on(selector, event, handler) {
    const key = `${selector}|${event}`;
    this.handlers.set(key, { selector: String(selector), event, handler });
    this.refreshDomPending(event);
  }

  off(selector, event = null) {
    const s = String(selector);
    for (const key of [...this.handlers.keys()]) {
      if (key.startsWith(`${s}|`) && (event == null || key.endsWith(`|${event}`))) this.handlers.delete(key);
    }
    if (this.pending?.sourceSelector === s) this.pending = null;
  }

  addClass(selector, cls) {
    const id = this.idFromSelector(selector);
    if (!id) return;
    if (!this.classes.has(cls)) this.classes.set(cls, new Set());
    this.classes.get(cls).add(id);
  }

  addElementFromHtml(html) {
    const id = /id=["']([^"']+)["']/.exec(String(html || ""))?.[1];
    const classes = /class=["']([^"']+)["']/.exec(String(html || ""))?.[1];
    if (!id || !classes) return;
    for (const cls of classes.split(/\s+/).filter(Boolean)) this.addClass(`#${id}`, cls);
  }

  removeClass(selector, cls) {
    if (String(selector).startsWith(".")) {
      const className = String(selector).slice(1);
      this.classes.delete(className);
      return;
    }
    const id = this.idFromSelector(selector);
    if (id && this.classes.has(cls)) this.classes.get(cls).delete(id);
  }

  remove(selector) {
    const s = String(selector || "");
    if (s.startsWith(".")) {
      this.classes.delete(s.slice(1));
      return;
    }
    const id = this.idFromSelector(s);
    if (id) {
      for (const ids of this.classes.values()) ids.delete(id);
    }
  }

  idFromSelector(selector) {
    const s = String(selector || "");
    if (!s.startsWith("#")) return null;
    return s.slice(1).split(/\s|>/)[0];
  }

  idsForSelector(selector) {
    const s = String(selector || "");
    if (s.startsWith("#")) return [this.idFromSelector(s)].filter(Boolean);
    if (s === ".country") return Object.keys(this.kernel.twilight.countries || {});
    if (s.startsWith(".")) return [...(this.classes.get(s.slice(1)) || new Set())];
    return [];
  }

  filterIdsForContext(ids, selector, event) {
    const status = this.lastStatus.toLowerCase();
    if (status.includes("extra 1 op available for southeast asia")) {
      if (event !== "mouseup") return [];
      return ids.filter((id) => {
        const country = this.kernel.twilight.countries[id];
        const saved = this.kernel.twilight.game?.countries?.[id] || country;
        return country?.region === "seasia" && (country?.place === 1 || saved?.place === 1);
      });
    }
    if (status.includes("extra 1 op available for asia")) {
      if (event !== "mouseup") return [];
      return ids.filter((id) => {
        const country = this.kernel.twilight.countries[id];
        const saved = this.kernel.twilight.game?.countries?.[id] || country;
        return (country?.region === "asia" || country?.region === "seasia") && (country?.place === 1 || saved?.place === 1);
      });
    }
    if (status.includes("coup")) return ids.filter((id) => this.kernel.isLegalCoupTarget(id));
    if (status.includes("realign")) return ids.filter((id) => this.kernel.isLegalRealignTarget(id));
    const bounded = this.currentBoundedCountryPrompt();
    if (bounded) return ids.filter((id) => this.isBoundedCountryLegal(id, bounded));
    if (status.includes("place") || event === "mouseup") {
      if (/^place \d+ influence$/.test(status) && event !== "mouseup") return [];
      return ids.filter((id) => {
        const live = this.kernel.twilight.countries[id] || this.kernel.twilight.game?.countries?.[id];
        const saved = this.kernel.twilight.game?.countries?.[id] || live;
        return live?.place === 1 || saved?.place === 1;
      });
    }
    if (selector === ".country") return ids;
    return ids;
  }

  normalizeStatus(message = this.lastStatus) {
    return String(message || "")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
  }

  currentBoundedCountryPrompt() {
    return this.detectBoundedCountryPrompt(this.lastStatus);
  }

  updateBoundedCountryPrompt() {
    const next = this.currentBoundedCountryPrompt();
    if (!next) {
      this.boundedCountryPrompt = null;
      this.boundedCountryCounts.clear();
      return;
    }
    if (!this.boundedCountryPrompt || this.boundedCountryPrompt.key !== next.key) {
      this.boundedCountryPrompt = next;
      this.boundedCountryCounts.clear();
    } else {
      this.boundedCountryPrompt = next;
    }
  }

  detectBoundedCountryPrompt(message) {
    const status = this.normalizeStatus(message);
    if (!status) return null;
    if (/place \d+ influence in africa or southeast asia \(1 per country\)/.test(status)) {
      return { key: "place-africa-seasia-1", mode: "place", regions: new Set(["africa", "seasia"]), maxPerCountry: 1 };
    }
    if (/place \d+ influence in non-us controlled countries in eastern europe \(1 per country\)/.test(status)) {
      return { key: "comecon-eastern-europe-1", mode: "place", countries: this.kernel.easternEuropeCountries(), forbiddenControl: "us", maxPerCountry: 1 };
    }
    if (/place 1 influence in each of \d+ non ussr-controlled countries in western europe/.test(status)) {
      return { key: "marshall-western-europe-1", mode: "place", countries: this.kernel.westernEuropeCountries(), forbiddenControl: "ussr", maxPerCountry: 1 };
    }
    if (/place 2 influence in central or south america/.test(status)) {
      return { key: "junta-central-south-america", mode: "place", regions: new Set(["camerica", "samerica"]), maxPerCountry: 1, requiresPlaceFlag: false };
    }
    if (/remove \d+ us influence from western europe \(max 1 per country\)/.test(status)) {
      return { key: "remove-us-western-europe-1", mode: "remove", countries: this.kernel.westernEuropeCountries(), influenceSide: "us", maxPerCountry: 1 };
    }
    if (/remove four influence from israel, uk or france/.test(status)) {
      return { key: "suez-remove-us-2", mode: "remove", countries: new Set(["israel", "uk", "france"]), influenceSide: "us", maxPerCountry: 2 };
    }
    if (/remove \d+ us influence from western europe \(max 2 per country\)/.test(status)) {
      return { key: "remove-us-western-europe-2", mode: "remove", countries: this.kernel.westernEuropeCountries(), influenceSide: "us", maxPerCountry: 2 };
    }
    if (/remove \d+ ussr influence from western europe \(max 2 per country\)/.test(status)) {
      return { key: "remove-ussr-western-europe-2", mode: "remove", countries: this.kernel.westernEuropeCountries(), influenceSide: "ussr", maxPerCountry: 2 };
    }
    if (/remove \d+ ussr influence from non-european countries \(max 2 per country\)/.test(status)) {
      return { key: "remove-ussr-non-europe-2", mode: "remove", countries: this.kernel.nonEuropeCountries(), influenceSide: "ussr", maxPerCountry: 2 };
    }
    if (/remove \d+ us influence from non-european countries \(max 2 per country\)/.test(status)) {
      return { key: "remove-us-non-europe-2", mode: "remove", countries: this.kernel.nonEuropeCountries(), influenceSide: "us", maxPerCountry: 2 };
    }
    return null;
  }

  isBoundedCountryLegal(id, bounded) {
    const country = this.kernel.twilight.countries?.[id];
    if (!country) return false;
    const alreadySelected = this.boundedCountryCounts.get(id) || 0;
    if (alreadySelected >= bounded.maxPerCountry) return false;
    if (bounded.countries && !bounded.countries.has(id)) return false;
    if (bounded.regions && !bounded.regions.has(country.region)) return false;
    if (bounded.forbiddenControl && this.kernel.control(id) === bounded.forbiddenControl) return false;
    if (bounded.influenceSide && Number(country[bounded.influenceSide] || 0) <= 0) return false;
    if (bounded.requiresPlaceFlag === false) return true;
    return country.place === 1 || (this.kernel.twilight.game?.countries?.[id]?.place === 1);
  }

  refreshDomPending(event = null) {
    const choices = [];
    const seen = new Set();
    for (const entry of this.handlers.values()) {
      if (event != null && entry.event !== event) continue;
      const ids = this.filterIdsForContext(this.idsForSelector(entry.selector), entry.selector, entry.event);
      for (const id of ids) {
        const key = `${entry.selector}|${entry.event}|${id}`;
        if (seen.has(key)) continue;
        seen.add(key);
        choices.push({
          type: "saito_dom",
          decision: entry.event === "mouseup" ? "country_mouseup" : "country_click",
          value: id,
          label: this.labelForValue(id),
          selector: entry.selector,
          event: entry.event,
        });
      }
    }
    if (!choices.length) return;
    this.pending = {
      prompt: this.lastStatus,
      choices,
      sourceSelector: "__dom__",
      event,
      callback: async (choice) => this.trigger(choice.selector, choice.event, choice.value),
    };
  }

  refreshAnyDomPending() {
    this.refreshDomPending(null);
  }

  async choose(action) {
    if (!this.pending) throw new Error("no pending Saito decision");
    const choice = this.pending.choices.find((c) => c.value === action.value && c.type === action.type);
    if (!choice) throw new Error(`invalid Saito decision choice: ${JSON.stringify(action)}`);
    const callback = this.pending.callback;
    const backButtonCallback = this.pending.backButtonCallback;
    this.pending = null;
    if (choice.type === "saito_bridge" && choice.value === "__skip_targetless_junta_ops__") {
      this.kernel.skipTargetlessJuntaFreeOps();
      return;
    }
    if (choice.type === "saito_bridge" && choice.value === "__skip_targetless_ops__") {
      this.kernel.skipTargetlessOps();
      return;
    }
    if (choice.type === "saito_back") {
      if (!backButtonCallback) throw new Error("pending Saito back-button decision has no callback");
      await backButtonCallback();
      return;
    }
    if (!callback) throw new Error("pending Saito decision has no callback");
    await callback(choice);
    this.noteBoundedCountrySelection(choice);
  }

  noteBoundedCountrySelection(choice) {
    const bounded = this.boundedCountryPrompt || this.currentBoundedCountryPrompt();
    if (!bounded || choice.type !== "saito_dom") return;
    const id = choice.value;
    if (!this.kernel.twilight.countries?.[id]) return;
    const selected = this.boundedCountryCounts.get(id) || 0;
    const next = selected + 1;
    this.boundedCountryCounts.set(id, next);
    if (next >= bounded.maxPerCountry) {
      this.kernel.twilight.countries[id].place = 0;
      if (this.kernel.twilight.game?.countries?.[id]) this.kernel.twilight.game.countries[id].place = 0;
    }
  }

  async trigger(selector, event, id) {
    const exact = this.handlers.get(`${selector}|${event}`);
    const eventPayload = {
      clientX: 0,
      clientY: 0,
      currentTarget: { id },
      target: { id },
    };
    const context = { __headlessId: id, id };
    if (event === "mouseup") {
      const down = this.handlers.get(`${selector}|mousedown`);
      if (down) await down.handler.call(context, eventPayload);
    }
    if (!exact) throw new Error(`no handler for ${selector} ${event}`);
    await exact.handler.call(context, eventPayload);
    if (String(selector).startsWith(".")) {
      const ids = this.classes.get(String(selector).slice(1));
      if (ids) ids.delete(id);
    }
  }
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function other(side) {
  return side === "us" ? "ussr" : "us";
}

function sideToPlayer(side) {
  return side === "ussr" ? 1 : 2;
}

function playerToSide(player) {
  const p = String(player);
  if (p === "1" || p === "ussr") return "ussr";
  if (p === "2" || p === "us") return "us";
  return null;
}

class SaitoTwilightKernel {
  constructor() {
    this.adapter = new HeadlessDecisionAdapter(this);
    global.__headlessAdapter = this.adapter;
    this.reset(1);
  }

  reset(seed = 1) {
    this.seed = seed;
    this.adapter.reset();
    global.__headlessAdapter = this.adapter;
    this.rand = mulberry32(seed);
    this.currentPlayer = "ussr";
    this.actionRound = 0;
    this.phase = "boot";
    this.setupRemaining = 6;
    this.setupQueueCommand = "placement\t1";
    this.headlineCards = { ussr: null, us: null };
    this.headlineOrder = [];
    this.resolvingHeadline = null;
    this.summitPending = null;
    this.tehranPending = null;
    this.voiceOfAmericaPending = null;
    this.destalinizationPending = null;
    this.suezCrisisPending = null;
    this.southAfricanUnrestPending = null;
    this.askNotPending = null;
    this.erasAdded = { early: false, mid: false, late: false };
    this.winner = null;
    this.terminalReason = null;
    this.log = [];
    this.twilight = new Twilight({});
    this.twilight.app = this.twilight.app || {};
    this.twilight.app.browser = {
      addElementToDom: (html) => this.adapter.addElementFromHtml(html),
      addElementToSelector: (html) => this.adapter.addElementFromHtml(html),
      replaceElementBySelector: () => {},
    };
    this.twilight.browser_active = 0;
    this.twilight.game.options = { deck: PRESET.deck, usbonus: PRESET.usbonus };
    this.twilight.game.players = ["ussr", "us"];
    this.twilight.game.player = 1;
    this.twilight.rollDice = (sides = 6, callback = null) => {
      const roll = Math.floor(this.rand() * sides) + 1;
      if (typeof callback === "function") callback(roll);
      return roll;
    };
    this.twilight.updateLog = (msg) => { if (msg) this.log.push(String(msg)); };
    this.twilight.saveGame = () => {};
    this.twilight.setPlayerActive = () => { this.twilight.browser_active = 1; };
    this.twilight.startClock = () => {};
    this.twilight.sendGameMoveTransaction = () => {
      const turn = Array.isArray(this.twilight.game.turn) ? [...this.twilight.game.turn] : [];
      this.twilight.game.queue.push(...turn);
      this.twilight.game.turn = [];
    };
    this.twilight.displayModal = () => {};
    this.twilight.showScoreOverlay = () => {};
    this.twilight.updateVictoryPoints = () => {};
    this.twilight.updateMilitaryOperations = () => {};
    this.twilight.updateDefcon = () => {};
    this.twilight.updateSpaceRace = () => {};
    this.twilight.hideCard = () => {};
    this.twilight.cardToText = this.twilight.cardToText || ((card) => this.cardsById?.[card]?.name || card);
    this.twilight.scale = (n) => n;
    this.patchInfluenceCallbacks();
    this.twilight.updateStatus = (message) => this.adapter.setStatus(message);
    this.twilight.updateStatusAndListCards = (message, cards, callback) => {
      this.adapter.setList(message, cards, typeof callback === "function" ? callback : null);
    };
    this.twilight.updateStatusWithOptions = (message, html, callback) => {
      this.adapter.setOptions(message, html, typeof callback === "function" ? callback : null);
    };
    this.twilight.hud.attachControlCallback = (callback) => this.adapter.attachControlCallback(callback);
    this.twilight.bindBackButtonFunction = (callback) => this.adapter.bindBackButton(callback);
    this.twilight.cancelBackButtonFunction = () => this.adapter.cancelBackButton();
    this.twilight.unbindBackButtonFunction = () => this.adapter.cancelBackButton();
    this.twilight.sendGameOverTransaction = (winner, reason = "game_over") => {
      this.winner = winner === "us" || winner === 2 ? "us" : "ussr";
      this.terminalReason = reason;
      this.twilight.game.over = 1;
    };
    this.twilight.sendStopGameTransaction = (reason = "game_stopped") => {
      const loser = this.twilight.game.player === 2 ? "us" : "ussr";
      this.winner = other(loser);
      this.terminalReason = reason;
      this.twilight.game.over = 1;
    };
    this.withSilencedConsole(() => this.twilight.initializeGame("headless"));
    this.ensureBoardPositionState();
    this.twilight.game.turn = [];
    this.twilight.moves = [];
    this.twilight.showInfluence = () => {};
    this.twilight.displayBoard = () => {};
    this.cardsById = this.returnCardsById();
    this.twilight.game.deck[0] = this.twilight.game.deck[0] || {};
    this.twilight.game.deck[0].cards = {};
    this.twilight.game.deck[0].hand = [];
    this.twilight.game.deck[0].discards = {};
    this.twilight.game.deck[0].removed = {};
    this.twilight.game.deck[0].crypt = [];
    this.deck = [];
    this.hands = { ussr: [], us: [] };
    this.discard = [];
    this.removed = [];
    this.processQueueUntilDecisionSync();
    this.syncSaitoHand();
    return this.observe(this.currentPlayer);
  }

  patchInfluenceCallbacks() {
    const wrapInfluenceMethod = (name) => {
      const original = this.twilight[name]?.bind(this.twilight);
      if (typeof original !== "function") return;
      this.twilight[name] = (country, inf, player, callback = null) => {
        let callbackCalled = false;
        const wrappedCallback = typeof callback === "function"
          ? (...args) => {
              callbackCalled = true;
              return callback(...args);
            }
          : null;
        const result = original(country, inf, player, wrappedCallback);
        if (typeof callback === "function" && !callbackCalled) {
          callback(country, player);
        }
        return result;
      };
    };
    wrapInfluenceMethod("placeInfluence");
    wrapInfluenceMethod("removeInfluence");
  }

  withSilencedConsole(fn) {
    const original = console.log;
    console.log = () => {};
    try {
      return fn();
    } finally {
      console.log = original;
    }
  }

  returnCardsById() {
    const early = this.twilight.returnEarlyWarCards(false);
    const mid = this.twilight.returnMidWarCards(false);
    const late = this.twilight.returnLateWarCards(false);
    return { ...early, ...mid, ...late, china: this.twilight.returnChinaCard() };
  }

  returnEraCards(era) {
    if (era === "early") return this.twilight.returnEarlyWarCards(false);
    if (era === "mid") return this.twilight.returnMidWarCards(false);
    if (era === "late") return this.twilight.returnLateWarCards(false);
    throw new Error(`unknown deck era: ${era}`);
  }

  ensureBoardPositionState() {
    const state = this.twilight.game.state || {};
    this.twilight.game.state = state;
    if (!Array.isArray(state.defcon_ps) || state.defcon_ps.length < 5) {
      state.defcon_ps = [
        { top: 2592, left: 1526 },
        { top: 2592, left: 1682 },
        { top: 2592, left: 1838 },
        { top: 2592, left: 1994 },
        { top: 2592, left: 2150 },
      ];
    }
    if (!Array.isArray(state.ar_ps) || state.ar_ps.length < 8) {
      state.ar_ps = [
        { top: 208, left: 920 },
        { top: 208, left: 1040 },
        { top: 208, left: 1155 },
        { top: 208, left: 1270 },
        { top: 208, left: 1390 },
        { top: 208, left: 1505 },
        { top: 208, left: 1625 },
        { top: 208, left: 1740 },
      ];
    }
    if (!Array.isArray(state.round_ps) || state.round_ps.length < 10) {
      state.round_ps = Array.from({ length: 10 }, (_, idx) => ({ top: 150, left: 3473 + idx * 154 }));
    }
    if (!Array.isArray(state.space_race_ps) || state.space_race_ps.length < 9) {
      state.space_race_ps = Array.from({ length: 9 }, (_, idx) => ({ top: 510, left: 3465 + idx * 172 }));
    }
    if (!Array.isArray(state.milops_ps) || state.milops_ps.length < 6) {
      state.milops_ps = Array.from({ length: 6 }, (_, idx) => ({ top: 2940, left: 1520 + idx * 155 }));
    }
    if (!Array.isArray(state.vp_ps) || state.vp_ps.length < 41) {
      state.vp_ps = Array.from({ length: 41 }, (_, idx) => ({ top: 2466, left: 3108 + idx * 32 }));
    }
  }

  addEraToDeck(era) {
    if (this.erasAdded[era]) return;
    const cards = this.returnEraCards(era);
    for (const id of Object.keys(cards)) {
      if (id === "china") continue;
      if (this.deck.includes(id)) continue;
      this.deck.push(id);
    }
    this.erasAdded[era] = true;
    this.shuffle(this.deck);
  }

  shuffle(items) {
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(this.rand() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
  }

  dealTo(side, target) {
    while (this.nonChinaHandCount(side) < target && this.deck.length > 0) {
      this.hands[side].push(this.deck.pop());
    }
    this.syncDeckCrypt();
  }

  nonChinaHandCount(side) {
    return this.hands[side].filter((card) => card !== "china").length;
  }

  syncDeckCrypt() {
    if (!this.twilight.game.deck[0]) this.twilight.game.deck[0] = {};
    this.twilight.game.deck[0].crypt = [...this.deck];
  }

  syncTrackedPilesFromSaito() {
    this.discard = Object.keys(this.twilight.game.deck[0]?.discards || {});
    this.removed = Object.keys(this.twilight.game.deck[0]?.removed || {});
  }

  addCardsToDrawDeck(cards) {
    for (const id of Object.keys(cards || {})) {
      if (id === "china") continue;
      if (this.hands.us.includes(id) || this.hands.ussr.includes(id)) continue;
      if (this.deck.includes(id)) continue;
      if (this.twilight.game.deck[0].discards?.[id]) continue;
      if (this.twilight.game.deck[0].removed?.[id]) continue;
      this.deck.push(id);
    }
    this.shuffle(this.deck);
    this.syncDeckCrypt();
  }

  reshuffleDiscardsIfNeeded() {
    const target = this.twilight.game.state.round >= 4 ? 9 : 8;
    const needed =
      Math.max(0, target - this.nonChinaHandCount("us")) +
      Math.max(0, target - this.nonChinaHandCount("ussr"));
    if (this.deck.length >= needed) return;
    const discards = Object.keys(this.twilight.game.deck[0]?.discards || {});
    if (!discards.length) return;
    for (const id of discards) {
      if (id === "china") continue;
      if (!this.deck.includes(id) && !this.hands.us.includes(id) && !this.hands.ussr.includes(id)) {
        this.deck.push(id);
      }
    }
    this.twilight.game.deck[0].discards = {};
    this.shuffle(this.deck);
    this.syncTrackedPilesFromSaito();
    this.syncDeckCrypt();
  }

  syncSaitoHand() {
    this.twilight.game.player = sideToPlayer(this.currentPlayer);
    this.twilight.game.deck[0].hand = this.hands[this.currentPlayer];
    this.twilight.game.deck[0].cards = this.twilight.game.deck[0].cards || {};
    this.twilight.game.deck[0].discards = this.twilight.game.deck[0].discards || {};
    this.twilight.game.deck[0].removed = this.twilight.game.deck[0].removed || {};
    this.syncDeckCrypt();
  }

  countriesArray() {
    return Object.entries(this.twilight.countries).map(([id, c], idx) => ({
      id,
      idx,
      name: c.name,
      region: c.region,
      stability: Number(c.control),
      bg: Number(c.bg),
      us: Number(c.us),
      ussr: Number(c.ussr),
      place: Number(c.place || 0),
      game_place: Number(this.twilight.game?.countries?.[id]?.place || 0),
      control: this.control(id),
      neighbours: c.neighbours || [],
    }));
  }

  cardsArray() {
    return Object.entries(this.cardsById).map(([id, c], idx) => ({
      id,
      idx,
      name: c.name,
      side: c.player,
      player: c.player,
      ops: Number(c.ops),
      scoring: Boolean(c.scoring),
      recurring: Boolean(c.recurring),
      era: c.p === 0 ? "early" : c.p === 1 ? "mid" : c.p === 2 ? "late" : "special",
    }));
  }

  control(countryId) {
    if (this.twilight.isControlled("us", countryId)) return "us";
    if (this.twilight.isControlled("ussr", countryId)) return "ussr";
    return "none";
  }

  easternEuropeCountries() {
    return new Set(["finland", "poland", "eastgermany", "austria", "czechoslovakia", "bulgaria", "hungary", "romania", "yugoslavia"]);
  }

  westernEuropeCountries() {
    return new Set(["canada", "uk", "sweden", "france", "benelux", "westgermany", "spain", "italy", "greece", "turkey", "denmark", "norway", "finland", "austria"]);
  }

  nonEuropeCountries() {
    return new Set(Object.entries(this.twilight.countries || {})
      .filter(([, country]) => country.region !== "europe")
      .map(([id]) => id));
  }

  legalSetupCountries() {
    if (this.phase === "setup_ussr") return [...this.easternEuropeCountries()];
    if (this.phase === "setup_us") return [...this.westernEuropeCountries()];
    if (this.phase === "setup_us_bonus") {
      return Object.entries(this.twilight.countries)
        .filter(([, country]) => Number(country.us || 0) > 0)
        .map(([id]) => id);
    }
    return [];
  }

  setupPrompt() {
    if (this.phase === "setup_ussr") return `USSR initial placement: place ${this.setupRemaining} influence in Eastern Europe`;
    if (this.phase === "setup_us") return `US initial placement: place ${this.setupRemaining} influence in Western Europe`;
    if (this.phase === "setup_us_bonus") return `US optional +2 placement: place ${this.setupRemaining} influence in countries with existing American influence`;
    return "";
  }

  createSetupActions() {
    const prompt = this.setupPrompt();
    this.adapter.setStatus(prompt);
    return this.legalSetupCountries().map((id) => ({
      type: "saito_dom",
      decision: "country_click",
      value: id,
      label: this.adapter.labelForValue(id),
      selector: "__setup__",
      event: "click",
      prompt,
    }));
  }

  legalActions() {
    if (this.winner) return [];
    this.processQueueUntilDecisionSync();
    if (this.phase.startsWith("setup_")) return this.createSetupActions();
    if (this.phase === "headline_resolve") this.settleHeadlineResolutionSync();
    if (this.phase === "headline_resolve" && (!this.adapter.pending || !this.adapter.pending.choices?.length)) {
      this.throwBridgeStall("headline resolution has no exposed decision");
    }
    this.syncCountryPlaceFlags();
    this.ensurePendingDecision();
    if (!this.adapter.pending || !this.adapter.pending.choices?.length) {
      this.createActionCardDecisionFallback();
    }
    this.syncCountryPlaceFlags();
    return (this.adapter.pending?.choices || []).map((choice) => ({
      ...choice,
      prompt: this.adapter.pending?.prompt || "",
    }));
  }

  queueTop() {
    return this.twilight.game.queue[this.twilight.game.queue.length - 1] || "";
  }

  queueCommand() {
    return String(this.queueTop()).split("\t")[0];
  }

  currentActionRound() {
    return Math.max(0, Number(this.twilight.game.state?.turn_in_round || 0) - 1);
  }

  throwBridgeStall(reason) {
    const debug = {
      reason,
      phase: this.phase,
      currentPlayer: this.currentPlayer,
      turn: this.twilight.game.state?.round,
      actionRound: this.currentActionRound(),
      prompt: this.adapter.lastStatus || "",
      queueTop: this.queueTop(),
      queueTail: this.twilight.game.queue.slice(-8),
      pendingChoices: this.adapter.pending?.choices?.map((choice) => choice.value) || [],
      resolvingHeadline: this.resolvingHeadline,
      logTail: this.log.slice(-8),
    };
    throw new Error(`Saito bridge stalled: ${JSON.stringify(debug)}`);
  }

  processQueueUntilDecisionSync() {
    for (let guard = 0; guard < 500; guard++) {
      if (this.winner || this.adapter.pending || this.phase.startsWith("setup_") || this.phase === "headline_ussr" || this.phase === "headline_us" || this.phase === "headline_resolve") return;
      if (!this.twilight.game.queue.length) return;
      if (this.handleBridgeQueueCommandSync()) continue;
      const actor = this.actorForQueueTop();
      if (actor) this.currentPlayer = actor;
      this.syncSaitoHand();
      const before = this.progressSignature();
      this.ensureBoardPositionState();
      this.withSilencedConsole(() => this.twilight.handleGameLoop());
      this.syncTrackedPilesFromSaito();
      if (this.adapter.pending || this.winner) return;
      if (this.progressSignature() === before) {
        this.adapter.refreshAnyDomPending();
        if (this.adapter.pending) return;
        if (this.retryEventWithAlternateActors()) continue;
        this.throwBridgeStall("Saito queue did not advance");
      }
    }
    this.throwBridgeStall("queue processing exceeded guard limit");
  }

  handleBridgeQueueCommandSync() {
    const queue = this.twilight.game.queue;
    const move = String(queue[queue.length - 1] || "");
    const parts = move.split("\t");
    const command = parts[0];

    if (command === "NOTIFY" || command === "notify" || command === "ACKNOWLEDGE") {
      this.log.push(move.slice(command.length + 1));
      queue.pop();
      return true;
    }

    if (command === "init" || command === "READY" || command === "OBSERVER_CHECKPOINT" || command === "observer_cards_update" || command === "update_observers" || command === "stage" || command === "modal") {
      queue.pop();
      return true;
    }

    if (command === "DECK") {
      const cards = JSON.parse(parts.slice(2).join("\t") || "{}");
      this.twilight.game.deck[0].cards = { ...(this.twilight.game.deck[0].cards || {}), ...cards };
      this.addCardsToDrawDeck(cards);
      queue.pop();
      return true;
    }

    if (command === "DECKBACKUP" || command === "DECKRESTORE" || command === "DECKENCRYPT" || command === "DECKXOR") {
      queue.pop();
      return true;
    }

    if (command === "SHUFFLE") {
      this.shuffle(this.deck);
      this.syncDeckCrypt();
      queue.pop();
      return true;
    }

    if (command === "DEAL") {
      const side = playerToSide(parts[2]);
      const count = Number(parts[3] || 0);
      if (!side) this.throwBridgeStall(`unknown DEAL side: ${move}`);
      this.dealTo(side, this.nonChinaHandCount(side) + count);
      queue.pop();
      return true;
    }

    if (command === "discard") {
      this.handleDiscardQueueCommand(parts);
      return true;
    }

    if (command === "placement" || command === "placement_bonus") {
      this.beginSetupFromQueue(move);
      return true;
    }

    if (command === "event" && parts[2] === "summit") {
      this.beginHeadlessSummitEvent(parts[1]);
      return true;
    }

    if (command === "event" && parts[2] === "tehran") {
      this.beginHeadlessTehranEvent(parts[1]);
      return true;
    }

    if (command === "event" && parts[2] === "voiceofamerica") {
      this.beginHeadlessVoiceOfAmericaEvent(parts[1]);
      return true;
    }

    if (command === "event" && parts[2] === "suezcrisis") {
      this.beginHeadlessSuezCrisisEvent(parts[1]);
      return true;
    }

    if (command === "event" && parts[2] === "destalinization") {
      this.beginHeadlessDestalinizationEvent(parts[1]);
      return true;
    }

    if (command === "event" && parts[2] === "southafrican") {
      this.beginHeadlessSouthAfricanEvent(parts[1]);
      return true;
    }

    if (command === "event" && parts[2] === "asknot") {
      this.beginHeadlessAskNotEvent(parts[1]);
      return true;
    }

    if (command === "tehran") {
      this.beginHeadlessTehranQueueCommand(parts);
      return true;
    }

    if (command === "headline") {
      queue.pop();
      this.phase = "headline_ussr";
      this.currentPlayer = "ussr";
      this.headlineCards = { ussr: null, us: null };
      this.headlineOrder = [];
      this.resolvingHeadline = null;
      this.adapter.reset();
      return true;
    }

    if (command === "clear") {
      queue.pop();
      if (parts[1] === "headline") this.clearHeadlineState();
      return true;
    }

    if (command === "resolve" && parts[1] === "ops" && !this.hasQueuedCommandBelowTop("ops")) {
      this.log.push("Bridge cleared orphaned resolve ops marker");
      while (queue.length > 0) {
        const top = String(queue[queue.length - 1] || "").split("\t");
        if (top[0] !== "resolve" || top[1] !== "ops") break;
        queue.pop();
      }
      return true;
    }

    if (command === "turn") {
      this.actionRound += 1;
      return false;
    }

    if (command === "deal") {
      const side = playerToSide(parts[1]);
      const target = this.twilight.game.state.round >= 4 ? 9 : 8;
      if (side) this.dealTo(side, target);
      queue.pop();
      return true;
    }

    if (command === "reshuffle") {
      this.reshuffleDiscardsIfNeeded();
      queue.pop();
      return true;
    }

    if (command === "sharehandsize") {
      const side = playerToSide(parts[1]);
      if (side) {
        const cards = this.hands[side].filter((card) => card !== "china");
        this.twilight.game.state[side === "ussr" ? "player1_hold_cards" : "player2_hold_cards"] = cards;
        this.twilight.game.state.opponent_cards_in_hand = cards.length;
      }
      queue.pop();
      return true;
    }

    if (command === "dynamic_deck_management" || command === "deckaddcards" || command === "bgs" || command === "flush") {
      queue.pop();
      return true;
    }

    if (command === "final_scoring") {
      this.withSilencedConsole(() => this.twilight.handleGameLoop());
      this.checkTerminal();
      return true;
    }

    return false;
  }

  beginHeadlessSummitEvent(player = "both") {
    const queue = this.twilight.game.queue;
    queue.pop();
    const side = playerToSide(player);
    const stats = this.twilight.game.state?.stats;
    if (stats && side) {
      const key = side === "us" ? "us_events_ops" : "ussr_events_ops";
      stats[key] = Number(stats[key] || 0) + this.cardOps("summit");
    }
    this.twilight.game.state.event_name = this.cardName("summit");
    this.log.push(`${String(player).toUpperCase()} triggers ${this.cardName("summit")} as an event`);

    const usBase = this.twilight.rollDice(6);
    const ussrBase = this.twilight.rollDice(6);
    let usRoll = usBase;
    let ussrRoll = ussrBase;
    this.log.push(`${this.cardName("summit")}: US rolls ${usBase}`);
    this.log.push(`${this.cardName("summit")}: USSR rolls ${ussrBase}`);

    const regions = [
      ["europe", "Europe"],
      ["mideast", "Middle-East"],
      ["asia", "Asia"],
      ["africa", "Africa"],
      ["camerica", "Central America"],
      ["samerica", "South America"],
    ];
    for (const [region, label] of regions) {
      if (this.twilight.doesPlayerDominateRegionForSummit("ussr", region) == 1) {
        this.log.push(`${label}: USSR +1 bonus`);
        ussrRoll += 1;
      }
    }
    for (const [region, label] of regions) {
      if (this.twilight.doesPlayerDominateRegionForSummit("us", region) == 1) {
        this.log.push(`${label}: US +1 bonus`);
        usRoll += 1;
      }
    }

    this.log.push(`${this.cardName("summit")}: US result ${usRoll} (+${usRoll - usBase} bonus)`);
    this.log.push(`${this.cardName("summit")}: USSR result ${ussrRoll} (+${ussrRoll - ussrBase} bonus)`);
    if (usRoll === ussrRoll) {
      this.log.push(`${this.cardName("summit")}: no winner`);
      this.finalizeEventCard("summit");
      return;
    }

    const winner = usRoll > ussrRoll ? "us" : "ussr";
    this.summitPending = { winner };
    this.currentPlayer = winner;
    this.syncSaitoHand();
    this.adapter.setOptions(
      `You win the ${this.cardName("summit")}:`,
      '<ul><li class="option" id="raise">raise DEFCON</li><li class="option" id="lower">lower DEFCON</li><li class="option" id="same">do not change</li></ul>',
      (choice) => this.chooseHeadlessSummit(choice),
    );
  }

  chooseHeadlessSummit(choice) {
    const pending = this.summitPending;
    if (!pending) return;
    if (!["raise", "lower", "same"].includes(choice)) throw new Error(`invalid ${this.cardName("summit")} choice: ${choice}`);
    if (pending.winner === "us") {
      this.twilight.game.state.vp += 2;
      this.log.push(`US receives 2 VP from ${this.cardName("summit")}`);
    } else {
      this.twilight.game.state.vp -= 2;
      this.log.push(`USSR receives 2 VP from ${this.cardName("summit")}`);
    }
    if (choice === "raise") {
      this.twilight.game.state.defcon = Math.min(5, Number(this.twilight.game.state.defcon || 0) + 1);
      this.log.push("DEFCON is raised by 1");
    } else if (choice === "lower") {
      this.twilight.lowerDefcon();
      this.log.push("DEFCON is lowered by 1");
    } else {
      this.log.push("DEFCON left untouched");
    }
    this.summitPending = null;
    this.finalizeEventCard("summit");
  }

  beginHeadlessTehranEvent(player = "us") {
    const queue = this.twilight.game.queue;
    queue.pop();
    const side = playerToSide(player) || "us";
    const stats = this.twilight.game.state?.stats;
    if (stats) {
      const key = side === "us" ? "us_events_ops" : "ussr_events_ops";
      stats[key] = Number(stats[key] || 0) + this.cardOps("tehran");
    }
    this.twilight.game.state.event_name = this.cardName("tehran");
    this.log.push(`${side.toUpperCase()} triggers ${this.cardName("tehran")} as an event`);
    this.resolveHeadlessTehran();
  }

  beginHeadlessTehranQueueCommand(parts) {
    const queue = this.twilight.game.queue;
    const keysnum = Math.max(0, Number(parts[2] || 0));
    queue.pop();
    if (parts[1] === "ussr") {
      for (let i = 0; i < keysnum; i++) queue.pop();
      this.resolveHeadlessTehran();
      return;
    }
    for (let i = 0; i < keysnum; i++) queue.pop();
  }

  resolveHeadlessTehran() {
    const state = this.twilight.game.state;
    const middleEastCountries = ["egypt", "libya", "israel", "lebanon", "syria", "iraq", "iran", "jordan", "gulfstates", "saudiarabia"];
    const hasUsControl = middleEastCountries.some((country) => this.twilight.isControlled?.("us", country) === 1);
    if (!hasUsControl) {
      this.log.push("US does not control any Middle-East Countries");
      this.finalizeEventCard("tehran");
      return;
    }

    state.events.ourmanintehran = 1;
    this.finalizeEventCard("tehran");

    const count = Math.min(5, this.deck.length);
    const options = this.deck.slice(-count).reverse();
    if (!options.length) {
      this.log.push(`${this.cardName("tehran")} found no draw-deck cards to review`);
      return;
    }

    this.currentPlayer = "us";
    this.syncSaitoHand();
    this.tehranPending = { options, discarded: [] };
    this.offerHeadlessTehranDecision();
  }

  offerHeadlessTehranDecision() {
    const pending = this.tehranPending;
    if (!pending) return;
    const choices = [...pending.options, "finished"];
    this.adapter.setList(`${this.cardName("tehran")}: select draw-deck cards to discard`, choices, (choice) => this.chooseHeadlessTehran(choice));
  }

  chooseHeadlessTehran(choice) {
    const pending = this.tehranPending;
    if (!pending) return;
    if (choice === "finished") {
      if (pending.discarded.length) {
        this.log.push(`${this.cardName("tehran")} discards ${pending.discarded.map((card) => this.cardName(card)).join(", ")}`);
      } else {
        this.log.push(`${this.cardName("tehran")} discards no draw-deck cards`);
      }
      this.tehranPending = null;
      return;
    }

    if (!pending.options.includes(choice)) throw new Error(`invalid ${this.cardName("tehran")} choice: ${choice}`);
    pending.options = pending.options.filter((card) => card !== choice);
    pending.discarded.push(choice);
    const idx = this.deck.lastIndexOf(choice);
    if (idx >= 0) this.deck.splice(idx, 1);
    const meta = this.cardsById[choice] || this.twilight.game.deck[0].cards?.[choice];
    if (meta) this.twilight.game.deck[0].discards[choice] = meta;
    this.syncTrackedPilesFromSaito();
    this.syncDeckCrypt();
    if (!pending.options.length) {
      this.log.push(`${this.cardName("tehran")} discards ${pending.discarded.map((card) => this.cardName(card)).join(", ")}`);
      this.tehranPending = null;
      return;
    }
    this.offerHeadlessTehranDecision();
  }

  beginHeadlessVoiceOfAmericaEvent(player = "us") {
    const queue = this.twilight.game.queue;
    queue.pop();
    const side = playerToSide(player) || "us";
    const stats = this.twilight.game.state?.stats;
    if (stats) {
      const key = side === "us" ? "us_events_ops" : "ussr_events_ops";
      stats[key] = Number(stats[key] || 0) + this.cardOps("voiceofamerica");
    }
    this.twilight.game.state.event_name = this.cardName("voiceofamerica");
    this.log.push(`${side.toUpperCase()} triggers ${this.cardName("voiceofamerica")} as an event`);

    const removable = this.voiceOfAmericaEligibleCountries()
      .reduce((total, id) => total + Math.min(2, Number(this.twilight.countries[id]?.ussr || 0)), 0);
    const remaining = Math.min(4, removable);
    if (remaining <= 0) {
      this.log.push(`${this.cardName("voiceofamerica")} has no non-European USSR influence to remove`);
      this.finalizeEventCard("voiceofamerica");
      return;
    }

    this.currentPlayer = "us";
    this.syncSaitoHand();
    this.voiceOfAmericaPending = { remaining, removedByCountry: new Map() };
    this.offerHeadlessVoiceOfAmericaDecision();
  }

  voiceOfAmericaEligibleCountries() {
    const pending = this.voiceOfAmericaPending;
    const removedByCountry = pending?.removedByCountry || new Map();
    return Object.entries(this.twilight.countries || {})
      .filter(([, country]) => country.region !== "europe")
      .filter(([, country]) => Number(country.ussr || 0) > 0)
      .map(([id]) => id)
      .filter((id) => Number(removedByCountry.get(id) || 0) < 2);
  }

  offerHeadlessVoiceOfAmericaDecision() {
    const pending = this.voiceOfAmericaPending;
    if (!pending) return;
    const choices = this.voiceOfAmericaEligibleCountries().map((id) => ({
      type: "saito_dom",
      decision: "country_click",
      value: id,
      label: this.adapter.labelForValue(id),
      selector: "__voiceofamerica__",
      event: "click",
    }));
    if (!choices.length) {
      this.log.push(`${this.cardName("voiceofamerica")} ends with no further legal removals`);
      this.voiceOfAmericaPending = null;
      this.finalizeEventCard("voiceofamerica");
      return;
    }
    this.adapter.pending = {
      prompt: `Remove ${pending.remaining} USSR influence from non-European countries (max 2 per country)`,
      choices,
      callback: async (choice) => this.chooseHeadlessVoiceOfAmerica(choice.value),
    };
    this.adapter.lastStatus = this.adapter.pending.prompt;
  }

  chooseHeadlessVoiceOfAmerica(country) {
    const pending = this.voiceOfAmericaPending;
    if (!pending) return;
    if (!this.voiceOfAmericaEligibleCountries().includes(country)) {
      throw new Error(`invalid ${this.cardName("voiceofamerica")} country: ${country}`);
    }
    const removed = Number(pending.removedByCountry.get(country) || 0) + 1;
    pending.removedByCountry.set(country, removed);
    this.twilight.removeInfluence(country, 1, "ussr");
    this.log.push(`US removes 1 USSR influence from ${this.twilight.countries[country]?.name || country}`);
    pending.remaining -= 1;
    if (pending.remaining <= 0) {
      this.voiceOfAmericaPending = null;
      this.finalizeEventCard("voiceofamerica");
      return;
    }
    this.offerHeadlessVoiceOfAmericaDecision();
  }

  beginHeadlessSuezCrisisEvent(player = "ussr") {
    const queue = this.twilight.game.queue;
    queue.pop();
    const side = playerToSide(player) || "ussr";
    const stats = this.twilight.game.state?.stats;
    if (stats) {
      const key = side === "us" ? "us_events_ops" : "ussr_events_ops";
      stats[key] = Number(stats[key] || 0) + this.cardOps("suezcrisis");
    }
    this.twilight.game.state.event_name = this.cardName("suezcrisis");
    this.log.push(`${side.toUpperCase()} triggers ${this.cardName("suezcrisis")} as an event`);

    const totalRemovable = this.suezCrisisEligibleCountries()
      .reduce((total, id) => total + Math.min(2, Number(this.twilight.countries[id]?.us || 0)), 0);
    const remaining = Math.min(4, totalRemovable);
    if (remaining <= 0) {
      this.log.push(`${this.cardName("suezcrisis")} has no US influence to remove from Israel, UK, or France`);
      this.finalizeEventCard("suezcrisis");
      return;
    }

    if (totalRemovable <= 4) {
      for (const country of this.suezCrisisEligibleCountries()) {
        const count = Math.min(2, Number(this.twilight.countries[country]?.us || 0));
        if (count <= 0) continue;
        this.twilight.removeInfluence(country, count, "us");
        this.log.push(`USSR removes ${count} US influence from ${this.twilight.countries[country]?.name || country} for ${this.cardName("suezcrisis")}`);
      }
      this.finalizeEventCard("suezcrisis");
      return;
    }

    this.currentPlayer = "ussr";
    this.syncSaitoHand();
    this.suezCrisisPending = { remaining, removedByCountry: new Map() };
    this.offerHeadlessSuezCrisisDecision();
  }

  suezCrisisEligibleCountries() {
    const pending = this.suezCrisisPending;
    const removedByCountry = pending?.removedByCountry || new Map();
    return ["uk", "france", "israel"]
      .filter((id) => Number(this.twilight.countries[id]?.us || 0) > 0)
      .filter((id) => Number(removedByCountry.get(id) || 0) < 2);
  }

  offerHeadlessSuezCrisisDecision() {
    const pending = this.suezCrisisPending;
    if (!pending) return;
    const choices = this.suezCrisisEligibleCountries().map((id) => ({
      type: "saito_dom",
      decision: "country_click",
      value: id,
      label: this.adapter.labelForValue(id),
      selector: "__suezcrisis__",
      event: "click",
    }));
    if (!choices.length) {
      this.log.push(`${this.cardName("suezcrisis")} ends with no further legal removals`);
      this.suezCrisisPending = null;
      this.finalizeEventCard("suezcrisis");
      return;
    }
    this.adapter.pending = {
      prompt: `Remove ${pending.remaining} US influence from Israel, UK, or France (max 2 per country)`,
      choices,
      callback: async (choice) => this.chooseHeadlessSuezCrisis(choice.value),
    };
    this.adapter.lastStatus = this.adapter.pending.prompt;
  }

  chooseHeadlessSuezCrisis(country) {
    const pending = this.suezCrisisPending;
    if (!pending) return;
    if (!this.suezCrisisEligibleCountries().includes(country)) {
      throw new Error(`invalid ${this.cardName("suezcrisis")} country: ${country}`);
    }
    const removed = Number(pending.removedByCountry.get(country) || 0) + 1;
    pending.removedByCountry.set(country, removed);
    this.twilight.removeInfluence(country, 1, "us");
    this.log.push(`USSR removes 1 US influence from ${this.twilight.countries[country]?.name || country} for ${this.cardName("suezcrisis")}`);
    pending.remaining -= 1;
    if (pending.remaining <= 0) {
      this.suezCrisisPending = null;
      this.finalizeEventCard("suezcrisis");
      return;
    }
    this.offerHeadlessSuezCrisisDecision();
  }

  beginHeadlessDestalinizationEvent(player = "ussr") {
    const queue = this.twilight.game.queue;
    queue.pop();
    const side = playerToSide(player) || "ussr";
    const stats = this.twilight.game.state?.stats;
    if (stats) {
      const key = side === "us" ? "us_events_ops" : "ussr_events_ops";
      stats[key] = Number(stats[key] || 0) + this.cardOps("destalinization");
    }
    this.twilight.game.state.events.destalinization_played = 1;
    this.twilight.game.state.event_name = this.cardName("destalinization");
    this.log.push(`${side.toUpperCase()} triggers ${this.cardName("destalinization")} as an event`);

    const removable = this.destalinizationRemovalCountries()
      .reduce((total, id) => total + Number(this.twilight.countries[id]?.ussr || 0), 0);
    const remainingRemove = Math.min(4, removable);
    if (remainingRemove <= 0) {
      this.log.push(`${this.cardName("destalinization")} has no USSR influence to relocate`);
      this.finalizeEventCard("destalinization");
      return;
    }

    this.currentPlayer = "ussr";
    this.syncSaitoHand();
    this.destalinizationPending = {
      phase: "remove",
      remainingRemove,
      removedCount: 0,
      remainingPlace: 0,
      placedByCountry: new Map(),
    };
    this.offerHeadlessDestalinizationDecision();
  }

  destalinizationRemovalCountries() {
    return Object.entries(this.twilight.countries || {})
      .filter(([, country]) => Number(country.ussr || 0) > 0)
      .map(([id]) => id);
  }

  destalinizationPlacementCountries() {
    const pending = this.destalinizationPending;
    const placedByCountry = pending?.placedByCountry || new Map();
    return Object.keys(this.twilight.countries || {})
      .filter((id) => this.twilight.isControlled?.("us", id) !== 1)
      .filter((id) => Number(placedByCountry.get(id) || 0) < 2);
  }

  offerHeadlessDestalinizationDecision() {
    const pending = this.destalinizationPending;
    if (!pending) return;
    const choices = (pending.phase === "remove" ? this.destalinizationRemovalCountries() : this.destalinizationPlacementCountries())
      .map((id) => ({
        type: "saito_dom",
        decision: "country_click",
        value: id,
        label: this.adapter.labelForValue(id),
        selector: "__destalinization__",
        event: "click",
      }));

    if (!choices.length) {
      if (pending.phase === "remove") {
        this.log.push(`${this.cardName("destalinization")} ends removal with no further USSR influence`);
        this.startHeadlessDestalinizationPlacement();
        return;
      }
      this.log.push(`${this.cardName("destalinization")} ends placement with no further legal countries`);
      this.destalinizationPending = null;
      this.finalizeEventCard("destalinization");
      return;
    }

    const prompt = pending.phase === "remove"
      ? `Remove ${pending.remainingRemove} USSR influence from existing countries`
      : `Add ${pending.remainingPlace} USSR influence to non-US controlled countries (max 2 per country)`;
    this.adapter.pending = {
      prompt,
      choices,
      callback: async (choice) => this.chooseHeadlessDestalinization(choice.value),
    };
    this.adapter.lastStatus = prompt;
  }

  startHeadlessDestalinizationPlacement() {
    const pending = this.destalinizationPending;
    if (!pending) return;
    pending.phase = "place";
    pending.remainingRemove = 0;
    pending.remainingPlace = pending.removedCount;
    if (pending.remainingPlace <= 0) {
      this.destalinizationPending = null;
      this.finalizeEventCard("destalinization");
      return;
    }
    this.offerHeadlessDestalinizationDecision();
  }

  chooseHeadlessDestalinization(country) {
    const pending = this.destalinizationPending;
    if (!pending) return;
    if (pending.phase === "remove") {
      if (!this.destalinizationRemovalCountries().includes(country)) {
        throw new Error(`invalid ${this.cardName("destalinization")} removal country: ${country}`);
      }
      this.twilight.removeInfluence(country, 1, "ussr");
      this.log.push(`USSR removes 1 influence from ${this.twilight.countries[country]?.name || country} for ${this.cardName("destalinization")}`);
      pending.remainingRemove -= 1;
      pending.removedCount += 1;
      if (pending.remainingRemove <= 0 || this.destalinizationRemovalCountries().length === 0) {
        this.startHeadlessDestalinizationPlacement();
        return;
      }
      this.offerHeadlessDestalinizationDecision();
      return;
    }

    if (!this.destalinizationPlacementCountries().includes(country)) {
      throw new Error(`invalid ${this.cardName("destalinization")} placement country: ${country}`);
    }
    const placed = Number(pending.placedByCountry.get(country) || 0) + 1;
    pending.placedByCountry.set(country, placed);
    this.twilight.placeInfluence(country, 1, "ussr");
    this.log.push(`USSR places 1 influence in ${this.twilight.countries[country]?.name || country} for ${this.cardName("destalinization")}`);
    pending.remainingPlace -= 1;
    if (pending.remainingPlace <= 0) {
      this.destalinizationPending = null;
      this.finalizeEventCard("destalinization");
      return;
    }
    this.offerHeadlessDestalinizationDecision();
  }

  beginHeadlessSouthAfricanEvent(player = "ussr") {
    const queue = this.twilight.game.queue;
    queue.pop();
    const side = playerToSide(player) || "ussr";
    const stats = this.twilight.game.state?.stats;
    if (stats) {
      const key = side === "us" ? "us_events_ops" : "ussr_events_ops";
      stats[key] = Number(stats[key] || 0) + this.cardOps("southafrican");
    }
    this.twilight.game.state.event_name = this.cardName("southafrican");
    this.log.push(`${side.toUpperCase()} triggers ${this.cardName("southafrican")} as an event`);

    this.currentPlayer = "ussr";
    this.syncSaitoHand();
    this.southAfricanUnrestPending = { step: "pick_mode" };
    this.offerHeadlessSouthAfricanUnrestDecision();
  }

  offerHeadlessSouthAfricanUnrestDecision() {
    const pending = this.southAfricanUnrestPending;
    if (!pending) return;
    const choices = [
      { type: "saito_choice", decision: "option", value: "southafrica", label: "2 in South Africa" },
      { type: "saito_choice", decision: "option", value: "adjacent", label: "1 in South Africa and 2 in neighboring country" },
    ];
    this.adapter.pending = {
      prompt: `${this.cardName("southafrican")}: choose placement mode`,
      choices,
      callback: async (choice) => this.chooseHeadlessSouthAfricanUnrest(choice.value),
    };
    this.adapter.lastStatus = this.adapter.pending.prompt;
  }

  chooseHeadlessSouthAfricanUnrest(value) {
    const pending = this.southAfricanUnrestPending;
    if (!pending) return;
    if (!["southafrica", "adjacent"].includes(value)) {
      throw new Error(`invalid ${this.cardName("southafrican")} option: ${value}`);
    }
    if (value === "southafrica") {
      this.twilight.placeInfluence("southafrica", 2, "ussr");
      this.log.push(`USSR places 2 influence in ${this.twilight.countries.southafrica?.name || "South Africa"} for ${this.cardName("southafrican")}`);
      this.southAfricanUnrestPending = null;
      this.finalizeEventCard("southafrican");
      return;
    }
    this.log.push(`USSR places 1 influence in ${this.twilight.countries.southafrica?.name || "South Africa"} for ${this.cardName("southafrican")}`);
    this.twilight.placeInfluence("southafrica", 1, "ussr");
    this.southAfricanUnrestPending = { step: "adjacent" };
    this.offerHeadlessSouthAfricanUnrestAdjacentDecision();
  }

  offerHeadlessSouthAfricanUnrestAdjacentDecision() {
    const pending = this.southAfricanUnrestPending;
    if (!pending) return;
    const neighbors = ["angola", "botswana"].filter((id) => Boolean(this.twilight.countries[id]));
    if (!neighbors.length) {
      this.log.push(`${this.cardName("southafrican")} has no legal neighboring countries to place influence`);
      this.southAfricanUnrestPending = null;
      this.finalizeEventCard("southafrican");
      return;
    }
    const choices = neighbors.map((id) => ({
      type: "saito_dom",
      decision: "country_click",
      value: id,
      label: this.adapter.labelForValue(id),
      selector: "__southafrican__",
      event: "click",
    }));
    this.adapter.pending = {
      prompt: `Place 2 influence in a neighboring country`,
      choices,
      callback: async (choice) => this.chooseHeadlessSouthAfricanUnrestAdjacent(choice.value),
    };
    this.adapter.lastStatus = this.adapter.pending.prompt;
  }

  chooseHeadlessSouthAfricanUnrestAdjacent(country) {
    const pending = this.southAfricanUnrestPending;
    if (!pending) return;
    if (!["angola", "botswana"].includes(country)) {
      throw new Error(`invalid ${this.cardName("southafrican")} neighboring country: ${country}`);
    }
    this.twilight.placeInfluence(country, 2, "ussr");
    this.log.push(`USSR places 2 influence in ${this.twilight.countries[country]?.name || country} for ${this.cardName("southafrican")}`);
    this.southAfricanUnrestPending = null;
    this.finalizeEventCard("southafrican");
  }

  beginHeadlessAskNotEvent(player = "us") {
    const queue = this.twilight.game.queue;
    queue.pop();
    const side = playerToSide(player) || "us";
    const stats = this.twilight.game.state?.stats;
    if (stats) {
      const key = side === "us" ? "us_events_ops" : "ussr_events_ops";
      stats[key] = Number(stats[key] || 0) + this.cardOps("asknot");
    }
    this.twilight.game.state.event_name = this.cardName("asknot");
    this.log.push(`${side.toUpperCase()} triggers ${this.cardName("asknot")} as an event`);

    const eligible = this.askNotEligibleCards();
    if (!eligible.length) {
      this.log.push(`US has no cards available to discard for ${this.cardName("asknot")}`);
      this.finalizeEventCard("asknot");
      return;
    }

    this.currentPlayer = "us";
    this.syncSaitoHand();
    this.askNotPending = { discarded: [] };
    this.offerHeadlessAskNotDecision();
  }

  askNotEligibleCards() {
    const state = this.twilight.game.state || {};
    const blocked = new Set(["china", state.headline_card, state.headline_opponent_card].filter(Boolean));
    const alreadyDiscarded = new Set(this.askNotPending?.discarded || []);
    return this.hands.us
      .filter((card) => !blocked.has(card))
      .filter((card) => !alreadyDiscarded.has(card));
  }

  offerHeadlessAskNotDecision() {
    const pending = this.askNotPending;
    if (!pending) return;
    const choices = this.askNotEligibleCards().map((card) => ({
      type: "saito_choice",
      decision: "list",
      value: card,
      label: this.cardName(card),
    }));
    choices.push({ type: "saito_choice", decision: "list", value: "finished", label: "finished" });
    this.adapter.pending = {
      prompt: `${this.cardName("asknot")}: select US cards to discard`,
      choices,
      callback: async (choice) => this.chooseHeadlessAskNot(choice.value),
    };
    this.adapter.lastStatus = this.adapter.pending.prompt;
  }

  chooseHeadlessAskNot(choice) {
    const pending = this.askNotPending;
    if (!pending) return;
    if (choice === "finished") {
      const count = pending.discarded.length;
      if (count > 0) {
        if (this.deck.length < count) this.reshuffleDiscardsIfNeeded();
        this.dealTo("us", this.nonChinaHandCount("us") + count);
        this.log.push(`US draws ${count} replacement card${count === 1 ? "" : "s"} for ${this.cardName("asknot")}`);
      } else {
        this.log.push(`US discards no cards for ${this.cardName("asknot")}`);
      }
      this.askNotPending = null;
      this.finalizeEventCard("asknot");
      return;
    }

    if (!this.askNotEligibleCards().includes(choice)) {
      throw new Error(`invalid ${this.cardName("asknot")} discard: ${choice}`);
    }
    this.removeFromHand("us", choice);
    const meta = this.cardsById[choice] || this.twilight.game.deck[0].cards?.[choice];
    if (meta && !this.twilight.game.deck[0].removed?.[choice]) {
      this.twilight.game.deck[0].discards[choice] = meta;
    }
    pending.discarded.push(choice);
    this.syncTrackedPilesFromSaito();
    this.log.push(`US discards ${this.cardName(choice)} for ${this.cardName("asknot")}`);
    this.offerHeadlessAskNotDecision();
  }

  handleDiscardQueueCommand(parts) {
    const queue = this.twilight.game.queue;
    queue.pop();
    const side = playerToSide(parts[1]);
    const card = parts[2];
    if (!side || !card) return;
    this.removeFromHand(side, card);
    const meta = this.cardsById[card] || this.twilight.game.deck[0].cards?.[card];
    if (meta && !this.twilight.game.deck[0].removed?.[card]) {
      this.twilight.game.deck[0].discards[card] = meta;
    }
    this.syncTrackedPilesFromSaito();
    this.log.push(`${side.toUpperCase()} discards ${this.cardName(card)}`);
  }

  beginSetupFromQueue(move) {
    const parts = String(move).split("\t");
    this.setupQueueCommand = move;
    if (parts[0] === "placement" && parts[1] === "1") {
      this.phase = "setup_ussr";
      this.currentPlayer = "ussr";
      this.setupRemaining = 6;
      this.twilight.game.deck[0].cards["china"] = this.cardsById.china || this.twilight.returnChinaCard();
      if (!this.hands.ussr.includes("china")) this.hands.ussr.push("china");
      this.adapter.reset();
      return;
    }
    if (parts[0] === "placement" && parts[1] === "2") {
      this.phase = "setup_us";
      this.currentPlayer = "us";
      this.setupRemaining = 7;
      this.adapter.reset();
      return;
    }
    if (parts[0] === "placement_bonus") {
      this.phase = "setup_us_bonus";
      this.currentPlayer = "us";
      this.setupRemaining = Number(parts[2] || PRESET.usbonus);
      this.adapter.reset();
      return;
    }
    this.throwBridgeStall(`unknown setup command: ${move}`);
  }

  scoringRegion(cardId) {
    return {
      europe: "europe",
      asia: "asia",
      mideast: "mideast",
      africa: "africa",
      camerica: "camerica",
      samerica: "samerica",
      seasia: "seasia",
      centralamerica: "camerica",
      southamerica: "samerica",
    }[cardId] || cardId;
  }

  defconAllows(region) {
    const defcon = this.twilight.game.state.defcon;
    if (region === "europe") return defcon >= 5;
    if (region === "asia" || region === "seasia") return defcon >= 4;
    if (region === "mideast") return defcon >= 3;
    return true;
  }

  isLegalCoupTarget(countryId) {
    const player = this.currentPlayer;
    const country = this.twilight.countries[countryId];
    const state = this.twilight.game.state;
    if (!country) return false;
    if (Array.isArray(state.limit_region) && state.limit_region.includes(country.region)) return false;
    if (state.limit_ignoredefcon === 0 && !this.defconAllows(country.region)) return false;
    if (state.events?.usjapan === 1 && countryId === "japan" && player === "ussr") return false;
    if (country.region === "europe" && state.events?.reformer === 1 && player === "ussr") return false;
    if (country.region === "europe" && state.events?.nato === 1 && player === "ussr") {
      const exempt = (countryId === "westgermany" && state.events?.nato_westgermany === 0)
        || (countryId === "france" && state.events?.nato_france === 0);
      if (!exempt && this.twilight.isControlled?.("us", countryId) === 1) return false;
    }
    if (player === "us" && Number(country.ussr) <= 0) return false;
    if (player === "ussr" && Number(country.us) <= 0) return false;
    return true;
  }

  isLegalRealignTarget(countryId) {
    const player = this.currentPlayer;
    const country = this.twilight.countries[countryId];
    const state = this.twilight.game.state;
    if (!country) return false;
    if (Array.isArray(state.limit_region) && state.limit_region.includes(country.region)) return false;
    if (state.limit_ignoredefcon === 0 && state.events?.inftreaty === 0 && !this.defconAllows(country.region)) return false;
    if (state.events?.usjapan === 1 && countryId === "japan" && player === "ussr") return false;
    if (country.region === "europe" && state.events?.nato === 1 && player === "ussr") {
      const exempt = (countryId === "westgermany" && state.events?.nato_westgermany === 0)
        || (countryId === "france" && state.events?.nato_france === 0);
      if (!exempt && this.twilight.isControlled?.("us", countryId) === 1) return false;
    }
    if (player === "us" && Number(country.ussr) <= 0) return false;
    if (player === "ussr" && Number(country.us) <= 0) return false;
    return true;
  }

  isTargetlessJuntaFreeOpsContext(originalChoices = [], hasLegalCoupTarget = false, hasLegalRealignTarget = false) {
    const parts = String(this.queueTop() || "").split("\t");
    if (parts[0] !== "ops" || parts[2] !== "junta" || String(parts[3]) !== "2") return false;
    if (parts[1] !== this.currentPlayer) return false;
    const state = this.twilight.game.state || {};
    if (state.events?.junta !== 1) return false;
    if (Number(state.limit_placement || 0) !== 1) return false;
    const limitedRegions = new Set(Array.isArray(state.limit_region) ? state.limit_region : []);
    for (const region of ["europe", "africa", "mideast", "asia", "seasia"]) {
      if (!limitedRegions.has(region)) return false;
    }
    if (originalChoices.length) {
      const values = new Set(originalChoices.map((choice) => choice.value));
      if (!values.has("coup") && !values.has("realign")) return false;
    }
    return !hasLegalCoupTarget && !hasLegalRealignTarget;
  }

  isTargetlessOpsContext(originalChoices = [], hasLegalCoupTarget = false, hasLegalRealignTarget = false) {
    const parts = String(this.queueTop() || "").split("\t");
    if (parts[0] !== "ops" || parts[1] !== this.currentPlayer) return false;
    const values = new Set(originalChoices.map((choice) => choice.value));
    if (!values.size) return !hasLegalCoupTarget && !hasLegalRealignTarget;
    for (const value of values) {
      if (value !== "coup" && value !== "realign") return false;
    }
    if (values.has("coup") && hasLegalCoupTarget) return false;
    if (values.has("realign") && hasLegalRealignTarget) return false;
    return true;
  }

  skipTargetlessJuntaFreeOps() {
    if (!this.isTargetlessJuntaFreeOpsContext([], false, false)) {
      this.throwBridgeStall("attempted targetless Junta free-ops skip outside Junta context");
    }
    const player = this.currentPlayer;
    this.log.push(`${player.toUpperCase()} has no legal Junta coup/realign target; free OPS skipped`);
    this.twilight.addMove("resolve\tops");
    this.twilight.addMove("unlimit\tplacement");
    this.twilight.addMove("unlimit\tmilops");
    this.twilight.addMove("unlimit\tregion");
    this.twilight.addMove("resolve\tjunta");
    this.twilight.endTurn();
    this.adapter.reset();
  }

  skipTargetlessOps() {
    if (!this.isTargetlessOpsContext([], false, false)) {
      this.throwBridgeStall("attempted targetless OPS skip outside targetless OPS context");
    }
    const player = this.currentPlayer;
    const parts = String(this.queueTop() || "").split("\t");
    const card = parts[2] || "unknown card";
    this.log.push(`${player.toUpperCase()} has no legal OPS target mode for ${this.cardName(card)}; OPS skipped`);
    this.twilight.addMove("resolve\tops");
    this.twilight.endTurn();
    this.adapter.reset();
  }

  step(action) {
    this.legalActions();
    return this.stepAsync(action);
  }

  async stepAsync(action) {
    this.legalActions();
    if (this.phase.startsWith("setup_")) {
      this.stepSetupPlacement(action);
      this.checkTerminal();
      return this.result();
    }
    this.syncCountryPlaceFlags();
    const beforeProgress = this.progressSignature();
    await this.adapter.choose(action);
    const promptAfterChoice = this.adapter.normalizeStatus(this.adapter.lastStatus || "");
    const forceDrainDomAction = promptAfterChoice.includes("extra 1 op available for");
    if (action?.type === "saito_dom" && !forceDrainDomAction) {
      this.adapter.refreshAnyDomPending();
      if (this.adapter.pending) {
        this.checkTerminal();
        return this.result();
      }
    }
    await this.drainUntilDecisionOrTurnEnd();
    if (action?.type === "saito_dom" && this.progressSignature() === beforeProgress) {
      this.adapter.off(action.selector, action.event);
      this.adapter.remove(`#${action.value}`);
      this.adapter.pending = null;
    }
    this.checkTerminal();
    return this.result();
  }

  stepSetupPlacement(action) {
    const country = String(action?.value || "");
    if (!this.legalSetupCountries().includes(country)) {
      throw new Error(`invalid setup placement country: ${country}`);
    }
    if (this.phase === "setup_ussr") {
      this.twilight.placeInfluence(country, 1, "ussr");
      this.log.push(`USSR initial placement: ${this.twilight.countries[country].name}`);
      this.setupRemaining -= 1;
      if (this.setupRemaining <= 0) {
        this.completeSetupQueueCommand();
      }
    } else if (this.phase === "setup_us") {
      this.twilight.placeInfluence(country, 1, "us");
      this.log.push(`US initial placement: ${this.twilight.countries[country].name}`);
      this.setupRemaining -= 1;
      if (this.setupRemaining <= 0) {
        this.completeSetupQueueCommand();
      }
    } else if (this.phase === "setup_us_bonus") {
      this.twilight.placeInfluence(country, 1, "us");
      this.log.push(`US optional +${PRESET.usbonus} placement: ${this.twilight.countries[country].name}`);
      this.setupRemaining -= 1;
      if (this.setupRemaining <= 0) {
        this.completeSetupQueueCommand();
      }
    }
    this.syncSaitoHand();
  }

  completeSetupQueueCommand() {
    const top = this.queueTop();
    if (top !== this.setupQueueCommand) {
      this.throwBridgeStall(`setup command mismatch: expected ${this.setupQueueCommand}, got ${top}`);
    }
    this.twilight.game.queue.pop();
    this.phase = "boot";
    this.setupRemaining = 0;
    this.adapter.reset();
    this.processQueueUntilDecisionSync();
  }

  settleHeadlineResolutionSync() {
    for (let guard = 0; guard < 100; guard++) {
      if (this.adapter.pending || this.winner || this.phase !== "headline_resolve") return;
      if (this.twilight.game.queue.length === 0) {
        this.queueNextHeadlineEvent();
        continue;
      }
      if (this.handleBridgeQueueCommandSync()) continue;
      const actor = this.actorForQueueTop();
      if (actor) this.currentPlayer = actor;
      this.syncSaitoHand();
      const before = this.progressSignature();
      this.ensureBoardPositionState();
      this.withSilencedConsole(() => this.twilight.handleGameLoop());
      this.syncTrackedPilesFromSaito();
      if (this.adapter.pending || this.winner) return;
      if (this.progressSignature() === before) {
        if (this.handleBridgeQueueCommandSync()) continue;
        this.adapter.refreshAnyDomPending();
        if (this.adapter.pending) return;
        if (this.retryEventWithAlternateActors()) continue;
        this.throwBridgeStall("headline queue did not advance");
      }
    }
    this.throwBridgeStall("headline resolution exceeded guard limit");
  }

  progressSignature() {
    const state = this.twilight.game.state || {};
    const countries = this.twilight.countries || {};
    let influence = "";
    for (const id of Object.keys(countries).sort()) {
      influence += `${id}:${countries[id].us || 0}:${countries[id].ussr || 0};`;
    }
    return JSON.stringify({
      phase: this.phase,
      currentPlayer: this.currentPlayer,
      turn: state.round || 1,
      actionRound: this.currentActionRound(),
      vp: state.vp,
      defcon: state.defcon,
      prompt: this.adapter.lastStatus || "",
      queue: this.twilight.game.queue.length,
      moves: this.twilight.moves.length,
      handUs: this.hands.us.length,
      handUssr: this.hands.ussr.length,
      influence,
    });
  }

  syncCountryPlaceFlags() {
    const liveCountries = this.twilight?.countries || {};
    const savedCountries = this.twilight?.game?.countries || {};
    for (const [id, saved] of Object.entries(savedCountries)) {
      if (liveCountries[id] && Object.prototype.hasOwnProperty.call(saved, "place")) {
        if (liveCountries[id].place === 1 || saved.place === 1) {
          liveCountries[id].place = 1;
          saved.place = 1;
        }
      }
    }
  }

  ensurePendingDecision() {
    if (this.adapter.pending || this.winner) return;
    if (this.phase === "headline_ussr") {
      this.createHeadlineDecision("ussr");
      return;
    }
    if (this.phase === "headline_us") {
      this.createHeadlineDecision("us");
      return;
    }
    if (this.phase !== "action") return;
    if (this.twilight.game.queue.length > 0) {
      this.processQueueUntilDecisionSync();
      return;
    }
    if (this.createActionCardDecisionFallback()) return;
    this.throwBridgeStall("action phase has no Saito queue command");
  }

  createActionCardDecisionFallback() {
    if (this.adapter.pending || this.winner || this.phase !== "action") return false;
    const side = this.currentPlayer;
    if (side !== "us" && side !== "ussr") return false;
    this.syncSaitoHand();
    const cards = this.playableFallbackCards(side);
    if (!cards.length) return false;
    const forcedScoring = this.forcedScoringCards(cards);
    const choices = forcedScoring.length ? forcedScoring : cards;
    const reason = forcedScoring.length ? "mandatory scoring card fallback" : "card selection fallback";
    this.log.push(`${side.toUpperCase()} ${reason}: ${choices.map((card) => this.cardName(card)).join(", ")}`);
    this.adapter.setList(`${side.toUpperCase()} pick a card`, choices, async (card) => {
      this.currentPlayer = side;
      this.syncSaitoHand();
      await this.twilight.playerTurnCardSelected(card, side);
    });
    return true;
  }

  playableFallbackCards(side) {
    const hand = [...(this.hands[side] || [])];
    if (!hand.length) return ["skipturn"];
    return hand.filter((card) => card !== "unintervention");
  }

  forcedScoringCards(cards) {
    const scoring = cards.filter((card) => this.cardsById?.[card]?.scoring);
    if (!scoring.length) return [];
    const roundsInTurn = Number(this.twilight.game.state?.round || 1) > 3 ? 7 : 6;
    const movesRemaining = roundsInTurn - Number(this.twilight.game.state?.turn_in_round || 0);
    return scoring.length >= movesRemaining ? scoring : [];
  }

  createHeadlineDecision(side) {
    this.currentPlayer = side;
    this.syncSaitoHand();
    const cards = this.hands[side].filter((card) => card !== "china" && card !== "unintervention");
    if (cards.length === 0) {
      this.log.push(`${side.toUpperCase()} skips headline with no valid cards`);
      this.headlineCards[side] = null;
      if (side === "ussr") {
        this.phase = "headline_us";
        this.currentPlayer = "us";
      } else {
        this.beginHeadlineResolution();
      }
      return;
    }
    this.adapter.setList(`${side.toUpperCase()} pick headline card`, cards, (card) => this.selectHeadlineCard(side, card));
  }

  selectHeadlineCard(side, card) {
    if (!this.hands[side].includes(card)) throw new Error(`${side} cannot headline card not in hand: ${card}`);
    if (card === "china" || card === "unintervention") throw new Error(`invalid headline card: ${card}`);
    this.headlineCards[side] = card;
    this.log.push(`${side.toUpperCase()} selects headline ${this.cardName(card)}`);
    this.adapter.reset();
    if (side === "ussr") {
      this.phase = "headline_us";
      this.currentPlayer = "us";
      return;
    }
    this.beginHeadlineResolution();
  }

  beginHeadlineResolution() {
    const uscard = this.headlineCards.us;
    const ussrcard = this.headlineCards.ussr;
    this.phase = "headline_resolve";
    this.twilight.game.state.headline = 1;
    this.twilight.game.state.headline_card = "";
    this.twilight.game.state.headline_opponent_card = "";
    if (uscard) this.log.push(`US headlines ${this.cardName(uscard)}`);
    if (ussrcard) this.log.push(`USSR headlines ${this.cardName(ussrcard)}`);
    if (!uscard && !ussrcard) {
      this.endHeadlinePhase();
      return;
    }

    if (uscard && ussrcard && (uscard === "defectors" || (ussrcard !== "defectors" && this.twilight.game.state.defectors_pulled_in_headline === 1))) {
      this.log.push(`US headline ${this.cardName("defectors")} cancels USSR headline ${this.cardName(ussrcard)}`);
      this.removeFromHand("us", uscard);
      this.removeFromHand("ussr", ussrcard);
      this.finalizeHeadlineCard("us", uscard);
      this.finalizeHeadlineCard("ussr", ussrcard);
      this.endHeadlinePhase();
      return;
    }

    const usOps = this.cardOps(uscard);
    const ussrOps = this.cardOps(ussrcard);
    if (uscard && ussrcard) this.headlineOrder = ussrOps > usOps ? ["ussr", "us"] : ["us", "ussr"];
    else this.headlineOrder = uscard ? ["us"] : ["ussr"];
    this.queueNextHeadlineEvent();
  }

  queueNextHeadlineEvent() {
    if (this.resolvingHeadline) {
      this.finalizeHeadlineCard(this.resolvingHeadline.side, this.resolvingHeadline.card);
      this.resolvingHeadline = null;
    }
    if (this.winner) return false;
    if (!this.headlineOrder.length) {
      this.endHeadlinePhase();
      return true;
    }

    const side = this.headlineOrder.shift();
    const card = this.headlineCards[side];
    if (!card) return this.queueNextHeadlineEvent();
    if (side === "ussr" && card !== "defectors" && this.twilight.game.state.defectors_pulled_in_headline === 1) {
      this.log.push(`USSR headline ${this.cardName(card)} is cancelled by ${this.cardName("defectors")}`);
      this.removeFromHand(side, card);
      this.finalizeHeadlineCard(side, card);
      return this.queueNextHeadlineEvent();
    }

    this.currentPlayer = side;
    this.syncSaitoHand();
    this.removeFromHand(side, card);
    this.twilight.game.state.headline_card = card;
    this.twilight.game.state.headline_opponent_card = this.headlineCards[other(side)] || "";
    this.twilight.game.state.player_to_go = sideToPlayer(side);
    this.resolvingHeadline = { side, card };
    this.log.push(`Resolving ${side.toUpperCase()} headline: ${this.cardName(card)}`);
    this.twilight.game.queue.push(`event\t${side}\t${card}`);
    return true;
  }

  endHeadlinePhase() {
    this.phase = "action";
    this.headlineOrder = [];
    this.resolvingHeadline = null;
    this.twilight.game.state.headline = 0;
    this.twilight.game.state.headline_card = "";
    this.twilight.game.state.headline_xor = "";
    this.twilight.game.state.headline_hash = "";
    this.twilight.game.state.headline_opponent_hash = "";
    this.twilight.game.state.headline_opponent_xor = "";
    this.twilight.game.state.headline_opponent_card = "";
    this.currentPlayer = "ussr";
    this.adapter.reset();
  }

  removeFromHand(side, card) {
    const idx = this.hands[side].indexOf(card);
    if (idx >= 0) this.hands[side].splice(idx, 1);
  }

  finalizeHeadlineCard(side, card) {
    if (!card) return;
    if (this.twilight.game.deck[0].removed?.[card] || this.twilight.game.deck[0].discards?.[card]) return;
    const meta = this.cardsById[card];
    if (!meta) return;
    if (meta.recurring == 1) {
      this.twilight.game.deck[0].discards[card] = meta;
      this.log.push(`${this.cardName(card)} discarded`);
    } else {
      this.twilight.game.deck[0].removed[card] = meta;
      this.log.push(`${this.cardName(card)} removed from game`);
    }
    this.discard = Object.keys(this.twilight.game.deck[0].discards || {});
    this.removed = Object.keys(this.twilight.game.deck[0].removed || {});
  }

  finalizeEventCard(card) {
    if (!card) return;
    if (this.twilight.game.deck[0].removed?.[card] || this.twilight.game.deck[0].discards?.[card]) return;
    const meta = this.cardsById[card] || this.twilight.game.deck[0].cards?.[card];
    if (!meta) return;
    if (meta.recurring == 1) {
      this.twilight.game.deck[0].discards[card] = meta;
      this.log.push(`${this.cardName(card)} discarded`);
    } else {
      this.twilight.game.deck[0].removed[card] = meta;
      delete this.twilight.game.deck[0].cards[card];
      this.log.push(`${this.cardName(card)} removed from game`);
    }
    this.syncTrackedPilesFromSaito();
  }

  cardName(card) {
    return this.cardsById?.[card]?.name || card;
  }

  cardOps(card) {
    return Number(this.cardsById?.[card]?.ops || 0);
  }

  async drainUntilDecisionOrTurnEnd() {
    for (let guard = 0; guard < 1000; guard++) {
      await Promise.resolve();
      if (this.adapter.pending || this.winner) return;
      if (this.phase === "headline_ussr" || this.phase === "headline_us") {
        this.ensurePendingDecision();
        return;
      }
      if (this.phase === "headline_resolve") {
        const previousPhase = this.phase;
        if (this.queueNextHeadlineEvent()) {
          if (previousPhase === "headline_resolve" && this.phase === "action") return;
          continue;
        }
      }
      if (this.twilight.game.queue.length > 0) {
        if (this.handleBridgeQueueCommandSync()) continue;
        const actor = this.actorForQueueTop();
        if (actor) this.currentPlayer = actor;
        this.syncSaitoHand();
        const before = this.progressSignature();
        this.ensureBoardPositionState();
        this.withSilencedConsole(() => this.twilight.handleGameLoop());
        this.syncTrackedPilesFromSaito();
        await Promise.resolve();
        if (this.adapter.pending || this.winner) return;
        if (this.progressSignature() === before) {
          if (this.handleBridgeQueueCommandSync()) continue;
          this.adapter.refreshAnyDomPending();
          if (this.adapter.pending) return;
          if (this.retryEventWithAlternateActors()) continue;
          this.throwBridgeStall("Saito queue did not advance");
        }
        continue;
      }
      if (this.isSubmittingMoves()) {
        this.adapter.reset();
        continue;
      }
      this.adapter.refreshAnyDomPending();
      if (this.adapter.pending) return;
      this.throwBridgeStall("no queue command or pending decision");
    }
    this.throwBridgeStall("async drain exceeded guard limit");
  }

  isSubmittingMoves() {
    return this.phase === "action" && String(this.adapter.lastStatus || "").toLowerCase().includes("submitting moves");
  }

  actorForQueueTop() {
    const queue = this.twilight.game.queue;
    const parts = String(queue[queue.length - 1] || "").split("\t");
    const command = parts[0];
    if (command === "event") {
      if (parts[2] === "starwars") return "us";
      if (parts[2] === "terrorism") return other(parts[1]);
      return parts[1] === "us" || parts[1] === "ussr" ? parts[1] : null;
    }
    if (command === "teardownthiswall") return playerToSide(parts[1]);
    if (command === "play") return playerToSide(parts[1]);
    if (command === "turn") return playerToSide(this.twilight.game.state?.turn || parts[1]);
    const candidate = {
      ops: parts[1],
      card: parts[1],
      discard: parts[1],
      space: parts[1],
    }[command];
    return candidate === "us" || candidate === "ussr" ? candidate : null;
  }

  hasQueuedCommandBelowTop(command) {
    const queue = this.twilight.game.queue || [];
    for (let i = queue.length - 2; i >= 0; i--) {
      if (String(queue[i] || "").split("\t")[0] === command) return true;
    }
    return false;
  }

  retryEventWithAlternateActors() {
    const queue = this.twilight.game.queue;
    const parts = String(queue[queue.length - 1] || "").split("\t");
    if (parts[0] !== "event") return false;
    const cardSide = this.cardsById?.[parts[2]]?.player;
    const eventSide = parts[1];
    const candidates = [];
    for (const side of [cardSide, other(cardSide), eventSide, other(eventSide)]) {
      if ((side === "us" || side === "ussr") && side !== this.currentPlayer && !candidates.includes(side)) {
        candidates.push(side);
      }
    }
    for (const side of candidates) {
      this.currentPlayer = side;
      this.syncSaitoHand();
      const before = queue.length;
      this.ensureBoardPositionState();
      this.withSilencedConsole(() => this.twilight.handleGameLoop());
      if (this.adapter.pending || this.winner) return true;
      if (queue.length !== before) return true;
    }
    return false;
  }

  checkTerminal() {
    const vp = this.twilight.game.state.vp;
    if (this.twilight.game.state.defcon < 2) {
      this.winner = other(this.currentPlayer);
      this.terminalReason = "nuclear_war";
    } else if (vp >= 20) {
      this.winner = "us";
      this.terminalReason = "vp_threshold";
    } else if (vp <= -20) {
      this.winner = "ussr";
      this.terminalReason = "vp_threshold";
    }
  }

  reward() {
    if (!this.winner) return { us: 0, ussr: 0 };
    return { us: this.winner === "us" ? 1 : -1, ussr: this.winner === "ussr" ? 1 : -1 };
  }

  chinaOwner() {
    if (this.hands.us.includes("china")) return "us";
    if (this.hands.ussr.includes("china")) return "ussr";
    const state = this.twilight.game.state || {};
    if (state.events?.china_card === "us" || state.china_card === "us") return "us";
    if (state.events?.china_card === "ussr" || state.china_card === "ussr") return "ussr";
    return "none";
  }

  observe(side) {
    const legal_actions = side === this.currentPlayer ? this.legalActions() : [];
    return {
      preset: PRESET,
      side,
      phase: this.phase,
      current_player: this.currentPlayer,
      turn: this.twilight.game.state.round || 1,
      action_round: this.currentActionRound(),
      vp: this.twilight.game.state.vp,
      defcon: this.twilight.game.state.defcon,
      headline: { ...this.headlineCards },
      milops: { us: this.twilight.game.state.milops_us, ussr: this.twilight.game.state.milops_ussr },
      space: { us: this.twilight.game.state.space_race_us, ussr: this.twilight.game.state.space_race_ussr },
      hand: [...this.hands[side]],
      hand_count: { us: this.hands.us.length, ussr: this.hands.ussr.length },
      deck_count: this.deck.length,
      discard: [...this.discard],
      removed: [...this.removed],
      events: this.twilight.game.state.events || {},
      event_name: this.twilight.game.state.event_name || "",
      queue_top: this.queueTop(),
      china_owner: this.chinaOwner(),
      prompt: this.adapter.lastStatus || "",
      countries: this.countriesArray(),
      legal_actions,
      terminal: Boolean(this.winner),
      winner: this.winner,
      log_tail: this.log.slice(-8),
    };
  }

  result() {
    return {
      observation: this.observe(this.currentPlayer),
      reward: this.reward(),
      done: Boolean(this.winner),
      info: {
        current_player: this.currentPlayer,
        winner: this.winner,
        terminal_reason: this.terminalReason,
        vp: this.twilight.game.state.vp,
        defcon: this.twilight.game.state.defcon,
      },
    };
  }

  gameLog() {
    return {
      preset: PRESET,
      turn: this.twilight.game.state.round || 1,
      action_round: this.currentActionRound(),
      current_player: this.currentPlayer,
      phase: this.phase,
      headline: { ...this.headlineCards },
      winner: this.winner,
      terminal_reason: this.terminalReason,
      vp: this.twilight.game.state.vp,
      defcon: this.twilight.game.state.defcon,
      hands: {
        us: [...this.hands.us],
        ussr: [...this.hands.ussr],
      },
      deck: [...this.deck],
      discard: [...this.discard],
      removed: [...this.removed],
      countries: this.countriesArray(),
      log: [...this.log],
    };
  }

  renderText(side = this.currentPlayer) {
    const obs = this.observe(side);
    const countries = obs.countries.slice(0, 28).map((c) => (
      `${c.id.padEnd(18)} ${c.region.padEnd(8)} us=${String(c.us).padStart(2)} ussr=${String(c.ussr).padStart(2)} ctl=${c.control}`
    )).join("\n");
    return [
      `Preset: ${PRESET.id} backend=saito`,
      `Turn ${obs.turn}.${obs.action_round} VP=${obs.vp} DEFCON=${obs.defcon} current=${obs.current_player}`,
      `${side.toUpperCase()} hand: ${obs.hand.join(", ")}`,
      countries,
    ].join("\n");
  }
}

async function handle(env, msg) {
  if (msg.cmd === "reset") return env.reset(msg.seed ?? 1);
  if (msg.cmd === "legal_actions") return env.legalActions();
  if (msg.cmd === "step") return env.stepAsync(msg.action);
  if (msg.cmd === "observe") return env.observe(msg.side || env.currentPlayer);
  if (msg.cmd === "render_text") return env.renderText(msg.side);
  if (msg.cmd === "cards") return env.cardsArray();
  if (msg.cmd === "countries") return env.countriesArray();
  if (msg.cmd === "preset") return PRESET;
  if (msg.cmd === "log") return env.gameLog();
  if (msg.cmd === "state") return env.observe(env.currentPlayer);
  throw new Error(`unknown command: ${msg.cmd}`);
}

function runServer() {
  const env = new SaitoTwilightKernel();
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  rl.on("line", async (line) => {
    let msg = {};
    try {
      msg = JSON.parse(line);
      const result = await handle(env, msg);
      process.stdout.write(`${JSON.stringify({ id: msg.id, ok: true, result })}\n`);
    } catch (err) {
      process.stdout.write(`${JSON.stringify({ id: msg.id, ok: false, error: err.stack || err.message })}\n`);
    }
  });
}

function selfTest() {
  const env = new SaitoTwilightKernel();
  env.reset(7);
  const run = async () => {
  for (let i = 0; i < 40 && !env.winner; i++) {
    const legal = env.legalActions();
    if (!legal.length) {
      const top = env.twilight.game.queue[env.twilight.game.queue.length - 1] || "";
      throw new Error(`no legal actions at step ${i}: player=${env.currentPlayer} status=${env.adapter.lastStatus} queue=${top}`);
    }
    await env.stepAsync(legal[Math.floor(env.rand() * legal.length)]);
  }
  console.log(JSON.stringify({ ok: true, backend: PRESET.backend, preset: PRESET, turn: env.twilight.game.state.round, vp: env.twilight.game.state.vp }));
  };
  return run();
}

if (require.main === module) {
  if (process.argv.includes("--self-test")) selfTest().catch((err) => { console.error(err); process.exit(1); });
  else runServer();
}

module.exports = { SaitoTwilightKernel, PRESET };
