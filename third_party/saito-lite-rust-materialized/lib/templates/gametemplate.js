"use strict";

class HeadlessGameTemplate {
  constructor(app = {}) {
    this.app = app;
    this.browser_active = 0;
    this.game = {
      id: "headless",
      player: 1,
      players: ["ussr", "us"],
      options: {},
      queue: [],
      deck: [],
      state: null,
      countries: null,
      over: 0,
      winner: 0,
      confirms_needed: [0, 0],
      saito_cards_added: [],
      saito_cards_added_reason: [],
      saito_cards_removed: [],
      saito_cards_removed_reason: [],
    };
    this.moves = [];
    this.publicKey = "headless";
    this.clock = {};
    this.hud = {
      mode: 0,
      card_width: 120,
      attachControlCallback: () => {},
    };
    this.overlay = { show: () => {}, hide: () => {}, clickToClose: false };
    this.playerbox = { setActive: () => {} };
  }

  addMove(move) { this.moves.push(move); }
  endTurn() { this.game.queue.push(...this.moves.reverse()); this.moves = []; }
  rollDice(sides = 6) { return Math.floor(Math.random() * sides) + 1; }
  updateLog() {}
  updateStatus() {}
  updateStatusWithOptions() {}
  updateStatusAndListCards() {}
  displayModal() {}
  displayBoard() {}
  displayChinaCard() {}
  updateVictoryPoints() {}
  updateActionRound() {}
  updateEventTiles() {}
  injectGameHTML() {}
  render() {}
  saveGamePreference() {}
  preloadImageArray() {}
  sendGameOverTransaction(winner) {
    this.game.over = 1;
    this.game.winner = winner;
  }
}

module.exports = HeadlessGameTemplate;

