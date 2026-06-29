"use strict";

class SaitoOverlay {
  constructor() {
    this.is_visible = false;
    this.clickToClose = false;
  }
  show() { this.is_visible = true; }
  hide() { this.is_visible = false; }
  render() { this.is_visible = true; }
}

module.exports = SaitoOverlay;

