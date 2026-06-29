# Ruleset: `optional_us_plus_2`

This is the v1 training/play preset.

## Saito Options

```js
game.options.deck = "optional";
game.options.usbonus = 2;
```

## Included Card Scope

Use the base game plus the official optional/expanded cards that Saito includes when `deck !== "original"`:

- Early War optional: `defectors`, `cambridge`, `specialrelation`, `norad`
- Mid War optional: `che`, `tehran`
- Late War optional: `iraniraq`, `yuri`, `awacs`

`muslimrevolution` is included as a normal Mid War card.

## Excluded Card Scope

Exclude Saito/community/End of History/Cold War Crazies add-on cards such as `berlinagreement`, `pinochet`, `revolutionsof1989`, `gouzenkoaffair`, `perestroika`, and similar non-vanilla extras.

## Starting Setup

The US receives two additional setup influence through the Saito `usbonus` option. This is influence, not operations points.

