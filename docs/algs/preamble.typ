#import "@preview/ouset:0.2.0": underset

#let underset = underset

#let scr(it) = math.class("normal", box({
  show math.equation: set text(stylistic-set: 1)
  $cal(it)$
}))
