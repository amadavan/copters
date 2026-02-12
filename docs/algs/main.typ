/*
 * The Rubber Article Template.
 *
 * Here is a quick run-down of the template.
 * Some example content has been added for you to see what the template looks like and how it works.
 * Some features of this template are explained here, so you might want to check it out.
 */

#import "@preview/rubber-article:0.5.0": *

// Layout and styling
#show: article.with(
  cols: none, // Tip: use #colbreak() instead of #pagebreak() to seamlessly toggle columns
  eq-chapterwise: true,
  eq-numbering: "(1.1)",
  header-display: true,
  header-title: "copters",
  lang: "en",
  page-margins: 1.75in,
  page-paper: "us-letter",
)

// Reset equation counter at each section/subsection
#show heading.where(level: 1): it => {
  counter(math.equation).update(0)
  it
}
#show heading.where(level: 2): it => {
  counter(math.equation).update(0)
  it
}

// Equation numbering: section.subsection.equation
#set math.equation(numbering: n => {
  let h = counter(heading).get()
  if h.len() >= 2 {
    [(#(h.at(0)).#(h.at(1)).#n)]
  } else if h.len() >= 1 {
    [(#(h.at(0)).#n)]
  } else {
    [(#n)]
  }
})

// Frontmatter
#maketitle(
  title: "copters: Algorithms Reference",
  authors: ("Avinash Madavan",),
  date: datetime.today().display("[day]. [month repr:long] [year]"),
)

This document is a description of the algorithms applied in copters. It is not intended to be a comprehensive treatment of optimization theory, but rather a dive into the implementation and key concepts used in the software. We will attempt to provide references to more complete treatments of the topics discussed here, but do not guarantee their completeness.

#outline(
  title: "Contents",
  indent: auto,
  depth: 2,
)


// Introduction
#include "intro.typ"

= Linear Programming algorithms

#include "mpc.typ"

= Nonlinear Programming algorithms

= Stochastic Programming algorithms

// Bibliography
#bibliography(
  "refs.bib",
  title: "References",
  style: "ieee",
)

