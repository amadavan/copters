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

= Nonlinear Programming algorithms

= Stochastic Programming algorithms

// Bibliography
#bibliography(
  "refs.bib",
  title: "References",
  style: "ieee",
)

