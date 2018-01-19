#!/bin/sh
xelatex report.tex
biber report
xelatex report.tex
