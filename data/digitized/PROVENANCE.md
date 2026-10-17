# Digitized data: provenance

All files in this directory were recovered from the two PDFs supplied by the
user (Mellado Muñoz et al., PRL 125, 020403 (2020), and its Supplemental
Material) by `scripts/00_digitize.py` (vector figures) and
`scripts/00b_digitize_fig1b.py` (raster Fig. 1). They are **digitized values
of the plotted means and error bars**, not the raw measurements, and will be
replaced by raw files in `data/raw/` when available.

## Method

* Vector figures (Figs. 2, 3, 4, S1, S2, S5, S7): the PDF content streams are
  interpreted (`scripts/pdfvec.py`), giving every painted path in page
  coordinates. Axis calibration uses the tick marks and the tick labels:
  labels drawn as glyph outlines (main paper, Fig. S2) are recognised by
  shape clustering (`scripts/glyphs.py`, cluster-to-character table built by
  eye from a contact sheet); labels drawn as text (Figs. S1, S5, S7) are read
  from the text operators. A linear fit of label value against tick position
  is required to reproduce the labels to better than 2% of the axis span
  (actual residuals are at machine precision). Data points are the centres of
  the marker paths; error bars are the two-point segments of the same colour
  through the marker. Curves are the polylines as drawn.
* Fig. 3: the tick labels are text whose individual positions are not
  recoverable with this parser, so the calibration uses the 11 x tick marks
  (0.5 to 1.5 step 0.1) and 9 y tick marks (0.0 to 0.8 step 0.1) read from
  the rendered figure.
* Fig. 1(b) is a raster image (1000 x 544 px). The frame and the outward tick
  marks are found from the coloured pixels; marker centres are centroids of
  saturated red / blue pixel clusters. Axis ranges 0 to 2.5 s, 0 to 1.0
  (1e16 m^-3) and 0 to 0.5 Hz are read from the rendered tick labels.

## Precision

Marker-centre recovery is exact to plotting precision (~0.1 pt out of an axis
span of 85 to 200 pt, i.e. <0.2% of the span) for vector figures, and about
1 px out of ~330 px for the raster Fig. 1(b) (~0.3% of span).

## Files

| file | figure | columns |
|---|---|---|
| fig1b_gammaK.csv | Fig. 1(b) | series, t (s), value (n_K in 1e16 m^-3 or gamma_K in Hz), blob sizes; nK_errorbar rows give (t, lower, upper) |
| fig2_nodiss.csv, fig2_diss.csv | Fig. 2(a-d) | t, N, N_err, T (nK), T_err, F, F_err, mu (nK), mu_err |
| fig2c_Feq_*.csv, fig2c_Feq_band_*.csv | Fig. 2(c) | equilibrium curve of Eq. (1) and its error band |
| fig2d_muBEC_*.csv | Fig. 2(d) | BEC chemical potential lines |
| fig3_open_nodiss.csv | Fig. 3 open symbols | tau, F, tau_err, F_err, marker point count (shape) |
| fig3_filled_diss.csv | Fig. 3 filled symbols | tau, F, errors, marker shape, RGB colour (time-coded) |
| fig3_eq1_curve.csv, fig3_model_dotted_*.csv | Fig. 3 | Eq. (1) curve and the rate-model dotted trajectories |
| fig4_deltaF.csv, fig4_fit_curve.csv | Fig. 4 | t, delta F, error; fitted exponential |
| figS1_omega_*.csv | Fig. S1 | t, mean trap frequency (Hz), calculated and measured |
| figS2i_*.csv, figS2ii_*.csv (+ *_curve_*) | Fig. S2(i),(ii) | as Fig. 2; (ii): t, N, tau, F, T |
| figS5_*.csv | Fig. S5 | t (ms), N; fits |
| figS7_*.csv | Fig. S7 | t, N_th (thousands), N_0 (thousands), F; model lines |
