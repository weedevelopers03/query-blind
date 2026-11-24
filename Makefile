# Reproducible figure & test targets (Methodology Guide §11.3)
PYTHON ?= python3
RESULTS_DIR ?= ../zoo12_summaries
FIG_OUT ?= ../thesis_draft/figures

.PHONY: figures test

figures:
	MPLCONFIGDIR=/tmp/mpl $(PYTHON) figures/fig1_m4_cka_collapse.py --results_dir $(RESULTS_DIR) --out_dir $(FIG_OUT)
	MPLCONFIGDIR=/tmp/mpl $(PYTHON) figures/fig2_qii_omega2_12models.py --results_dir $(RESULTS_DIR) --out_dir $(FIG_OUT)
	MPLCONFIGDIR=/tmp/mpl $(PYTHON) figures/fig3_waterbirds_m3_l8_l11.py --results_dir $(RESULTS_DIR) --out_dir $(FIG_OUT)
	MPLCONFIGDIR=/tmp/mpl $(PYTHON) figures/fig4_i1_i2_summary.py --results_dir $(RESULTS_DIR) --out_dir $(FIG_OUT)
	MPLCONFIGDIR=/tmp/mpl $(PYTHON) figures/fig5_ralr_vs_cls.py --results_dir $(RESULTS_DIR) --out_dir $(FIG_OUT)
	MPLCONFIGDIR=/tmp/mpl $(PYTHON) figures/fig6_h4_registers.py --results_dir $(RESULTS_DIR) --out_dir $(FIG_OUT)
	MPLCONFIGDIR=/tmp/mpl $(PYTHON) figures/fig7_deit3_i1_s4_mechanism.py --results_dir $(RESULTS_DIR) --out_dir $(FIG_OUT)

test:
	$(PYTHON) -m pytest tests/test_registry.py tests/test_stats.py tests/test_preprocess.py tests/test_pe_interpolation.py tests/test_deit3_mass_shift.py tests/test_ralr_openimages.py -v
