all : synthetic_simulations lpd_simulations real_data sensitivity


synthetic_simulations : run_synthetic_simulations
	python process_synthetic_simulation.py
	python make_test_curves.py
	python make_test_curves_ratios.py


lpd_simulations : run_lpd_stn run_lpd_p
	python Figure3.py
	python FigureS.4.py


real_data : run_real_data
	python process_real_benchmarks.py


sensitivity : run_sensitivity
	python process_sensitivity.py


run_real_data : real_benchmarks.py
	python real_benchmarks.py abalone
	python real_benchmarks.py athletes
	python real_benchmarks.py bank8FM
	python real_benchmarks.py bodyfat
	python real_benchmarks.py cpu
	python real_benchmarks.py diabetes
	python real_benchmarks.py fishcatch
	python real_benchmarks.py kin8nm
	python real_benchmarks.py liver
	python real_benchmarks.py mu284
	python real_benchmarks.py puma32H
	python real_benchmarks.py puma8NH
	python real_benchmarks.py wisconsin


jun_synthetic_simulations : synthetic_simulation.py
	python synthetic_simulation.py simulation1
	python synthetic_simulation.py simulation2
	python synthetic_simulation.py simulation3
	python synthetic_simulation.py simulation4
	python synthetic_simulation.py simulation5
	python synthetic_simulation.py simulation6
	python synthetic_simulation.py simulation7
	python synthetic_simulation.py friedman1
	python synthetic_simulation.py friedman2
	python synthetic_simulation.py friedman3


run_sensitivity : sensitivity_analysis.py
	python sensitivity_analysis.py simulation1
	python sensitivity_analysis.py simulation2
	python sensitivity_analysis.py simulation3
	python sensitivity_analysis.py simulation4
	python sensitivity_analysis.py simulation5
	python sensitivity_analysis.py simulation6
	python sensitivity_analysis.py simulation7
	python sensitivity_analysis.py friedman1
	python sensitivity_analysis.py friedman2
	python sensitivity_analysis.py friedman3


run_lpd_stn : lpd_simulations.py
	python lpd_simulations.py simulation1 -p 10 -s 0.75
	python lpd_simulations.py simulation1 -p 10 -s 3
	python lpd_simulations.py simulation1 -p 10 -s 1.5
	python lpd_simulations.py simulation1 -p 10 -s 1
	python lpd_simulations.py simulation1 -p 10 -s 0.75

	python lpd_simulations.py simulation2 -p 10 -s 0.75
	python lpd_simulations.py simulation2 -p 10 -s 3
	python lpd_simulations.py simulation2 -p 10 -s 1.5
	python lpd_simulations.py simulation2 -p 10 -s 1
	python lpd_simulations.py simulation2 -p 10 -s 0.75

	python lpd_simulations.py simulation3 -p 10 -s 0.75
	python lpd_simulations.py simulation3 -p 10 -s 3
	python lpd_simulations.py simulation3 -p 10 -s 1.5
	python lpd_simulations.py simulation3 -p 10 -s 1
	python lpd_simulations.py simulation3 -p 10 -s 0.75

	python lpd_simulations.py simulation4 -p 10 -s 0.75
	python lpd_simulations.py simulation4 -p 10 -s 3
	python lpd_simulations.py simulation4 -p 10 -s 1.5
	python lpd_simulations.py simulation4 -p 10 -s 1
	python lpd_simulations.py simulation4 -p 10 -s 0.75


run_lpd_p : lpd_simulations.py
	python lpd_simulations.py simulation1 -p 2 -s 3
	python lpd_simulations.py simulation1 -p 5 -s 3
	python lpd_simulations.py simulation1 -p 10 -s 3
	python lpd_simulations.py simulation1 -p 15 -s 3
	python lpd_simulations.py simulation1 -p 20 -s 3

	python lpd_simulations.py simulation2 -p 2 -s 3
	python lpd_simulations.py simulation2 -p 5 -s 3
	python lpd_simulations.py simulation2 -p 10 -s 3
	python lpd_simulations.py simulation2 -p 15 -s 3
	python lpd_simulations.py simulation2 -p 20 -s 3

	python lpd_simulations.py simulation3 -p 2 -s 3
	python lpd_simulations.py simulation3 -p 5 -s 3
	python lpd_simulations.py simulation3 -p 10 -s 3
	python lpd_simulations.py simulation3 -p 15 -s 3
	python lpd_simulations.py simulation3 -p 20 -s 3

	python lpd_simulations.py simulation4 -p 2 -s 3
	python lpd_simulations.py simulation4 -p 5 -s 3
	python lpd_simulations.py simulation4 -p 10 -s 3
	python lpd_simulations.py simulation4 -p 15 -s 3
	python lpd_simulations.py simulation4 -p 20 -s 3
