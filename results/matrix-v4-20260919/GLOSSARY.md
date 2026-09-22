# Glossary of cell values

Every canonical token that appears in `methodology.csv`,
`solution_shape.csv` and the HTML report. Generated from
`src/autotokamak/bench/glossary.py`, so it cannot drift from the code
that emits these values.

## How the prompt was solved (cross-comparison values)

**`claimed_only`** — The workspace talks about stored-file validation (finite_frac, stored_ok) but no function both loads a file and checks finiteness. The gate is reported, not performed.

**`enforced`** — A pilot runs and its success rate is compared against a threshold before the campaign proceeds — the task's 50% pilot gate.

**`forked_process`** — Solves run in explicitly forked/spawned processes without a pool — the isolation of a pool, managed by hand.

**`hand_built_triangulation`** — A mesh built outside OFT (e.g. scipy.spatial.Delaunay) and handed to the solver. The task forbids this explicitly: TokaMaker validates mesh topology and rejects hand-built triangulations.

**`hdf5`** — One HDF5 file holding many samples. Scales best and carries attributes/provenance, at the price of concurrent-write care.

**`imported_not_subprocessed`** — predict.py is referenced but only imported or described, never run as the documented command. Code that works in the session's process and fails in a clean one is exactly what this misses.

**`in_process_serial`** — Solves run one after another inside the driving process. Simplest, and the slowest: no parallelism, and one bad solve can poison the shared solver state.

**`index`** — An index/manifest (CSV or JSONL) sits beside the arrays, listing every sample and its status. What makes a campaign resumable and auditable rather than a directory to be re-scanned.

**`lcfs_polygon`** — A grid point is inside the plasma iff it lies within the LCFS polygon — point-in-polygon against the D-shape boundary, whatever the helper is called (contains_points, _points_in_poly, inside_lcfs_mask). Geometric, per sample, and independent of the surrogate's own output.

**`manual_barycentric`** — The grid point is located in the mesh by hand (find_simplex / barycentric weights) and psi blended from the triangle's vertices.

**`module_level_once`** — OFT_env is constructed at import or in one top-level block, with no guard. Works while the module is imported once, with nothing stopping a second construction.

**`npy`** — Raw .npy arrays, typically one per field per sample.

**`npz_per_solve`** — One compressed .npz per solved sample.

**`oft_gs_domain`** — Meshed through OFT's own API (gs_Domain / define_region / build_mesh) as the worked examples do. What the task's meshing milestone asks for, and what TokaMaker's topology checks accept.

**`one_dominant_module`** — One file holds most of the pipeline's lines, with small helpers around it.

**`per_solve`** — An OFT_env is constructed inside the per-solve function. Directly against the documented limit if those solves share a process; usually a latent crash rather than a design.

**`per_worker_process`** — Each worker process builds its own OFT_env. The only arrangement that lets the campaign use more than one core, since the limit is per process — and it contains a crash to one worker. Costs a fresh env (and mesh) per worker.

**`pickle`** — Python-pickled objects (pickle/joblib/torch.save) — usually the model artifact rather than the dataset.

**`process_pool`** — multiprocessing.Pool / ProcessPoolExecutor fans solves across processes. Parallel, and each solve is insulated — a segfault kills a worker, not the campaign.

**`psi_norm_threshold`** — Inside/outside is decided by normalised flux (psi_n <= 1). Uses the solve's own notion of the boundary rather than the requested shape; sensitive to how psi_n is normalised.

**`reload_and_check_finite`** — A function both re-loads a written artifact and checks its finite fraction — the task's storage-validation gate, actually implemented. This is the check that catches an all-NaN grid stored as a success.

**`run_without_threshold`** — A pilot runs, but nothing compares its success rate to a threshold: the campaign starts regardless of what the pilot found.

**`scipy_interpolator`** — The solver's nodal values are re-interpolated onto the grid with scipy (RegularGridInterpolator / griddata / CloughTocher). Adds an interpolation error on top of the solve, and depends on how the nodes were sampled first.

**`single_entry_script`** — Exactly one file is runnable as a program.

**`single_script`** — The whole pipeline is one or two files.

**`singleton_reused`** — One OFT_env is built once and handed out on every call — a cached module global, a class guarding construction, or an lru_cache. Honours OFT's one-env-per-process limit in the simplest way, at the cost of keeping every solve in one process: a solver crash takes the campaign with it.

**`solver_field_eval`** — psi is read on the frozen 64x96 grid through the solver's own finite-element interpolator (get_field_eval). Evaluates the FE basis directly, so no second interpolation error is introduced.

**`solver_native_nan`** — Whatever the solver itself leaves undefined outside the plasma is kept as NaN. No separate mask to get wrong — and no mask at prediction time either, unless one is reconstructed.

**`subprocess_per_batch`** — The driver shells out (usually to its own script) to run a batch. Insulates solver state between batches and survives a hard crash; pays process start-up and serialises through files.

**`subprocess_reruns_predict`** — Something in the workspace invokes predict.py as a subprocess with the contract CLI — the task's deliverable self-test, in a fresh process, which is the only way to catch an import-order or global-state dependency.

**`test_absent_from_fitting_functions`** — No function that calls .fit()/backward()/optimizer.step() references a test path or array. Necessary, not sufficient: the task also forbids the test set influencing model SELECTION and ACQUISITION, which this check cannot see.

**`test_referenced_in_fitting_function`** — A function that fits the model also touches something named test. A flag for review, not a verdict — read the cited line; it may be a legitimate final-evaluation call sharing a function.

**`training_valid_mask`** — The predictor masks with a valid-pixel mask derived from the training data (e.g. pixels finite in training). Cheap and stable, but the mask cannot adapt to a geometry unlike those seen in training — the same family of defect as the run that passed 9/9 gates while predicting plasma in vacuum.

## Method and decision vocabulary

**`adaptive_in_name_only`** — Every stated criterion names nothing but randomness: the round was labelled adaptive and was not.

**`agent`** — The meta-loop stopped because the decision policy chose to terminate.

**`agree`** — The criterion stated in the log/report is the one the code computes.

**`autoencoder`** — A learned nonlinear latent space instead of a linear basis.

**`bagging`** — Models trained on bootstrap resamples of the data.

**`budget`** — Stop because a solve or time budget was exhausted.

**`chain_agreement`** — Share of a cell's replicates that chose the modal chain of methods — method reproducibility, which is not the same as score reproducibility.

**`cnn_decoder`** — A convolutional decoder emitting the field as an image.

**`continue`** — Another adaptive round followed this one.

**`criterion_switched`** — The acquisition criterion CHANGED between rounds — logic reacting to what it measured, rather than one fixed rule executed n times.

**`cv_select`** — Selection by cross-validation score.

**`deep_ensemble`** — Several independently initialised models, averaged.

**`early_stopping`** — Training halted on a validation criterion.

**`enrich_active`** — Meta-loop action: acquire new samples ON PURPOSE, by residual-driven UCB where a trained winner exists and PCA-GP variance where it does not. The typed equivalent of an L2/L3 agent's adaptive round.

**`evidence_grounded`** — The round's validation error against baseline was recorded, so its choice can be checked against what was known at the time. An ungrounded round's reasoning is unfalsifiable.

**`extend_search`** — Meta-loop action: spend more effort on model search (a nested Phase-2 run with an emphasis or wider hyperparameters) rather than on more data.

**`feasibility`** — Candidate choice weighted by whether solves there are expected to converge, steering away from regions with failed solves.

**`gp`** — Gaussian-process regression: calibrated uncertainty, cubic scaling.

**`gradient_boosting`** — Boosted trees (sklearn/XGBoost/LightGBM).

**`gradient_sensitivity`** — Points where the response is steep or curving — a sensitivity or Jacobian-driven criterion.

**`grid`** — A full-factorial grid over the parameters.

**`grid_search`** — Exhaustive grid search.

**`halton`** — A Halton low-discrepancy sequence.

**`iterations_cap`** — The meta-loop stopped because it ran out of iterations, not because it had succeeded.

**`kernel_ridge`** — Kernel ridge regression — a GP's mean without its variance.

**`knn`** — k-nearest-neighbour regression.

**`lhs`** — Latin hypercube: stratified in every dimension at once.

**`manual_sweep`** — Hyperparameters fixed or tuned by hand.

**`maximin`** — Points chosen to maximise the minimum pairwise distance.

**`mc_dropout`** — Dropout left active at inference and sampled repeatedly — an ensemble's spread at one model's training cost.

**`mismatch`** — The stated criterion and the implemented one have nothing in common. The strongest available signal that a run's account of itself is wrong.

**`mlp_sklearn`** — sklearn's MLPRegressor.

**`mlp_torch`** — A hand-written PyTorch MLP. The most common choice in this corpus, usually mapping 5 parameters to PCA coefficients.

**`model_informed`** — The function that chooses points actually calls the surrogate. A model-derived criterion that never does is not one, whatever it is named.

**`optuna`** — Optuna search over hyperparameters.

**`partial`** — Stated and implemented criteria overlap but do not match: one side names a component the other never does — see claimed-but-not-computed.

**`pca`** — The psi field is compressed to a handful of principal components and the model predicts the coefficients. The standard field-surrogate move: it turns 6144 outputs into tens, and caps accuracy at whatever the truncated basis can represent.

**`per_pixel`** — The model predicts grid points directly, with no reduction — simple, and the most parameters to fit.

**`plateau`** — Stop because improvement had flattened.

**`pod_svd`** — An SVD/POD basis of the field — the same idea as PCA.

**`poly_ridge`** — Polynomial features into a ridge regressor.

**`prose_only`** — A method named in the README or report.json that the code never evidences.

**`random`** — Points drawn uniformly at random. As the CANDIDATE POOL this is normal and harmless; as the selection criterion it means the round was adaptive in name only.

**`random_draw`** — The batch is drawn at random from the candidates.

**`random_forest`** — A forest of regression trees.

**`random_search`** — Randomised hyperparameter search.

**`rbf_interpolant`** — Radial-basis-function interpolation through the samples.

**`regen_dataset`** — Meta-loop action: regenerate the dataset with new sweep settings (e.g. more samples, a different envelope) — a blind append, not a targeted one.

**`residual_ucb`** — Points are targeted where the model is MEASURABLY wrong — an error model fit on residuals or out-of-fold error, often with a UCB trade-off between exploiting known weakness and exploring. Uses measurement rather than the model's self-assessment, which can be confidently wrong.

**`ridge_linear`** — Plain linear/ridge regression in the input parameters.

**`round_cap`** — Stop because the 3-round cap was reached.

**`sobol`** — A Sobol low-discrepancy sequence — quasi-random, extensible.

**`space_filling`** — Purely geometric coverage of the input box: farthest-point/maximin, distance to the existing training set, k-means. Needs no model at all — which makes it robust, and makes it NOT adaptive in the sense of reacting to what was learned.

**`spline_basis`** — The field is represented by spline or RBF basis functions.

**`stop_threshold_met`** — The campaign ended with the task's 70% criterion satisfied — the intended way to finish early.

**`stop_unexplained`** — The campaign ended and no per-round validation error was recorded, so why it stopped cannot be read from the artifacts at all.

**`stop_without_threshold`** — The campaign ended although validation error had NOT reached 0.30x baseline: the round cap, a budget, or a timeout ended it, not the stopping rule.

**`svr`** — Support-vector regression.

**`target_reached`** — The meta-loop stopped because its accuracy target was met — the intended early finish.

**`terminate`** — Meta-loop action: stop, with a stated reason and confidence.

**`threshold`** — Candidates above a score threshold are taken.

**`top_k`** — The highest-scoring candidates are taken (argsort/topk).

**`typed_decision`** — The criterion came from the L0/L1 typed action schema rather than from free-form agent prose — structurally present every iteration.

**`uncertainty`** — Points are ranked by the model's own predictive spread, without the record saying where that spread comes from. The family term; the two entries below are its specific forms.

**`uncertainty_ensemble`** — Spread across an ENSEMBLE of models (deep ensemble, bootstrap, or MC-dropout samples) — 'query by committee'. Needs no probabilistic model, and its quality depends entirely on the members disagreeing for real rather than sharing an initialisation.

**`uncertainty_gp`** — Posterior variance of a Gaussian process (or an acquisition built on it, e.g. expected improvement). Principled and calibrated where the GP's kernel assumptions hold; costly as the dataset grows.

**`undocumented`** — The code implements a criterion that the log and report never name.

**`uniform_random`** — Independent uniform draws, with no stratification.

**`unverifiable_from_code`** — No function that chooses points could be classified, so the stated criterion can be neither confirmed nor contradicted.

**`val_threshold_70pct`** — The task's own rule: stop once validation error is at most 0.30x the mean-predictor baseline, i.e. a 70% reduction.
